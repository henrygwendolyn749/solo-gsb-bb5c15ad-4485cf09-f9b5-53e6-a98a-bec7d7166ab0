"""启动期崩溃恢复：原位修复备份（.repairbak）的扫描与回滚。

原位修复（PUT /chunks/repair）在会话锁内按
“旧分片改名 .repairbak → 新分片原子落位 → 提交事务 → 删除备份”
的顺序执行。若进程在旧分片改名之后、备份删除之前退出（无论事务是否
已经提交），磁盘上会残留备份，且可能出现两种不一致：

1. 正式分片路径缺失，而数据库 chunks 仍登记该精确区间——已登记区间
   缺文件，拼装时才暴露失败；
2. 正式分片路径落入了修复的新内容，但新内容从未向客户端确认
   （事务回滚/未提交，或提交后还来不及返回响应进程就退出）。

服务在**开始接收请求之前**统一处理：扫描所有修复备份，按数据库核对
精确分片区间，仅恢复仍处于 uploading 会话的备份——在会话锁内以旧备份
为准原子改回正式路径，**即使同路径已经落入新文件也覆盖之**；恢复成功
后才删除备份。恢复失败（备份损坏、长度不符、磁盘错误）则保留备份并
抛出异常使启动失败（fail-closed），绝不带错对外服务。

备份文件名只携带区间，不携带修复结果，因此无法区分“事务已提交的新
内容”与“未确认的新内容”：一律以旧备份为准。已确认的修复若恰好在
提交后崩溃，客户端重发同一 repair 即可重新得到新内容。
"""
from __future__ import annotations

import logging
from pathlib import Path

from . import db, storage
from .locks import session_locks

logger = logging.getLogger("resumable.recovery")


class RepairRecoveryError(RuntimeError):
    """启动期修复备份恢复失败：服务必须中止启动。"""


def pending_recovery_backups() -> set[Path]:
    """返回仍需保留/恢复的修复备份路径集合。

    判定标准（纯数据库核对，不触碰磁盘内容）：备份文件名能解析出
    (session_id, start, end)，该会话状态为 uploading，且 chunks 表中
    存在完全一致的精确分片区间。其余备份（已完成/放弃/过期会话、
    会话或区间已不存在、文件名无法解析）不在集合内。

    后台定时清理用它保护备份：集合内的备份无论多旧都不得删除。
    """
    protected: set[Path] = set()
    conn = db.get_conn()
    for backup in storage.iter_repair_backups():
        parsed = storage.parse_repair_backup(backup)
        if parsed is None:
            continue
        session_id, start, end = parsed
        row = conn.execute(
            "SELECT status FROM upload_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None or row["status"] != "uploading":
            continue
        exact = conn.execute(
            "SELECT 1 FROM chunks "
            "WHERE session_id = ? AND start_offset = ? AND end_offset = ?",
            (session_id, start, end),
        ).fetchone()
        if exact is not None:
            protected.add(backup)
    return protected


def _restore_one(
    session_id: str, start: int, end: int, backup: Path, size: int
) -> None:
    """在会话锁内把单个旧备份恢复为正式分片；失败保留备份并抛出。"""
    final = storage.part_path(session_id, start, end)
    with session_locks.get(session_id):
        # 锁内按数据库再次核对：恢复期间没有任何请求在处理（启动期），
        # 锁是为了与后台清理/懒过期路径串行，并复核精确区间仍然登记。
        row = db.get_conn().execute(
            "SELECT status FROM upload_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None or row["status"] != "uploading":
            raise RepairRecoveryError(
                f"repair backup session {session_id} is no longer uploading "
                f"({None if row is None else row['status']})"
            )
        exact = db.get_conn().execute(
            "SELECT 1 FROM chunks "
            "WHERE session_id = ? AND start_offset = ? AND end_offset = ?",
            (session_id, start, end),
        ).fetchone()
        if exact is None:
            raise RepairRecoveryError(
                f"repair backup {backup} has no matching registered chunk "
                f"[{start},{end}) in session {session_id}"
            )

        # 恢复前先核对旧备份自身完整：长度必须等于登记区间长度。
        # 备份损坏时原样保留，绝不用短文件顶替正式分片。
        try:
            backup_size = backup.stat().st_size
        except OSError as exc:
            raise RepairRecoveryError(
                f"cannot stat repair backup {backup}: {exc}"
            ) from exc
        if backup_size != size:
            raise RepairRecoveryError(
                f"repair backup {backup} size mismatch: "
                f"disk {backup_size}, registered {size}"
            )

        # 以旧备份为准：正式路径即使已落入未确认的新文件也先删除，
        # 再把备份同目录原子改名回来。同卷 rename 原子，任何瞬间要么
        # 是新文件、要么是旧备份，不存在半成品。任何文件系统错误都
        # 保留备份并转为启动失败，绝不在备份/正式分片双缺时继续启动。
        try:
            final.parent.mkdir(parents=True, exist_ok=True)
            final.unlink(missing_ok=True)
            backup.replace(final)
        except OSError as exc:
            raise RepairRecoveryError(
                f"failed to restore repair backup {backup} -> {final}: {exc}"
            ) from exc

        # 恢复后复核落位结果；不符则回滚（把内容移回备份名）并报错，
        # 备份保留到下次启动再试，绝不留下“已恢复”的假状态。
        try:
            restored_size = final.stat().st_size
        except OSError as exc:
            raise RepairRecoveryError(
                f"restored chunk {final} unreadable after recovery: {exc}"
            ) from exc
        if restored_size != size:
            try:
                final.replace(backup)
            except OSError:
                pass
            raise RepairRecoveryError(
                f"restored chunk {session_id} [{start},{end}) has size "
                f"{restored_size}, expected {size}; backup kept at {backup}"
            )

    # 会话锁外删除已完成使命的备份；删除失败不影响正确性（下轮清理
    # 会看到正式分片已在位、备份仍受保护而跳过），但记日志以便排查。
    backup.unlink(missing_ok=True)


def recover_pending_repairs() -> list[tuple[str, int, int]]:
    """启动期扫描并恢复所有未完成的原位修复。

    在服务开始接收请求前调用（FastAPI lifespan 启动阶段）。返回成功
    恢复的 (session_id, start, end) 列表。任一备份恢复失败即抛出
    :class:`RepairRecoveryError`，已恢复的保持有效、未恢复的备份原样
    保留，调用方应中止启动。
    """
    recovered: list[tuple[str, int, int]] = []
    conn = db.get_conn()
    for backup in storage.iter_repair_backups():
        parsed = storage.parse_repair_backup(backup)
        if parsed is None:
            # 非本服务形态的点文件不属于修复备份，交给临时清理逻辑。
            continue
        session_id, start, end = parsed
        row = conn.execute(
            "SELECT status FROM upload_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        # 只恢复上传中会话的备份；终态会话的目录本应由主流程删除，
        # 残留备份留给定时清理按存活时长回收。
        if row is None or row["status"] != "uploading":
            continue
        chunk = conn.execute(
            "SELECT size FROM chunks "
            "WHERE session_id = ? AND start_offset = ? AND end_offset = ?",
            (session_id, start, end),
        ).fetchone()
        # 上传中会话却找不到精确登记区间：数据库与备份无法对应，
        # 不能猜测处理，保留备份并中止启动等待人工介入。
        if chunk is None:
            raise RepairRecoveryError(
                f"repair backup {backup} has no matching registered chunk "
                f"[{start},{end}) for uploading session {session_id}"
            )
        _restore_one(session_id, start, end, backup, chunk["size"])
        recovered.append((session_id, start, end))
        logger.info(
            "recovered interrupted chunk repair: session=%s range=[%d,%d)",
            session_id, start, end,
        )
    return recovered
