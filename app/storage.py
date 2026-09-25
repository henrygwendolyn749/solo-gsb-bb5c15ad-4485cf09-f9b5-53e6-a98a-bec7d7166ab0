"""文件系统布局、区间算法与拼装工具。"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat as stat_module
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from . import settings


# ---------------------------------------------------------------------------
# 时间：统一 UTC，ISO-8601 带 Z 后缀，字符串可直接做字典序比较。
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def future_expiry() -> tuple[str, str]:
    """返回 (当前时间, 过期时间) 的 ISO 字符串。"""
    now = utcnow()
    return iso(now), iso(now + timedelta(seconds=settings.ttl_seconds))


# ---------------------------------------------------------------------------
# 目录布局
#
#   DATA_DIR/
#     db 文件（默认 resumable.db*）
#     files/<session_id>                完成后的成品（原子 rename 落地）
#     parts/<session_id>/<start>-<end>  已收分片
#     tmp/<随机名>                       正在接收/拼装的临时文件
#
# tmp 下的临时文件从创建起即登记到 active_tempfiles：只要登记还在（请求
# 仍存活，哪怕客户端长时间停顿、mtime 已超过一小时），定时清理就绝不删除；
# 提交落位或失败清理时在同一把登记锁内摘除。进程退出后登记表随之消失，
# 遗留文件再由定时清理按 mtime 兜底回收（见 ActiveTempFiles）。
# ---------------------------------------------------------------------------

def files_dir() -> Path:
    return settings.data_dir / "files"


def parts_dir(session_id: str) -> Path:
    return settings.data_dir / "parts" / session_id


def part_path(session_id: str, start: int, end: int) -> Path:
    return parts_dir(session_id) / f"{start}-{end}"


def part_repair_backup_path(session_id: str, start: int, end: int) -> Path:
    """已收分片原位修复期间旧分片的同目录备份路径。

    与分片成品位于同一目录（同一文件系统），因此新旧分片互换只是
    目录内 rename，全程原子；会话锁保证同一会话同时只有一个修复，
    固定名称不会冲突。它以点号开头，拼装时只按 chunks 表取
    ``<start>-<end>`` 正式分片，备份永远不会被读入成品。
    """
    return parts_dir(session_id) / f".{start}-{end}.repairbak"


_REPAIR_BACKUP_RE = re.compile(r"\.(\d+)-(\d+)\.repairbak$")


def parse_repair_backup(path: Path) -> tuple[str, int, int] | None:
    """把 ``.repairbak`` 备份路径解析为 (session_id, start, end)。

    只接受本模块生成的固定形态 ``parts/<session_id>/.<start>-<end>.repairbak``；
    无法按形态解析（例如其他点文件）返回 None，由调用方忽略。
    """
    match = _REPAIR_BACKUP_RE.search(path.name)
    if match is None:
        return None
    session_id = path.parent.name
    if not session_id or session_id in (".", ".."):
        return None
    return session_id, int(match.group(1)), int(match.group(2))


def iter_repair_backups() -> list[Path]:
    """列出磁盘上所有修复备份文件（``parts/<session_id>/.*.repairbak``）。"""
    parts_root = settings.data_dir / "parts"
    if not parts_root.exists():
        return []
    return [p for p in parts_root.glob("*/.*.repairbak") if p.is_file()]


def artifact_path(session_id: str) -> Path:
    return files_dir() / session_id


def tmp_dir() -> Path:
    return settings.data_dir / "tmp"


def ensure_dirs() -> None:
    for d in (files_dir(), settings.data_dir / "parts", tmp_dir()):
        d.mkdir(parents=True, exist_ok=True)


def new_temp_file() -> Path:
    """在 tmp 目录下创建一个新的临时文件并登记活动保护（已存在、已关闭）。

    创建与登记在同一把活动登记锁内完成，因此文件一旦对清理线程可见，
    就必然处于受保护状态，不存在“先被清理看到、后登记”的窗口。返回路径
    的所有权归调用方：用完（落位 rename 或失败 unlink）后必须调用
    :meth:`active_tempfiles.discard <ActiveTempFiles.discard>`
    （通常经 :func:`receive_temp_file` 上下文管理器自动完成）。
    """
    return active_tempfiles.create()


def remove_session_tree(session_id: str) -> None:
    """删除会话的分片目录与成品（若存在），用于放弃/过期清理。"""
    shutil.rmtree(parts_dir(session_id), ignore_errors=True)
    artifact = artifact_path(session_id)
    try:
        artifact.unlink()
    except FileNotFoundError:
        pass


class ActiveTempFiles:
    """tmp 接收文件的进程内活动登记表。

    背景：PUT 分片与原位修复都先在 tmp 流式落接收文件，弱网客户端可能
    长时间停顿——仅按 mtime（一小时前）清理会把**仍在接收**的文件 unlink，
    随后提交落位必然失败。登记表是权威的“活动期”判据：

    * ``create`` 在同一把锁内创建文件并登记：登记期间（请求全程存活，
      哪怕停顿超过一小时）文件一律受保护，清理跳过；
    * ``sweep`` 与登记共用这把锁，先在锁内判定再删除，判定与删除原子，
      不会出现“刚判为闲置、马上被新请求使用”的 TOCTOU；
    * 提交落位/失败清理后在同一把锁内 ``discard``，异常路径由
      :func:`receive_temp_file` 的 finally 兜底摘除；
    * 登记表仅存在于进程内存：进程退出后遗留文件不再有任何登记，重启后
      仍按 mtime 阈值被清理回收。

    本服务按单 worker 部署（见 Dockerfile 注释，会话锁同样是进程内对象），
    进程内登记与清理线程天然共享同一张表。
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._active: set[Path] = set()

    def create(self) -> Path:
        """持锁创建临时文件并登记，返回其路径。"""
        with self._guard:
            fd, name = tempfile.mkstemp(dir=tmp_dir(), prefix="recv-", suffix=".part")
            os.close(fd)
            path = Path(name)
            self._active.add(path)
            return path

    def discard(self, path: str | Path) -> None:
        """摘除活动登记（落位 rename 之后或失败 unlink 之后）。

        幂等：文件已不在表中（例如重复清理）时无副作用。
        """
        with self._guard:
            self._active.discard(Path(path))

    def active_count(self) -> int:
        """当前受保护的临时文件数（供测试与运维诊断）。"""
        with self._guard:
            return len(self._active)

    def snapshot(self) -> frozenset[Path]:
        """当前受保护临时文件路径的快照（供测试与运维诊断）。"""
        with self._guard:
            return frozenset(self._active)

    def sweep(self, older_than: datetime) -> int:
        """删除早于给定时间且当前不受保护的临时文件，返回删除条数。

        扫描、活动判定与删除全程持有登记锁：与 ``create``/``discard``
        串行，清理看到的“未登记”状态不可能在 unlink 前被任何请求改成
        “活动中”。受保护文件一律跳过（与其 mtime 无关）；只有未登记
        （无主/进程遗留）且超时的文件才删除。
        """
        removed = 0
        with self._guard:
            for entry in tmp_dir().glob("*"):
                if entry in self._active:
                    # 仍在接收/拼装：请求存活即受保护，停顿多久都不删。
                    continue
                try:
                    st = entry.stat()
                except FileNotFoundError:
                    continue
                if not stat_module.S_ISREG(st.st_mode):
                    continue
                mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                if mtime >= older_than:
                    continue
                try:
                    entry.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
        return removed


active_tempfiles = ActiveTempFiles()


@contextmanager
def receive_temp_file() -> Iterator[Path]:
    """创建受活动期保护的接收临时文件，退出作用域时保证摘除保护。

    成功路径：调用方把临时文件 rename 落位（原路径随即消失），退出时
    只摘除登记；失败路径：调用方 unlink 临时文件，退出时同样摘除登记。
    任何异常（含客户端断连、请求取消、提交线程抛出）都会走到 finally，
    不会留下“永远受保护”的幽灵登记。
    """
    tmp = new_temp_file()
    try:
        yield tmp
    finally:
        active_tempfiles.discard(tmp)


def sweep_old_tempfiles(older_than: datetime) -> int:
    """删除早于给定时间且没有活动请求持有的临时文件（崩溃残留兜底）。

    活动判定与删除在同一把登记锁内完成（见 :class:`ActiveTempFiles`）。
    """
    return active_tempfiles.sweep(older_than)


def sweep_repair_backups(
    older_than: datetime, protected: Iterable[Path] = ()
) -> None:
    """删除分片目录中崩溃残留的修复备份（.repairbak，正常路径会立即回收）。

    备份文件不参与拼装（拼装只按 chunks 表取正式分片名），残留无害；
    这里只负责回收磁盘空间。``protected`` 中的备份（经数据库核对仍需
    在启动时恢复、或属于正在上传会话的备份）**一律跳过，无视存活时长**，
    定时清理绝不能删掉恢复所需的旧分片内容。其余备份只可能来自已结束
    （完成/放弃/过期）会话或进行中修复——后者持有会话锁且持续时间以秒计，
    按存活时长过滤即可；前者则在等待人工/下轮启动恢复期间继续保留。
    """
    protected_set = {Path(p) for p in protected}
    for entry in iter_repair_backups():
        if entry in protected_set:
            continue
        try:
            mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
        except FileNotFoundError:
            continue
        if mtime < older_than:
            try:
                entry.unlink()
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# 区间工具：区间一律采用半开 [start, end)。
# ---------------------------------------------------------------------------

def merge_intervals(intervals: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    """把可能相邻但不重叠的区间合并为连续区间列表。"""
    ordered = sorted(((int(s), int(e)) for s, e in intervals), key=lambda x: x[0])
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def missing_intervals(
    received: Iterable[Sequence[int]], total_size: int
) -> list[tuple[int, int]]:
    """根据已收区间计算 [0, total_size) 内的缺失区间。"""
    missing: list[tuple[int, int]] = []
    cursor = 0
    for start, end in merge_intervals(received):
        if start > cursor:
            missing.append((cursor, start))
        if end > cursor:
            cursor = end
    if cursor < total_size:
        missing.append((cursor, total_size))
    return missing


def overlaps_any(
    start: int,
    end: int,
    existing: Iterable[Sequence[int]],
) -> bool:
    """半开区间相交判定：start < other.end 且 other.start < end。"""
    return any(start < o_end and o_start < end for o_start, o_end in existing)


# ---------------------------------------------------------------------------
# 文件名与拼装
# ---------------------------------------------------------------------------

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\-一-鿿]+")


def safe_filename(name: str) -> str:
    """去掉路径分量与危险字符，只保留安全的文件名部分。"""
    name = name.replace("\\", "/").split("/")[-1]
    name = name.lstrip(".")
    cleaned = _SAFE_NAME.sub("_", name).strip("._ ")
    return cleaned or "file"


def assemble_artifact(
    session_id: str, chunks: Sequence[tuple[int, int]], total_size: int
) -> tuple[str, str]:
    """按偏移顺序把分片拼装成成品。

    成品先写到 tmp 下的临时文件，再由调用方做 SHA-256 校验；
    校验通过后由调用方原子 rename 到 files 目录。返回临时文件路径。

    每个分片在读取前再次核对磁盘大小，防止文件被外部破坏。
    """
    tmp_out = new_temp_file()
    seen = 0
    sha = hashlib.sha256()
    try:
        with tmp_out.open("wb") as out:
            for start, end in chunks:
                if start != seen:
                    raise ValueError(f"chunk gap at offset {seen}, chunk starts at {start}")
                src = part_path(session_id, start, end)
                size = src.stat().st_size
                if size != end - start:
                    raise ValueError(f"chunk {start}-{end} size on disk is {size}")
                with src.open("rb") as fh:
                    while True:
                        block = fh.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                        sha.update(block)
                seen = end
        if seen != total_size:
            raise ValueError(f"assembled {seen} bytes, expected {total_size}")
    except Exception:
        tmp_out.unlink(missing_ok=True)
        active_tempfiles.discard(tmp_out)
        raise
    # 成功：路径与活动登记一并转交给调用方（complete 事务），
    # 由其在发布或失败清理后 discard。
    return str(tmp_out), sha.hexdigest()


def hash_part_file(session_id: str, start: int, end: int) -> tuple[int, str] | None:
    """流式读取已登记分片文件，返回 (实际字节数, SHA-256)；文件缺失返回 None。

    纯只读，供分片完整性诊断使用：字节数按实际读取统计（不经 stat），
    读取与长度核对之间不存在竞态。调用方必须持有会话锁，保证读到的是
    稳定状态（修复/完成/放弃都无法在中途插入）。
    """
    path = part_path(session_id, start, end)
    sha = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as fh:
            while True:
                block = fh.read(1024 * 1024)
                if not block:
                    break
                sha.update(block)
                size += len(block)
    except FileNotFoundError:
        return None
    return size, sha.hexdigest()


def hash_artifact_file(session_id: str) -> tuple[int, str] | None:
    """流式读取已完成会话的成品文件，返回 (实际字节数, SHA-256)；缺失返回 None。

    与 :func:`hash_part_file` 同一口径：字节数按实际读取统计（不经 stat），
    纯只读，供成品完整性巡检使用。调用方必须持有会话锁，保证读取期间
    成品不会被删除会话流程清理。
    """
    path = artifact_path(session_id)
    sha = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as fh:
            while True:
                block = fh.read(1024 * 1024)
                if not block:
                    break
                sha.update(block)
                size += len(block)
    except FileNotFoundError:
        return None
    return size, sha.hexdigest()


def atomic_publish(tmp_file: str, session_id: str) -> None:
    """把拼装好的临时文件原子替换为成品文件。"""
    os.replace(tmp_file, artifact_path(session_id))


# ---------------------------------------------------------------------------
# 跨会话复制：从已完成会话的成品中切出一段，流式写入临时文件
# ---------------------------------------------------------------------------


def copy_artifact_segment(
    source_session_id: str,
    start: int,
    end: int,
) -> str:
    """把成品文件的 [start, end) 字节流式复制为 tmp 下的新临时文件。

    仅读取已完成会话的成品（files/<id>），不接触分片目录。
    返回临时文件路径；读取前核对区间落在文件大小内，读完核对字节数，
    成品被并发截断/替换会在此暴露为 OSError/ValueError，绝不产生短文件。
    任何失败都清理刚创建的临时文件。
    """
    src = artifact_path(source_session_id)
    length = end - start
    tmp_out = new_temp_file()
    try:
        with tmp_out.open("wb") as out, src.open("rb") as fh:
            actual_size = fh.seek(0, os.SEEK_END)
            if end > actual_size:
                raise ValueError(
                    f"source range [{start},{end}) exceeds artifact size {actual_size}"
                )
            fh.seek(start)
            remaining = length
            while remaining > 0:
                block = fh.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError(
                        f"source artifact ended at {length - remaining} of {length} bytes"
                    )
                out.write(block)
                remaining -= len(block)
    except BaseException:
        tmp_out.unlink(missing_ok=True)
        active_tempfiles.discard(tmp_out)
        raise
    # 成功：路径与活动登记转交给调用方（copy-chunk 提交），
    # 由其在落位或失败清理后 discard。
    return str(tmp_out)
