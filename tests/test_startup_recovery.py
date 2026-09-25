"""启动期崩溃恢复：扫描原位修复残留的 .repairbak 并按数据库精确区间回滚。

覆盖：
  * 旧分片改名后、新分片落位前崩溃（正式路径缺失）→ 启动恢复旧内容；
  * 新分片已落位但从未确认（含已提交但客户端未收到响应）→ 仍以旧备份为准；
  * 恢复成功后备份删除；
  * 备份长度与登记区间不符 / rename 失败 → 保留备份并抛错（启动失败）；
  * 已完成/已过期会话的备份不恢复，由定时清理按时长回收；
  * FastAPI lifespan 启动阶段在接收请求前完成恢复；
  * 定时清理永不删除仍需恢复的备份（与 test_repair_chunk 中对应用例互补）。
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import storage
from app.recovery import RepairRecoveryError, recover_pending_repairs


def _create(client: TestClient, data: bytes, name: str = "data.bin"):
    resp = client.post(
        "/api/v1/sessions",
        json={
            "filename": name,
            "total_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "chunk_size": 0,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


def _put(client: TestClient, sid: str, start: int, end: int, content: bytes):
    return client.put(
        f"/api/v1/sessions/{sid}/chunks?start={start}&end={end}",
        content=content,
    )


def _crash_during_repair(
    sid: str, start: int, end: int, old: bytes, *, unconfirmed_new: bytes | None
) -> Path:
    """构造“修复中途进程退出”的磁盘状态：旧分片在备份名下，正式路径按
    ``unconfirmed_new`` 决定为缺失（改名后、落位前崩溃）或落入未确认新文件。"""
    final = storage.part_path(sid, start, end)
    backup = storage.part_repair_backup_path(sid, start, end)
    final.replace(backup)
    if unconfirmed_new is not None:
        final.write_bytes(unconfirmed_new)
    return backup


def _complete_ok(client: TestClient, sid: str, data: bytes) -> None:
    r = client.post(f"/api/v1/sessions/{sid}/complete")
    assert r.status_code == 200, r.text
    dl = client.get(f"/api/v1/sessions/{sid}/download")
    assert dl.status_code == 200
    assert dl.content == data


# ---------------------------------------------------------------------------
# 基本恢复：两类崩溃点都以旧备份为准
# ---------------------------------------------------------------------------


def test_recover_restores_old_chunk_when_final_missing(client: TestClient) -> None:
    data = bytes((i * 5 + 1) % 256 for i in range(1000))
    sid = _create(client, data)
    _put(client, sid, 0, 1000, data).raise_for_status()
    backup = _crash_during_repair(sid, 0, 1000, data, unconfirmed_new=None)

    final = storage.part_path(sid, 0, 1000)
    assert not final.exists()  # 已登记区间缺文件。

    recovered = recover_pending_repairs()
    assert recovered == [(sid, 0, 1000)]
    assert final.read_bytes() == data
    assert not backup.exists()

    # 恢复后会话可正常完成（拼装不再缺文件）。
    _complete_ok(client, sid, data)


def test_recover_old_backup_wins_over_unconfirmed_new_file(
    client: TestClient,
) -> None:
    data = b"old" * 400  # 1200 字节
    sid = _create(client, data)
    _put(client, sid, 0, 1200, data).raise_for_status()
    new_bytes = b"NEW" * 400
    assert len(new_bytes) == 1200
    backup = _crash_during_repair(
        sid, 0, 1200, data, unconfirmed_new=new_bytes
    )

    final = storage.part_path(sid, 0, 1200)
    assert final.read_bytes() == new_bytes

    recovered = recover_pending_repairs()
    assert recovered == [(sid, 0, 1200)]
    # 即使同路径已落入未确认的新文件，也以旧备份为准覆盖之。
    assert final.read_bytes() == data
    assert not backup.exists()
    _complete_ok(client, sid, data)


def test_recover_multiple_sessions_and_chunks(client: TestClient) -> None:
    data1 = b"aaaa" * 250
    sid1 = _create(client, data1)
    _put(client, sid1, 0, 1000, data1[:1000]).raise_for_status()

    data2 = b"bbbb" * 250
    sid2 = _create(client, data2)
    _put(client, sid2, 0, 1000, data2[:1000]).raise_for_status()

    b1 = _crash_during_repair(sid1, 0, 1000, data1[:1000], unconfirmed_new=b"1" * 1000)
    b2 = _crash_during_repair(sid2, 0, 1000, data2[:1000], unconfirmed_new=None)

    recovered = sorted(recover_pending_repairs())
    assert recovered == sorted([(sid1, 0, 1000), (sid2, 0, 1000)])
    assert storage.part_path(sid1, 0, 1000).read_bytes() == data1[:1000]
    assert storage.part_path(sid2, 0, 1000).read_bytes() == data2[:1000]
    assert not b1.exists() and not b2.exists()


def test_recover_idempotent_after_success(client: TestClient) -> None:
    data = b"z" * 300
    sid = _create(client, data)
    _put(client, sid, 0, 300, data).raise_for_status()
    _crash_during_repair(sid, 0, 300, data, unconfirmed_new=b"q" * 300)

    assert recover_pending_repairs() == [(sid, 0, 300)]
    # 备份已删除：再次恢复为空操作。
    assert recover_pending_repairs() == []
    assert storage.part_path(sid, 0, 300).read_bytes() == data


# ---------------------------------------------------------------------------
# 非上传中会话的备份不恢复
# ---------------------------------------------------------------------------


def test_recover_ignores_backups_of_completed_sessions(client: TestClient) -> None:
    data = b"done" * 100
    sid = _create(client, data)
    _put(client, sid, 0, 400, data).raise_for_status()
    _complete_ok(client, sid, data)

    # 完成后分片目录已删；手动放一个残留备份，恢复必须跳过它。
    d = storage.parts_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    backup = d / ".0-400.repairbak"
    backup.write_bytes(b"stale")

    assert recover_pending_repairs() == []
    assert backup.exists()


def test_recover_ignores_backups_of_expired_sessions(
    client: TestClient, monkeypatch
) -> None:
    from app import db

    data = b"e" * 200
    sid = _create(client, data)
    _put(client, sid, 0, 200, data).raise_for_status()
    backup = _crash_during_repair(sid, 0, 200, data, unconfirmed_new=None)

    # 直接把库状态置为 expired（不删目录）：过期会话的备份不再恢复。
    with db.immediate() as tx:
        tx.execute(
            "UPDATE upload_sessions SET status = 'expired' WHERE id = ?", (sid,)
        )

    assert recover_pending_repairs() == []
    assert backup.exists()
    assert not storage.part_path(sid, 0, 200).exists()


# ---------------------------------------------------------------------------
# 恢复失败：保留备份并报错（启动必须失败）
# ---------------------------------------------------------------------------


def test_recover_corrupt_backup_keeps_backup_and_raises(client: TestClient) -> None:
    data = b"good" * 250  # 1000 字节
    sid = _create(client, data)
    _put(client, sid, 0, 1000, data).raise_for_status()
    backup = _crash_during_repair(
        sid, 0, 1000, data, unconfirmed_new=b"?" * 1000
    )
    # 备份内容被截断（与登记区间长度 1000 不符）。
    backup.write_bytes(b"short")

    final = storage.part_path(sid, 0, 1000)
    with pytest.raises(RepairRecoveryError, match="size mismatch"):
        recover_pending_repairs()
    # 备份保留、未确认新文件不被破坏，等待下次启动/人工介入。
    assert backup.exists()
    assert final.read_bytes() == b"?" * 1000

    # 修好备份后再次启动即恢复成功。
    backup.write_bytes(data)
    assert recover_pending_repairs() == [(sid, 0, 1000)]
    assert final.read_bytes() == data
    assert not backup.exists()


def test_recover_rename_failure_keeps_backup_and_raises(
    client: TestClient, monkeypatch
) -> None:
    data = b"r" * 500
    sid = _create(client, data)
    _put(client, sid, 0, 500, data).raise_for_status()
    backup = _crash_during_repair(
        sid, 0, 500, data, unconfirmed_new=b"n" * 500
    )
    final = storage.part_path(sid, 0, 500)

    real_replace = Path.replace

    def failing_replace(self, target, *args, **kwargs):
        if self == backup and target == final:
            raise OSError("simulated rename failure")
        return real_replace(self, target, *args, **kwargs)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(RepairRecoveryError):
        recover_pending_repairs()
    monkeypatch.undo()

    # 备份原样保留；unconfirmed 新文件在 rename 前已被删除（同目录 rename
    # 原子保证，实际故障窗口内正式路径要么是新文件要么是旧备份，不会并存），
    # 旧内容唯一来源是保留下来的备份。
    assert backup.exists()
    assert backup.read_bytes() == data
    assert not final.exists()

    # 故障消除后再次启动恢复成功。
    assert recover_pending_repairs() == [(sid, 0, 500)]
    assert final.read_bytes() == data
    assert not backup.exists()


def test_recover_orphan_backup_without_registered_chunk_raises(
    client: TestClient,
) -> None:
    """uploading 会话存在，但 chunks 表无对应精确区间：无法判定，报错中止启动。"""
    data = b"o" * 1000
    sid = _create(client, data)
    _put(client, sid, 0, 1000, data).raise_for_status()

    d = storage.parts_dir(sid)
    orphan = d / ".100-200.repairbak"
    orphan.write_bytes(b"x" * 100)

    with pytest.raises(RepairRecoveryError, match="no matching registered chunk"):
        recover_pending_repairs()
    assert orphan.exists()
    # 清理人为构造的孤儿备份，避免污染共享测试数据目录。
    orphan.unlink()


# ---------------------------------------------------------------------------
# 定时清理保护仍需恢复的备份
# ---------------------------------------------------------------------------


def test_cleanup_does_not_delete_backup_pending_recovery(
    client: TestClient,
) -> None:
    from app.cleanup import run_cleanup

    data = b"c" * 400
    sid = _create(client, data)
    _put(client, sid, 0, 400, data).raise_for_status()
    backup = _crash_during_repair(
        sid, 0, 400, data, unconfirmed_new=b"m" * 400
    )
    # 即使备份已远超 1 小时清理阈值，且多轮清理执行……
    old_ts = time.time() - 24 * 3600
    os.utime(backup, (old_ts, old_ts))
    run_cleanup()
    run_cleanup()
    assert backup.exists()

    # 启动恢复后备份删除，此后清理自然无物可删。
    assert recover_pending_repairs() == [(sid, 0, 400)]
    assert not backup.exists()
    run_cleanup()
    assert storage.part_path(sid, 0, 400).read_bytes() == data


# ---------------------------------------------------------------------------
# lifespan：服务开始接收请求前完成恢复
# ---------------------------------------------------------------------------


def test_lifespan_recovers_backups_before_serving(client: TestClient) -> None:
    import anyio

    from app.main import app

    data = b"L" * 600
    sid = _create(client, data)
    _put(client, sid, 0, 600, data).raise_for_status()
    backup = _crash_during_repair(
        sid, 0, 600, data, unconfirmed_new=b"U" * 600
    )
    final = storage.part_path(sid, 0, 600)
    assert final.read_bytes() == b"U" * 600

    # 直接进入一次应用 lifespan（等价于新进程启动）：启动阶段必须完成恢复。
    async def _run() -> None:
        async with app.router.lifespan_context(app):
            pass

    anyio.run(_run)

    assert final.read_bytes() == data
    assert not backup.exists()
    # 服务可用后整文件校验通过。
    _complete_ok(client, sid, data)
