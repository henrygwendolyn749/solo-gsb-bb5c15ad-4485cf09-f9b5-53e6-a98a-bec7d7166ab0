"""接收临时文件的活动期保护（慢速分片上传 / 原位修复）。

PUT 分片与原位修复都先在 tmp/ 流式落接收文件。定时清理按“一小时前
mtime”删除遗留文件时，不得误删**仍在接收**（哪怕客户端长时间停顿）
的临时文件，否则随后 replace 落位必然失败。

规则：
  * 临时文件从 create（与登记同一把锁原子完成）到提交落位/失败清理
    全程登记在 active_tempfiles，登记期间清理一律跳过；
  * 清理的“活动判定 → unlink”与登记/摘除共用同一把锁，不存在刚判为
    闲置就被重新使用的竞态（接收文件名随机唯一，不会复用旧名）；
  * 任何异常路径（客户端断连、长度/摘要不符、提交失败）都摘除登记，
    不留幽灵保护；
  * 进程退出后登记表消失：重启后遗留的旧 mtime 文件照常回收。
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app import storage


@pytest.fixture(autouse=True)
def _ensure_tmp_dir() -> None:
    """纯单元测试不经过 lifespan，先确保 tmp 目录存在。"""
    storage.ensure_dirs()


def _old_cutoff() -> storage.datetime:
    # 清理阈值：比现在早一小时，与 cleanup.run_cleanup 的口径一致。
    return storage.utcnow() - timedelta(hours=1)


def _make_orphan_old_file() -> storage.Path:
    """模拟进程退出遗留：在 tmp 直接造一个旧 mtime 的无登记文件。"""
    path = storage.tmp_dir() / "recv-orphan-leftover.part"
    path.write_bytes(b"leftover")
    old = time.time() - 2 * 3600
    os.utime(path, (old, old))
    return path


# ---------------------------------------------------------------------------
# 登记表与清理的单元语义
# ---------------------------------------------------------------------------


def test_registered_tempfile_is_never_swept_regardless_of_mtime() -> None:
    with storage.receive_temp_file() as tmp:
        tmp.write_bytes(b"receiving")
        # 接收停顿超过一小时：mtime 已老于清理阈值，但仍登记在案。
        old = time.time() - 2 * 3600
        os.utime(tmp, (old, old))

        removed = storage.sweep_old_tempfiles(_old_cutoff())
        assert removed == 0
        assert tmp.exists()
        assert tmp.read_bytes() == b"receiving"
        assert storage.active_tempfiles.active_count() == 1
    # 作用域退出后登记摘除；文件由调用方负责删除（此处模拟落位/清理）。
    tmp.unlink(missing_ok=True)
    assert storage.active_tempfiles.active_count() == 0


def test_unregistered_stale_tempfile_is_reaped() -> None:
    leftover = _make_orphan_old_file()
    removed = storage.sweep_old_tempfiles(_old_cutoff())
    assert removed == 1
    assert not leftover.exists()


def test_unregistered_recent_tempfile_is_kept() -> None:
    # 无登记但 mtime 很新的文件不删（阈值留一小时宽限）。
    path = storage.tmp_dir() / "recv-recent.part"
    path.write_bytes(b"recent")
    try:
        assert storage.sweep_old_tempfiles(_old_cutoff()) == 0
        assert path.exists()
    finally:
        path.unlink(missing_ok=True)


def test_sweep_and_registration_are_serialized(monkeypatch) -> None:
    """清理持锁扫描/删除期间，新文件不可能在“判闲置之后、unlink 之前”登记。

    以线程同步复现竞态点：清理线程进入 tmp 扫描后人为阻塞（仍持有登记
    锁）；另一线程试图创建并登记接收文件。由于创建/登记与清理共用同一把
    锁，它必须排队到整轮清理结束——而那时删除判定早已完成，绝不会删到
    刚登记的活动文件。
    """
    leftover = _make_orphan_old_file()
    entered = threading.Event()
    release = threading.Event()

    real_glob = storage.Path.glob

    def gated_glob(self, pattern):
        if self == storage.tmp_dir() and pattern == "*":
            entered.set()
            release.wait(timeout=5)
        return real_glob(self, pattern)

    monkeypatch.setattr(storage.Path, "glob", gated_glob)

    t = threading.Thread(target=storage.sweep_old_tempfiles, args=(_old_cutoff(),))
    created = threading.Event()
    t.start()
    assert entered.wait(timeout=5)
    # 清理已进入临界区并持锁：此后创建+登记必须排队，不能插入。
    r = threading.Thread(
        target=_register_aged_file_while, args=(release, created)
    )
    r.start()
    try:
        time.sleep(0.1)
        assert not created.is_set()
    finally:
        release.set()
        t.join(timeout=5)
        r.join(timeout=5)

    assert not t.is_alive()
    assert not r.is_alive()
    assert not leftover.exists()


def _register_aged_file_while(release: threading.Event, created: threading.Event) -> None:
    with storage.receive_temp_file() as tmp:
        tmp.write_bytes(b"alive")
        old = time.time() - 2 * 3600
        os.utime(tmp, (old, old))
        created.set()
        # 持有登记直到清理线程释放锁并跑完删除，模拟停顿中的请求。
        release.wait(timeout=5)
        assert tmp.exists()
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 端到端：慢速流式接收期间跑清理，分片仍能正常落位
# ---------------------------------------------------------------------------


def _create_session(client: TestClient, data: bytes) -> str:
    resp = client.post(
        "/api/v1/sessions",
        json={
            "filename": "data.bin",
            "total_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


def _slow_stream(chunks: list[bytes], after_first_block):
    """逐块产出请求体；首块后执行钩子（跑清理），模拟接收停顿超过一小时。"""
    for i, chunk in enumerate(chunks):
        if i == 1:
            after_first_block()
        yield chunk


def _age_active_tmp_behind_cutoff() -> None:
    """把当前唯一活动接收文件的 mtime 拨到两小时前（停顿很久）。"""
    active = storage.active_tempfiles.snapshot()
    assert len(active) == 1, f"expected exactly one protected recv file, got {active}"
    old = time.time() - 2 * 3600
    for path in active:
        os.utime(path, (old, old))


def test_slow_put_chunk_survives_cleanup_during_pause(client: TestClient) -> None:
    data = b"".join(bytes([i % 256]) * 1024 for i in range(8))  # 8 KiB

    sid = _create_session(client, data)
    first = data[:3072]
    rest = data[3072:]

    def cleanup_mid_receive() -> None:
        # 接收停顿：活动文件已存在且 mtime 超过一小时；此时清理绝不能删它。
        _age_active_tmp_behind_cutoff()
        removed = storage.sweep_old_tempfiles(_old_cutoff())
        assert removed == 0
        assert len(storage.active_tempfiles.snapshot()) == 1

    resp = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end={len(data)}",
        content=_slow_stream([first, rest], cleanup_mid_receive),
        headers={"Content-Length": str(len(data))},
    )
    assert resp.status_code == 200, resp.text
    # 落位成功：临时文件已转成正式分片，登记摘除。
    assert storage.part_path(sid, 0, len(data)).read_bytes() == data
    assert storage.active_tempfiles.active_count() == 0
    assert list(storage.tmp_dir().glob("*.part")) == []


def test_slow_repair_chunk_survives_cleanup_during_pause(client: TestClient) -> None:
    old_data = b"a" * 4096
    new_data = b"b" * 4096
    sid = _create_session(client, old_data)
    assert (
        client.put(
            f"/api/v1/sessions/{sid}/chunks?start=0&end=4096", content=old_data
        ).status_code
        == 200
    )

    def cleanup_mid_receive() -> None:
        _age_active_tmp_behind_cutoff()
        # 遗留文件与停顿中的接收文件同时存在：清理只回收遗留者。
        leftover = _make_orphan_old_file()
        removed = storage.sweep_old_tempfiles(_old_cutoff())
        assert removed == 1
        assert not leftover.exists()
        assert len(storage.active_tempfiles.snapshot()) == 1

    resp = client.put(
        f"/api/v1/sessions/{sid}/chunks/repair?start=0&end=4096",
        content=_slow_stream([new_data[:2048], new_data[2048:]], cleanup_mid_receive),
        headers={"Content-Length": "4096"},
    )
    assert resp.status_code == 200, resp.text
    assert storage.part_path(sid, 0, 4096).read_bytes() == new_data
    assert storage.active_tempfiles.active_count() == 0
    assert list(storage.tmp_dir().glob("*.part")) == []


@pytest.mark.parametrize(
    "url,total",
    [
        ("/chunks?start=0&end=4096", 4096),
        ("/chunks/repair?start=0&end=4096", 4096),
    ],
)
def test_failed_receive_releases_protection(
    client: TestClient, url: str, total: int
) -> None:
    # 修复路径需要先有一片已登记分片。
    if "repair" in url:
        seed = b"z" * total
        sid = _create_session(client, seed)
        assert (
            client.put(
                f"/api/v1/sessions/{sid}/chunks?start=0&end={total}", content=seed
            ).status_code
            == 200
        )
    else:
        sid = _create_session(client, b"x" * total)

    before = storage.active_tempfiles.active_count()
    # 流式接收完成后的摘要不符：临时文件被删、保护被摘除，下轮清理不会
    # 因幽灵登记永远跳过该路径。
    body = b"y" * total
    resp = client.put(
        f"/api/v1/sessions/{sid}{url}",
        content=body,
        headers={"X-Chunk-SHA256": "0" * 64},
    )
    assert resp.status_code == 422, resp.text
    assert storage.active_tempfiles.active_count() == before
    assert list(storage.tmp_dir().glob("*.part")) == []


def test_stream_abort_releases_protection(client: TestClient) -> None:
    """接收途中客户端断连：异常路径也必须摘除登记。"""
    total = 4096
    sid = _create_session(client, b"q" * total)

    def aborted_stream():
        yield b"partial"
        # 模拟客户端中断（生成器抛异常），服务端流式循环随之失败。
        raise RuntimeError("client gone")

    with pytest.raises(RuntimeError, match="client gone"):
        client.put(
            f"/api/v1/sessions/{sid}/chunks?start=0&end={total}",
            content=aborted_stream(),
            headers={"Content-Length": str(total)},
        )
    assert storage.active_tempfiles.active_count() == 0
    assert list(storage.tmp_dir().glob("*.part")) == []


def test_orphan_tempfile_from_dead_process_is_reaped_on_next_sweep(
    client: TestClient,
) -> None:
    """进程重启语义：登记表是空的，旧 mtime 遗留文件立即被回收。"""
    # 正常请求后登记表应归零（无幽灵登记）。
    assert storage.active_tempfiles.active_count() == 0
    leftover = _make_orphan_old_file()
    removed = storage.sweep_old_tempfiles(_old_cutoff())
    assert removed == 1
    assert not leftover.exists()
