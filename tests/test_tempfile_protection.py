"""临时接收文件的活动期保护。

慢速分片上传（PUT /chunks）与原位修复（PUT /chunks/repair）都先在
``tmp/`` 下落一个接收文件，成功后 rename 落位、失败则 unlink。定时清理
按 mtime 删除一小时前的临时文件以回收进程退出遗留；但请求长时间停顿
（弱网卡顿）时，仍在接收的文件 mtime 也会超过阈值。本模块覆盖：

  * 接收进行中（即使把 mtime 调到很久之前）清理绝不删除活动文件，
    请求随后仍能正常落位完成——直接复现原“误删导致落位失败”的缺陷；
  * 清理判定与创建/释放共用同一把锁，不存在“刚判为闲置就被重新使用”
    的窗口：并发创建与清理交错时活动文件始终受保护；
  * 异常路径（断流、长度不符、摘要不符、修复失败）释放保护，
    登记表不泄漏；
  * 进程退出遗留（无活动登记）且超过阈值的文件仍被回收，
    新于阈值的文件保留；
  * 拼装成品（complete）与服务端复制（copy-chunk）的临时文件同样
    全程受保护，结束后保护释放。
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from app import storage
from app.cleanup import run_cleanup
from app.main import app


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _create(client: TestClient, data: bytes, name: str = "data.bin") -> str:
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


def _recv_files() -> list:
    return list(storage.tmp_dir().glob("recv-*.part"))


def _age_old(path) -> None:
    old = time.time() - 24 * 3600
    os.utime(path, (old, old))


@pytest.fixture(autouse=True)
def _isolated_tmp():
    """数据目录在全部测试间共享；每个用例开始前清空 tmp 并复核无活动登记。"""
    storage.ensure_dirs()
    for p in _recv_files():
        p.unlink(missing_ok=True)
    assert storage.active_temp_file_count() == 0
    yield
    for p in _recv_files():
        p.unlink(missing_ok=True)


@pytest.fixture()
def loop():
    """专用事件循环跑在独立线程上：主测试线程可在接收中途插入清理。"""
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)
    loop.close()


async def _put_with_stall(path: str, payload: bytes, gate: threading.Event) -> httpx.Response:
    """经 ASGI 传输增量发送请求体：服务端确实在“收到 1 字节后停顿”。

    显式给 content-length：否则 httpx 对未知长度的异步生成体会先缓冲
    整段请求体（服务端也会因缺头返回 411），无法复现接收中途停顿。
    """

    async def body():
        yield payload[:1]
        await asyncio.to_thread(gate.wait, 10)
        yield payload[1:]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        req = httpx.Request(
            "PUT",
            f"http://test{path}",
            content=body(),
            headers={"content-length": str(len(payload))},
        )
        return await client.send(req)


def _wait_for_recv(timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        recv = _recv_files()
        if recv:
            return recv
        time.sleep(0.01)
    return []


# ---------------------------------------------------------------------------
# 慢速 PUT /chunks：接收停顿期间清理不得删除活动接收文件
# ---------------------------------------------------------------------------


def test_slow_chunk_upload_survives_cleanup_during_stall(
    client: TestClient, loop
) -> None:
    data = bytes((i * 7 + 3) % 256 for i in range(5000))
    sid = _create(client, data)

    gate = threading.Event()
    task = asyncio.run_coroutine_threadsafe(
        _put_with_stall(f"/api/v1/sessions/{sid}/chunks?start=0&end=5000", data, gate),
        loop,
    )
    try:
        recv = _wait_for_recv()
        assert len(recv) == 1
        # 接收文件此刻正被请求持有：登记表里有它。
        assert storage.active_temp_file_count() >= 1

        # 请求长时间停顿：mtime 远超 1 小时清理阈值，多轮清理也不得删除。
        _age_old(recv[0])
        run_cleanup(storage.utcnow() - timedelta(hours=1))
        run_cleanup(storage.utcnow() - timedelta(hours=1))
        assert recv[0].exists()
        assert storage.active_temp_file_count() >= 1
    finally:
        gate.set()
        resp = task.result(timeout=10)

    assert resp.status_code == 200, resp.text
    # 落位完成：登记表已释放。
    assert _recv_files() == []

    r = client.post(f"/api/v1/sessions/{sid}/complete")
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# 慢速 PUT /chunks/repair：同样受保护
# ---------------------------------------------------------------------------


def test_slow_chunk_repair_survives_cleanup_during_stall(
    client: TestClient, loop
) -> None:
    old = b"old" * 1000
    sid = _create(client, old)
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=3000", content=old
    )
    assert r.status_code == 200, r.text

    new = b"new" * 1000
    gate = threading.Event()
    task = asyncio.run_coroutine_threadsafe(
        _put_with_stall(
            f"/api/v1/sessions/{sid}/chunks/repair?start=0&end=3000", new, gate
        ),
        loop,
    )
    try:
        recv = _wait_for_recv()
        assert len(recv) == 1
        _age_old(recv[0])
        run_cleanup(storage.utcnow() - timedelta(hours=1))
        assert recv[0].exists()
    finally:
        gate.set()
        resp = task.result(timeout=10)

    assert resp.status_code == 200, resp.text
    assert storage.part_path(sid, 0, 3000).read_bytes() == new
    assert _recv_files() == []


# ---------------------------------------------------------------------------
# 创建/清理同步：判定闲置与删除在同一把锁内，新创建的文件始终受保护
# ---------------------------------------------------------------------------


def test_sweep_never_touches_active_files_even_when_ancient() -> None:
    storage.ensure_dirs()
    before = storage.active_temp_file_count()
    active = storage.new_temp_file()
    try:
        assert storage.active_temp_file_count() == before + 1
        _age_old(active)
        storage.sweep_old_tempfiles(storage.utcnow() - timedelta(hours=1))
        assert active.exists()
    finally:
        active.unlink(missing_ok=True)
        storage.release_temp_file(active)
    assert storage.active_temp_file_count() == before


def test_create_and_sweep_race_keeps_every_active_file() -> None:
    storage.ensure_dirs()
    created: list = []
    stop = threading.Event()

    def _creator() -> None:
        for _ in range(50):
            p = storage.new_temp_file()
            created.append(p)
            time.sleep(0)

    sweeper = threading.Thread(
        target=lambda: [
            storage.sweep_old_tempfiles(storage.utcnow() - timedelta(hours=1))
            for _ in range(200)
            if not stop.wait(0)
        ]
    )
    creator = threading.Thread(target=_creator)
    sweeper.start()
    creator.start()
    creator.join(timeout=10)
    stop.set()
    sweeper.join(timeout=10)

    # 所有已创建并仍登记的文件，在并发清理中一个都不能丢。
    assert len(created) == 50
    for p in created:
        assert p.exists()
        p.unlink(missing_ok=True)
        storage.release_temp_file(p)
    assert storage.active_temp_file_count() == 0


# ---------------------------------------------------------------------------
# 进程退出遗留：无活动登记且超阈值才回收
# ---------------------------------------------------------------------------


def test_sweep_reclaims_unregistered_leftover_but_keeps_recent() -> None:
    storage.ensure_dirs()
    stale = storage.new_temp_file()
    recent = storage.new_temp_file()
    try:
        # 模拟进程退出：登记表随进程消失（直接释放保护，文件仍在盘上）。
        storage.release_temp_file(stale)
        storage.release_temp_file(recent)
        _age_old(stale)  # recent 保持当前 mtime

        removed = storage.sweep_old_tempfiles(
            storage.utcnow() - timedelta(hours=1)
        )
        assert not stale.exists()
        assert recent.exists()
        assert stale in removed
        recent.unlink(missing_ok=True)
    finally:
        storage.release_temp_file(stale)
        storage.release_temp_file(recent)


# ---------------------------------------------------------------------------
# 异常路径释放保护（登记表不泄漏）
# ---------------------------------------------------------------------------


def test_digest_mismatch_releases_protection(client: TestClient) -> None:
    data = b"x" * 500
    sid = _create(client, data)
    before = storage.active_temp_file_count()
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=500",
        content=data,
        headers={"X-Chunk-SHA256": "a" * 64},
    )
    assert r.status_code == 422
    assert storage.active_temp_file_count() == before
    assert _recv_files() == []


def test_wrong_length_releases_protection(client: TestClient) -> None:
    data = b"y" * 500
    sid = _create(client, data)
    before = storage.active_temp_file_count()
    # 声明 500 字节，实发 499。
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=500",
        content=data[:499],
    )
    assert r.status_code == 400
    assert storage.active_temp_file_count() == before
    assert _recv_files() == []


async def _put_broken_stream(path: str) -> None:
    """发送 7 字节后中途断开，触发服务端流式接收路径的异常清理。"""

    async def body():
        yield b"partial"
        raise httpx.ReadError("client gone")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        try:
            req = httpx.Request(
                "PUT",
                f"http://test{path}",
                content=body(),
                headers={"content-length": "500"},
            )
            await client.send(req)
        except (httpx.ReadError, httpx.RequestError):
            pass


def test_broken_stream_releases_protection(client: TestClient, loop) -> None:
    data = b"z" * 500
    sid = _create(client, data)

    future = asyncio.run_coroutine_threadsafe(
        _put_broken_stream(f"/api/v1/sessions/{sid}/chunks?start=0&end=500"),
        loop,
    )
    future.result(timeout=10)

    # 接收中途失败：临时文件已随上下文删除，保护已释放。
    deadline = time.time() + 5
    while storage.active_temp_file_count() and time.time() < deadline:
        time.sleep(0.01)
    assert storage.active_temp_file_count() == 0
    assert _recv_files() == []


def test_repair_digest_mismatch_releases_protection(client: TestClient) -> None:
    data = b"r" * 400
    sid = _create(client, data)
    assert client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=400", content=data
    ).status_code == 200
    before = storage.active_temp_file_count()
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks/repair?start=0&end=400",
        content=b"s" * 400,
        headers={"X-Chunk-SHA256": "a" * 64},
    )
    assert r.status_code == 422
    assert storage.active_temp_file_count() == before
    assert _recv_files() == []
    # 旧分片原样保留。
    assert storage.part_path(sid, 0, 400).read_bytes() == data


# ---------------------------------------------------------------------------
# complete / copy-chunk：临时文件全程受保护，结束后登记表释放
# ---------------------------------------------------------------------------


def test_complete_releases_tempfile_protection(client: TestClient) -> None:
    data = b"c" * 1500
    sid = _create(client, data)
    half = 750
    assert client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end={half}",
        content=data[:half],
    ).status_code == 200
    assert client.put(
        f"/api/v1/sessions/{sid}/chunks?start={half}&end=1500",
        content=data[half:],
    ).status_code == 200
    assert storage.active_temp_file_count() == 0
    assert client.post(f"/api/v1/sessions/{sid}/complete").status_code == 200
    assert storage.active_temp_file_count() == 0
    assert _recv_files() == []


def test_copy_chunk_releases_tempfile_protection(client: TestClient) -> None:
    src_data = b"s" * 800
    src = _create(client, src_data)
    assert client.put(
        f"/api/v1/sessions/{src}/chunks?start=0&end=800", content=src_data
    ).status_code == 200
    assert client.post(f"/api/v1/sessions/{src}/complete").status_code == 200

    tgt_data = b"t" * 800
    tgt = _create(client, tgt_data)
    r = client.post(
        f"/api/v1/sessions/{tgt}/copy-chunk",
        json={
            "source_session_id": src,
            "source_start": 0,
            "source_end": 800,
            "target_start": 0,
        },
    )
    assert r.status_code == 200, r.text
    assert storage.active_temp_file_count() == 0
    assert _recv_files() == []
