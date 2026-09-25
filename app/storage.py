"""文件系统布局、区间算法与拼装工具。"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Iterable, Iterator, Sequence

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


# ---------------------------------------------------------------------------
# 活动期临时文件登记
#
# 接收分片（PUT /chunks、PUT /chunks/repair）、拼装成品与跨会话复制都先
# 在 tmp/ 下落一个随机命名的临时文件，随后才 rename 落位。定时清理按
# mtime 删除一小时前的临时文件（回收进程退出遗留）；但请求长时间停顿
# （弱网卡顿）时，仍在使用的接收文件 mtime 也会超过一小时——若被清理
# 线程 unlink，随后的落位 rename 必然失败。
#
# 因此每个临时文件自创建起就在本进程的登记表中占一条活动记录，清理与
# 登记共用同一把锁：
#   * “创建文件并登记”与“清理扫描”互斥，磁盘上不可能存在一个清理
#     看不到保护、而业务稍后还要使用的文件——也就不会出现“刚被判为
#     闲置就被重新使用”的窗口；
#   * 活动记录只在文件落位（rename 离开 tmp/）或失败清理（unlink）
#     之后才释放，且释放与删除在同一把锁内成对完成；
#   * 登记表只在本进程内存活：进程崩溃/退出后记录随进程消失，遗留
#     文件仍由 mtime 阈值正常回收。
# ---------------------------------------------------------------------------


class _ActiveTempFiles:
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._active: set[Path] = set()

    @staticmethod
    def _key(path: Path) -> Path:
        # 清理侧来自目录枚举（绝对路径），统一绝对化以便集合比对。
        return Path(path).absolute()

    def create(self) -> Path:
        """在 tmp/ 创建一个新的临时文件，并原子地登记为活动文件。

        mkstemp 与集合登记在清理同锁内完成：清理线程拿到锁时看到的
        永远是“文件已在盘 + 已受保护”的一致状态，不会把刚创建的文件
        当成无主遗留删掉。
        """
        with self._guard:
            fd, name = tempfile.mkstemp(
                dir=tmp_dir(), prefix="recv-", suffix=".part"
            )
            os.close(fd)
            path = Path(name)
            self._active.add(self._key(path))
            return path

    def release(self, path: Path) -> None:
        """放弃对某临时文件的活动保护（幂等）。

        调用方必须已经完成文件的最终处置：成功时已 rename 离开 tmp/，
        失败时已 unlink。文件尚未处置时释放会重新暴露给定时清理。
        """
        with self._guard:
            self._active.discard(self._key(path))

    def active_count(self) -> int:
        """当前受活动期保护的临时文件数（供测试与巡检使用）。"""
        with self._guard:
            return len(self._active)

    def sweep(self, older_than: datetime) -> list[Path]:
        """删除早于给定时间且**不在活动期内**的临时文件，返回删除清单。

        判定闲置与执行删除在同一把锁内一次完成：扫描期间任何新创建/
        重新登记都要等扫描结束，因此不存在“清理判为闲置后、删除前文件
        又被业务使用”的竞态。活动记录内的文件无论 mtime 多旧都跳过。
        """
        removed: list[Path] = []
        with self._guard:
            for entry in tmp_dir().glob("*"):
                if self._key(entry) in self._active:
                    continue
                try:
                    st = entry.stat()
                except (FileNotFoundError, OSError):
                    continue
                mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                if mtime >= older_than or not entry.is_file():
                    continue
                try:
                    entry.unlink()
                except FileNotFoundError:
                    pass
                else:
                    removed.append(entry)
        return removed


_active_tempfiles = _ActiveTempFiles()


def ensure_dirs() -> None:
    for d in (files_dir(), settings.data_dir / "parts", tmp_dir()):
        d.mkdir(parents=True, exist_ok=True)


def new_temp_file() -> Path:
    """在 tmp 目录下创建一个新的临时文件（已存在、已关闭），自创建起
    即受活动期保护。

    调用方在文件最终落位（rename）或自行 unlink 后**必须**调用
    :func:`release_temp_file` 释放保护（异常路径同样要释放）；流式接收
    路径请直接使用 :func:`receive_temp_file` 上下文管理器，由它保证
    异常路径的删除与释放。
    """
    return _active_tempfiles.create()


def release_temp_file(path: str | Path) -> None:
    """释放 :func:`new_temp_file` 取得的活动期保护（幂等）。"""
    _active_tempfiles.release(Path(path))


def active_temp_file_count() -> int:
    """当前受活动期保护的临时文件数（供测试使用）。"""
    return _active_tempfiles.active_count()


@contextmanager
def receive_temp_file() -> Iterator[Path]:
    """流式接收请求体期间的临时文件上下文：创建即保护，退出即清理释放。

    正常路径：业务在 ``with`` 内把文件 rename 落位，文件已离开 tmp/，
    退出时幂等的 unlink 找不到文件，保护随之释放。失败路径（含请求
    断流、校验不过、落位/登记失败）：文件仍在 tmp/，退出时被统一
    unlink 并释放保护——任何异常都不会让保护泄漏，也不会让仍在使用
    的文件被定时清理误删。
    """
    path = new_temp_file()
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
        release_temp_file(path)


def remove_session_tree(session_id: str) -> None:
    """删除会话的分片目录与成品（若存在），用于放弃/过期清理。"""
    shutil.rmtree(parts_dir(session_id), ignore_errors=True)
    artifact = artifact_path(session_id)
    try:
        artifact.unlink()
    except FileNotFoundError:
        pass


def sweep_old_tempfiles(older_than: datetime) -> list[Path]:
    """删除早于给定时间、且不在活动期内的临时文件（回收崩溃/退出遗留），
    返回被删除的文件路径列表。

    活动期（创建→落位/失败清理）内的接收文件无视 mtime 一律保留，
    判定与删除在登记表的同一把锁内完成，因此请求长时间停顿导致 mtime
    变旧时，仍在写入的文件不会被误删；进程退出后登记表随进程消失，
    超过阈值的遗留文件仍照常回收。
    """
    return _active_tempfiles.sweep(older_than)


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
    校验通过后由调用方原子 rename 到 files 目录。返回临时文件路径，
    其活动期保护一并移交调用方：调用方在 rename 落位或失败 unlink 后
    必须调用 :func:`release_temp_file` 释放（本函数自身的失败路径已
    自行删除并释放，不会泄漏保护）。

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
    except BaseException:
        tmp_out.unlink(missing_ok=True)
        release_temp_file(tmp_out)
        raise
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
    活动期保护随返回路径移交给调用方（rename 落位或失败 unlink 后调用
    :func:`release_temp_file` 释放）；本函数自身的任何失败都清理刚创建
    的临时文件并释放保护。
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
        release_temp_file(tmp_out)
        raise
    return str(tmp_out)
