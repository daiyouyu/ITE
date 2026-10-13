"""读取可用内存，为断联实验限制并发训练进程。"""

from __future__ import annotations

import os
from pathlib import Path


GIB = 1024 ** 1
RESERVED_MEMORY_BYTES = 1 * GIB
MEMORY_PER_TRAINING_JOB_BYTES = 2 * GIB


def available_memory_bytes() -> int | None:
    """返回系统及 cgroup 限制下的可用内存；无法读取时返回 None。

    Linux 容器的宿主机空闲内存可能高于容器限额，因此取两者较小值。
    """
    candidates: list[int] = []
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            candidates.append(int(status.available_physical))
    else:
        try:
            candidates.append(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, AttributeError):
            pass
        for limit_path, usage_path in (
            (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
            (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
             Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
        ):
            try:
                limit_text = limit_path.read_text(encoding="ascii").strip()
                if limit_text == "max":
                    continue
                remaining = int(limit_text) - int(usage_path.read_text(encoding="ascii"))
                if remaining >= 0:
                    candidates.append(remaining)
            except (OSError, ValueError):
                continue
    return min(candidates) if candidates else None


def safe_training_jobs(requested: int) -> tuple[int, int | None]:
    """按可用内存收紧训练并发数，始终预留系统内存。

    Args:
        requested: 用户允许的最大重训练进程数。

    Returns:
        实际建议并发数及检测到的可用内存字节数。
    """
    if requested < 1:
        raise ValueError("训练进程数必须大于或等于 1")
    available = available_memory_bytes()
    if available is None:
        return 1, None
    minimum = RESERVED_MEMORY_BYTES + 2 * GIB
    if available < minimum:
        raise MemoryError(
            f"断联实验启动前可用内存仅 {available / GIB:.1f} GiB；"
            f"至少需要 {minimum / GIB:.1f} GiB。"
        )
    affordable = max(1, (available - RESERVED_MEMORY_BYTES) // MEMORY_PER_TRAINING_JOB_BYTES)
    return min(requested, affordable), available
