"""Hardware detection for local model setup."""

import platform
import struct
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUInfo:
    """NVIDIA GPU information."""
    name: str
    vram_total_mb: int
    vram_free_mb: int
    index: int = 0


@dataclass
class HardwareProfile:
    """Detected system hardware profile."""
    total_ram_mb: int
    available_ram_mb: int
    cpu_count: int
    cpu_arch: str          # "x86_64", "aarch64", etc.
    platform: str          # "windows", "linux", "darwin"
    gpus: List[GPUInfo] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def best_gpu(self) -> Optional[GPUInfo]:
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.vram_free_mb)

    @property
    def total_vram_mb(self) -> int:
        if not self.gpus:
            return 0
        return self.best_gpu.vram_free_mb


def detect_hardware(cpu_only: bool = False) -> HardwareProfile:
    """Detect system hardware capabilities.

    Args:
        cpu_only: If True, skip GPU detection entirely.

    Returns:
        HardwareProfile with detected specs.
    """
    ram_total, ram_available = _detect_ram()
    gpus = [] if cpu_only else _detect_gpus()

    return HardwareProfile(
        total_ram_mb=ram_total,
        available_ram_mb=ram_available,
        cpu_count=_detect_cpu_count(),
        cpu_arch=platform.machine() or "unknown",
        platform=_detect_platform(),
        gpus=gpus,
    )


def _detect_platform() -> str:
    """Detect OS platform."""
    system = platform.system().lower()
    if system == "windows" or "msys" in system or "mingw" in system:
        return "windows"
    elif system == "darwin":
        return "darwin"
    elif system == "linux":
        return "linux"
    return system


def _detect_cpu_count() -> int:
    """Detect number of CPU cores."""
    import os
    return os.cpu_count() or 1


def _detect_ram() -> tuple:
    """Detect total and available RAM in MB.

    Returns:
        (total_mb, available_mb) tuple.
    """
    # Strategy 1: psutil (cross-platform, most reliable)
    try:
        import psutil
        mem = psutil.virtual_memory()
        return (mem.total // (1024 * 1024), mem.available // (1024 * 1024))
    except ImportError:
        logger.debug("psutil not available, trying fallback RAM detection")

    # Strategy 2: Platform-specific fallbacks
    system = platform.system().lower()

    if system == "windows" or "msys" in system or "mingw" in system:
        return _detect_ram_windows()
    elif system == "linux":
        return _detect_ram_linux()
    elif system == "darwin":
        return _detect_ram_darwin()

    logger.warning("Could not detect RAM, assuming 8GB")
    return (8192, 4096)


def _detect_ram_windows() -> tuple:
    """Detect RAM on Windows using ctypes."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total = stat.ullTotalPhys // (1024 * 1024)
        available = stat.ullAvailPhys // (1024 * 1024)
        return (total, available)
    except Exception as e:
        logger.debug(f"Windows RAM detection failed: {e}")
        return (8192, 4096)


def _detect_ram_linux() -> tuple:
    """Detect RAM on Linux via /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        info = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                info[key] = int(parts[1])  # value in kB

        total = info.get("MemTotal", 8 * 1024 * 1024) // 1024
        available = info.get("MemAvailable", total // 2) // 1024
        return (total, available)
    except Exception as e:
        logger.debug(f"Linux RAM detection failed: {e}")
        return (8192, 4096)


def _detect_ram_darwin() -> tuple:
    """Detect RAM on macOS."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            total = int(result.stdout.strip()) // (1024 * 1024)
            # macOS doesn't have a simple "available" — estimate 50%
            return (total, total // 2)
    except Exception as e:
        logger.debug(f"macOS RAM detection failed: {e}")
    return (8192, 4096)


def _detect_gpus() -> List[GPUInfo]:
    """Detect NVIDIA GPUs.

    Tries pynvml first, then nvidia-smi subprocess fallback.
    Returns empty list if no NVIDIA GPU found.
    """
    # Strategy 1: pynvml
    gpus = _detect_gpus_pynvml()
    if gpus is not None:
        return gpus

    # Strategy 2: nvidia-smi
    gpus = _detect_gpus_nvidia_smi()
    if gpus is not None:
        return gpus

    logger.info("No NVIDIA GPU detected, will use CPU-only mode")
    return []


def _detect_gpus_pynvml() -> Optional[List[GPUInfo]]:
    """Detect GPUs using pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()

        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(GPUInfo(
                name=name,
                vram_total_mb=mem_info.total // (1024 * 1024),
                vram_free_mb=mem_info.free // (1024 * 1024),
                index=i,
            ))

        pynvml.nvmlShutdown()
        logger.info(f"Detected {len(gpus)} GPU(s) via pynvml")
        return gpus
    except ImportError:
        logger.debug("pynvml not available")
        return None
    except Exception as e:
        logger.debug(f"pynvml detection failed: {e}")
        return None


def _detect_gpus_nvidia_smi() -> Optional[List[GPUInfo]]:
    """Detect GPUs using nvidia-smi subprocess."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        logger.debug("nvidia-smi not found in PATH")
        return None

    try:
        result = subprocess.run(
            [nvidia_smi,
             "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            logger.debug(f"nvidia-smi failed: {result.stderr}")
            return None

        gpus = []
        for i, line in enumerate(result.stdout.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append(GPUInfo(
                    name=parts[0],
                    vram_total_mb=int(float(parts[1])),
                    vram_free_mb=int(float(parts[2])),
                    index=i,
                ))

        logger.info(f"Detected {len(gpus)} GPU(s) via nvidia-smi")
        return gpus
    except Exception as e:
        logger.debug(f"nvidia-smi detection failed: {e}")
        return None
