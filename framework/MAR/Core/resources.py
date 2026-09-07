from __future__ import annotations

from dataclasses import asdict, dataclass
import subprocess
import threading
import time

import psutil


@dataclass(frozen=True)
class ResourceUsage:
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    peak_gpu_memory_mib: float | None
    mean_gpu_utilization_pct: float | None
    samples: int
    gpu_scope: str = "host"

    def to_dict(self) -> dict:
        return asdict(self)


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._cpu_started = 0.0
        self._rss: list[int] = []
        self._gpu_memory: list[float] = []
        self._gpu_utilization: list[float] = []

    def _processes(self) -> list[psutil.Process]:
        try:
            return [self.process, *self.process.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return [self.process]

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss = 0
            for process in self._processes():
                try:
                    rss += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self._rss.append(rss)
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                )
                if result.returncode == 0 and result.stdout.strip():
                    rows = [[float(value.strip()) for value in line.split(",")] for line in result.stdout.splitlines()]
                    self._gpu_memory.append(sum(row[0] for row in rows))
                    self._gpu_utilization.append(sum(row[1] for row in rows) / len(rows))
            except (OSError, subprocess.TimeoutExpired, ValueError):
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "ResourceMonitor":
        cpu = self.process.cpu_times()
        self._cpu_started = cpu.user + cpu.system
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(self.interval_seconds * 2, 1.0))

    def result(self) -> ResourceUsage:
        cpu = self.process.cpu_times()
        cpu_seconds = cpu.user + cpu.system - self._cpu_started
        return ResourceUsage(
            wall_seconds=max(time.monotonic() - self._started, 0.0),
            cpu_seconds=max(cpu_seconds, 0.0),
            peak_rss_bytes=max(self._rss, default=0),
            peak_gpu_memory_mib=max(self._gpu_memory) if self._gpu_memory else None,
            mean_gpu_utilization_pct=sum(self._gpu_utilization) / len(self._gpu_utilization) if self._gpu_utilization else None,
            samples=len(self._rss),
        )
