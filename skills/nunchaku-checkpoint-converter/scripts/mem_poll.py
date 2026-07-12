"""nvidia-smi-based peak GPU memory poller.

``torch.cuda.max_memory_allocated()`` only tracks memory that went through
torch's own caching allocator. The native Nunchaku engine (the ``nunchaku``
PyPI package) allocates its quantized transformer weights via raw CUDA calls
that bypass that allocator entirely, so benchmarking it with the torch metric
silently undercounts its real VRAM footprint -- sometimes by several GiB.
``GpuMemPoller`` samples the device's actual memory usage via ``nvidia-smi``
on a background thread instead, which sees every allocation regardless of
which library made it.

Usage:

    from mem_poll import GpuMemPoller

    with GpuMemPoller(interval_s=0.05) as poller:
        run_inference()
    print(poller.peak_gib)

Use this for any benchmark row that exercises the native Nunchaku engine.
``torch.cuda.max_memory_allocated()`` remains fine for benchmarking the
Diffusers Nunchaku Lite converted pipeline itself, since that path allocates
entirely through torch's allocator.
"""

from __future__ import annotations

import subprocess
import threading
import time


class GpuMemPoller:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                ).stdout.strip()
                val = int(out.splitlines()[0])
                self.peak_mib = max(self.peak_mib, val)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self) -> "GpuMemPoller":
        self.peak_mib = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def peak_gib(self) -> float:
        return self.peak_mib / 1024
