"""Regression tests for repeated OVF exports."""

import json
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="RSS is read from /proc")
def test_repeated_ovf_exports_have_bounded_memory(tmp_path):
    # Isolate CUDA initialization and allocator state from the other tests.
    code = textwrap.dedent(
        """
        import gc
        import json
        from pathlib import Path
        from mumaxplus import Ferromagnet, Grid, World

        def rss_bytes():
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
            raise RuntimeError('VmRSS is unavailable')

        world = World(cellsize=(1e-9, 1e-9, 1e-9))
        magnet = Ferromagnet(world, Grid((128, 128, 2)))
        magnet.enable_demag = False
        magnet.magnetization = (1, 2, 3)
        quantity = magnet.magnetization
        for _ in range(10):
            quantity.save_ovf('repeated.ovf')
        gc.collect()
        before = rss_bytes()
        for _ in range(512):
            quantity.save_ovf('repeated.ovf')
        gc.collect()
        print(json.dumps({'growth_bytes': rss_bytes() - before}))
        """
    )
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        check=True, capture_output=True, text=True, timeout=120,
    )
    growth = json.loads(result.stdout.strip().splitlines()[-1])["growth_bytes"]
    # The leaking backend retains 192 MiB here; allow 32 MiB for allocator noise.
    assert growth < 32 * 1024**2, f"Repeated OVF exports retained {growth} bytes"
