"""Exercise the real startup import order in a fresh interpreter."""
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("override, expected", [(None, 1), ("2", 2)])
def test_mcp_startup_limits_openblas_and_preserves_explicit_override(override, expected):
    env = os.environ.copy()
    env.pop("OPENBLAS_NUM_THREADS", None)
    if override is not None:
        env["OPENBLAS_NUM_THREADS"] = override
    code = """
import acadmem.mcp_server
import json
from threadpoolctl import threadpool_info
print(json.dumps([p['num_threads'] for p in threadpool_info() if p['internal_api'] == 'openblas']))
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=60, check=True)
    pools = json.loads(result.stdout)
    assert pools, "The test must observe the NumPy/OpenBLAS pool loaded by the server"
    assert all(count == expected for count in pools)
