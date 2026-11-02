"""xDiT's tuned MXFP4 GEMM table must reach AITER's lookup next to AITER's own tables.

AITER reads its table once per process, so each check runs in a fresh interpreter.
"""

import json
import subprocess
import sys

import pytest

pytest.importorskip("aiter")

_CHECK = """
import hashlib, json, os, sys
from xfuser.model_executor.layers.aiter_tuned_configs import A4W4_ENV, _aiter_a4w4_tables, register_aiter_a4w4_tuned_gemms
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
import pandas as pd

digest = lambda paths: {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths}
aiter_tables = _aiter_a4w4_tables()
before = digest(aiter_tables)
added = register_aiter_a4w4_tuned_gemms("flux3_klein_a4w4_tuned_gemm.csv")
from aiter.ops.gemm_op_a4w4 import get_GEMM_config
flux3 = get_GEMM_config(522, 27648, 3072)
own = pd.read_csv(aiter_tables[0])
own = own[(own.gfx == get_gfx_runtime()) & (own.cu_num == get_cu_num())].iloc[0]
own_config = get_GEMM_config(int(own.M), int(own.N), int(own.K))
print(json.dumps(dict(
    arch=[get_gfx_runtime(), get_cu_num()], added=added,
    flux3=flux3 and flux3["kernelName"], own=own_config and own_config["kernelName"], own_expected=own.kernelName,
    aiter_tables_unchanged=digest(aiter_tables) == before,
)))
"""


def test_flux3_table_resolves_beside_aiter_tables():
    result = subprocess.run([sys.executable, "-c", _CHECK], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout.strip().splitlines()[-1])
    if report["arch"] != ["gfx950", 256]:
        pytest.skip(f"the FLUX.3 table is tuned for gfx950 with 256 CUs, not {report['arch']}")

    assert report["added"] > 0
    assert report["flux3"] == "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256E"
    assert report["own"] == report["own_expected"]
    assert report["aiter_tables_unchanged"]
