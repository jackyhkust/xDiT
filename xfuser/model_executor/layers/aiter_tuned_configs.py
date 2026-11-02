"""Tuned AITER GEMM configs shipped with xDiT, added to AITER's own tables.

AITER reads ``AITER_CONFIG_GEMM_A4W4`` once, at the first MXFP4 GEMM of the process. With the
variable unset it merges its default table with ``configs/model_configs/*a4w4_blockscale_tuned_gemm*.csv``;
set, it merges the ``os.pathsep``-separated files listed. Registering a table here sets the variable to
those same tables (or the user's own value) followed by the xDiT table.

A shape that AITER's tables already cover keeps AITER's entry. AITER's merge raises on a duplicate
shape and rewrites the source files while resolving it, so such rows are dropped from a filtered copy
of the xDiT table before it is listed.
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

from xfuser.logger import init_logger

logger = init_logger(__name__)

A4W4_ENV = "AITER_CONFIG_GEMM_A4W4"
_A4W4_TABLE = "a4w4_blockscale_tuned_gemm"
_CONFIG_DIR = Path(__file__).parent / "aiter_configs"


def _aiter_configs_dir() -> Path | None:
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(spec.submodule_search_locations[0]) / "configs"


def _aiter_a4w4_tables() -> list[str]:
    """The tables AITER would merge for MXFP4 GEMMs if nothing were registered."""
    env = (os.environ.get(A4W4_ENV) or "").strip()
    if env:
        return [path for path in env.split(os.pathsep) if path]
    configs = _aiter_configs_dir()
    if configs is None:
        return []
    model_tables = sorted(
        str(path)
        for path in (configs / "model_configs").glob(f"*{_A4W4_TABLE}*.csv")
        if path.is_file() and "untuned" not in path.name
    )
    return [str(configs / f"{_A4W4_TABLE}.csv"), *model_tables]


def _shape_keys(frame) -> list[str]:
    return [column for column in ("gfx", "cu_num", "M", "N", "K") if column in frame.columns]


def _aiter_already_loaded() -> bool:
    module = sys.modules.get("aiter.ops.gemm_op_a4w4")
    return module is not None and hasattr(getattr(module, "get_GEMM_config", None), "gemm_dict")


def register_aiter_a4w4_tuned_gemms(table: str) -> int:
    """Add ``aiter_configs/<table>`` to AITER's MXFP4 GEMM tables; returns the number of rows added.

    Takes effect only before the process's first MXFP4 GEMM.
    """
    import pandas as pd

    if _aiter_already_loaded():
        logger.warning(f"AITER loaded its MXFP4 GEMM table before {table} was registered; not using it")
        return 0
    base = _aiter_a4w4_tables()
    if not base:
        return 0
    rows = pd.read_csv(_CONFIG_DIR / table)
    for path in base:
        if not Path(path).is_file():
            continue
        existing = pd.read_csv(path)
        keys = [key for key in _shape_keys(existing) if key in rows.columns]
        covered = rows[keys].merge(existing[keys].drop_duplicates(), how="left", indicator=True)["_merge"] == "both"
        rows = rows[~covered.to_numpy()]
    if rows.empty:
        return 0
    out_dir = Path(tempfile.gettempdir()) / "xdit_aiter_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered = out_dir / f"{Path(table).stem}.{os.getpid()}.csv"
    rows.to_csv(filtered, index=False)
    os.environ[A4W4_ENV] = os.pathsep.join([*base, str(filtered)])
    logger.info(f"Registered {len(rows)} tuned AITER MXFP4 GEMM shapes from {table}")
    return len(rows)
