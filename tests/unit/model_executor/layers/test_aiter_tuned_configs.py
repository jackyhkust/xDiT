import os
import sys
import tempfile
from types import SimpleNamespace

import pytest

from xfuser.model_executor.layers import aiter_tuned_configs as configs

pd = pytest.importorskip("pandas")

COLUMNS = ["gfx", "cu_num", "M", "N", "K", "kernelId", "splitK", "us", "kernelName", "tflops", "bw", "errRatio"]


def _row(m, kernel="_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"):
    return ["gfx950", 256, m, 3072, 3072, 14, 0, 8.0, kernel, 1.0, 1.0, 0.0]


def _write(path, ms):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_row(m) for m in ms], columns=COLUMNS).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def tables(tmp_path, monkeypatch):
    aiter_dir = tmp_path / "aiter" / "configs"
    default = _write(aiter_dir / "a4w4_blockscale_tuned_gemm.csv", [1, 80])
    model = _write(aiter_dir / "model_configs" / "dsv3_a4w4_blockscale_tuned_gemm.csv", [160])
    _write(aiter_dir / "model_configs" / "dsv3_a4w4_blockscale_untuned_gemm.csv", [240])
    _write(tmp_path / "xdit" / "shipped.csv", [80, 160, 240, 320])
    monkeypatch.setattr(configs, "_aiter_configs_dir", lambda: aiter_dir)
    monkeypatch.setattr(configs, "_CONFIG_DIR", tmp_path / "xdit")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    monkeypatch.delenv(configs.A4W4_ENV, raising=False)
    return SimpleNamespace(default=default, model=model, tmp=tmp_path)


def test_registration_lists_aiter_tables_then_rows_they_lack(tables):
    assert configs.register_aiter_a4w4_tuned_gemms("shipped.csv") == 2

    *base, added = os.environ[configs.A4W4_ENV].split(os.pathsep)
    assert base == [tables.default, tables.model]
    assert pd.read_csv(added).M.tolist() == [240, 320]


def test_registration_keeps_user_tables(tables, monkeypatch):
    user = _write(tables.tmp / "user.csv", [320])
    monkeypatch.setenv(configs.A4W4_ENV, user)

    assert configs.register_aiter_a4w4_tuned_gemms("shipped.csv") == 3

    *base, added = os.environ[configs.A4W4_ENV].split(os.pathsep)
    assert base == [user]
    assert pd.read_csv(added).M.tolist() == [80, 160, 240]


def test_second_registration_adds_nothing(tables):
    configs.register_aiter_a4w4_tuned_gemms("shipped.csv")
    first = os.environ[configs.A4W4_ENV]

    assert configs.register_aiter_a4w4_tuned_gemms("shipped.csv") == 0
    assert os.environ[configs.A4W4_ENV] == first


def test_registration_after_aiter_loaded_its_table_is_skipped(tables, monkeypatch):
    lookup = SimpleNamespace(gemm_dict={})
    monkeypatch.setitem(sys.modules, "aiter.ops.gemm_op_a4w4", SimpleNamespace(get_GEMM_config=lookup))

    assert configs.register_aiter_a4w4_tuned_gemms("shipped.csv") == 0
    assert configs.A4W4_ENV not in os.environ


def test_shipped_flux3_table_is_in_aiter_format():
    table = pd.read_csv(configs._CONFIG_DIR / "flux3_klein_a4w4_tuned_gemm.csv")

    assert table.columns.tolist() == COLUMNS
    assert not table.duplicated(["gfx", "cu_num", "M", "N", "K"]).any()
    # asm kernels without split-K: the same output bits as AITER's default kernel choice
    assert table.kernelName.str.startswith("_ZN").all()
    assert (table.splitK == 0).all()
    assert set(zip(table.N, table.K)) == {(27648, 3072), (3072, 3072), (3072, 9216), (3072, 12288)}
