"""Optional Vision smoke. Skipped unless PIPELINE_V2_NETWORK=1."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from pipeline.orchestrator import RunConfig, run

pytestmark = pytest.mark.skipif(
    os.environ.get("PIPELINE_V2_NETWORK") != "1",
    reason="set PIPELINE_V2_NETWORK=1 to fetch a few days of BTC from Vision",
)


def test_btc_vision_smoke(tmp_path: Path):
    # October 2026 is a closed month, so the monthly funding zip exists.
    # Three days is enough to build 1m -> 1h -> 1d and dollar bars.
    cfg = RunConfig(
        start=date(2026, 10, 1),
        end=date(2026, 10, 3),
        assets=["BTCUSDT"],
        stages=["fetch", "bars", "features", "dataset", "validate"],
        bar_types=["time_1m", "time_1h", "time_1d", "dollar"],
        data_root=tmp_path,
        force=True,
        fetch_timeout=180,
    )
    code = run(cfg)
    assert code == 0
    assert list((tmp_path / "gold").rglob("*.parquet"))
