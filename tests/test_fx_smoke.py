"""One Dukascopy hour-range for one pair and one day. Off unless PIPELINE_FX_NETWORK=1."""

from __future__ import annotations

import os
from datetime import date

import pytest

from pipeline.orchestrator import RunConfig, run

pytestmark = pytest.mark.network


@pytest.mark.skipif(os.environ.get("PIPELINE_FX_NETWORK") != "1", reason="set PIPELINE_FX_NETWORK=1")
def test_fx_one_pair_one_day(tmp_path):
    code = run(RunConfig(
        start=date(2024, 1, 16),
        end=date(2024, 1, 16),
        assets=["EURUSD"],
        market="fx",
        stages=["fetch", "bars", "features", "dataset", "validate"],
        bar_types=["time_1m", "time_1h"],
        data_root=tmp_path,
        force=False,
    ))
    assert code < 2
