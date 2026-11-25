"""``mlsys certify`` on small synthetic roots. Each proof is shown to catch a planted defect,
and each refusal to leave the stored files untouched."""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

import pipeline.chunked as chunked
import pipeline.features as features
from pipeline.certify import certificate_path, certify, code_identity
from pipeline.cli import build_parser, main
from pipeline.fetch_external import _write, external_dir
from pipeline.io import atomic_write_parquet, file_sha256
from pipeline.orchestrator import RunConfig, run
from pipeline.paths import fx_hour_path, gold_month_path
from fixtures import small_windows, write_bronze_day
from test_resume import _day

MAY = [date(2024, 5, 1) + timedelta(days=i) for i in range(3)]
PAIRS = {"EURUSD": (1.11, 0.00005), "USDJPY": (108.5, 0.005), "GBPUSD": (1.31, 0.00007)}


def _crypto_root(root, days=MAY, assets=("BTCUSDT", "ETHUSDT"), bar_types=("time_1h",), constant_dollar=True):
    for i, asset in enumerate(assets):
        for k, d in enumerate(days):
            write_bronze_day(root, asset, d, _day(10 * i + k + 1, 100.0 * (i + 1) + k, d))
    cfg = RunConfig(
        start=days[0], end=days[-1], assets=list(assets), stages=["bars", "features", "dataset"],
        bar_types=list(bar_types), data_root=root, windows=small_windows(),
        threshold_overrides={a: 5000.0 for a in assets} if constant_dollar else None,
    )
    assert run(cfg) == 0
    return cfg


def _fx_root(root):
    rng = np.random.default_rng(0)
    d0 = datetime(2020, 1, 7, tzinfo=timezone.utc)  # Tuesday to Thursday: no weekend close
    for pair, (p0, half) in PAIRS.items():
        px = p0
        for hour in range(72):
            h0 = d0 + timedelta(hours=hour)
            offs = np.sort(rng.integers(0, 3_600_000, 120))
            mids = px * np.exp(np.cumsum(rng.normal(0, 2e-5, 120)))
            px = float(mids[-1])
            ticks = pl.DataFrame({"ts_ms": int(h0.timestamp() * 1000) + offs, "bid": mids - half, "ask": mids + half})
            ticks = ticks.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
                                       (pl.col("ask") - pl.col("bid")).alias("spread"))
            atomic_write_parquet(ticks, fx_hour_path(root, pair, h0.date(), h0.hour), schema_name="fx_ticks_v1",
                                 params={"synthetic": True})
    rows = [{"area": area, "event_time_ms": int((d0 + timedelta(days=i)).timestamp() * 1000), "rate_pct": rate}
            for i in range(-10, 4) for area, rate in (("US", 1.625), ("XM", 0.0), ("JP", -0.1), ("GB", 0.75))]
    _write(pl.DataFrame(rows), external_dir(root, "bis") / "policy_rates.parquet", "external_bis_policy_v1",
           {"synthetic": True}, key=["area", "event_time_ms"])
    cfg = RunConfig(start=d0.date(), end=d0.date() + timedelta(days=2), assets=list(PAIRS), market="fx",
                    stages=["bars", "features", "dataset"], bar_types=["time_1h"], data_root=root)
    assert run(cfg) == 0
    return cfg


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


def _shas(root, sub):
    return {str(p.relative_to(root)): file_sha256(p) for p in (root / sub).rglob("*.parquet")}


def test_certify_reproduces_and_proves_a_fresh_crypto_build(tmp_path):
    cfg = _crypto_root(tmp_path)
    report = certify(cfg, rebuild=True, network=False)
    assert _check(report, "status").severity == "PASS"
    reproduce = _check(report, "reproduce")
    assert reproduce.severity == "PASS", reproduce.detail
    assert "byte-identical" in reproduce.detail
    proof = _check(report, "causality:gold_xd:time_1h")
    assert proof.severity == "PASS", proof.detail
    assert not proof.detail.startswith("no cross-sectional")  # the proof compared real xd values
    assert _check(report, "crosscheck").severity == "WARN"  # --skip-network is a warning, not a pass
    cert = json.loads(certificate_path(cfg).read_text())
    assert cert["exit"] == report.code
    assert cert["range"] == ["2024-05-01", "2024-05-03"]
    assert cert["data"]["partitions"] > 0 and len(cert["data"]["sha256_digest"]) == 64
    assert {c["name"] for c in cert["checks"]} >= {"code", "status", "reproduce", "validate", "crosscheck"}
    assert cert["details"]["validate"], "the validate stage's own checks are kept in the certificate"


def test_reproduce_fails_when_a_stored_value_was_edited(tmp_path):
    cfg = _crypto_root(tmp_path)
    path = gold_month_path(tmp_path, "time_1h", "BTCUSDT", 2024, 5)
    edited = pl.read_parquet(path).with_columns(
        pl.when(pl.int_range(pl.len()) == 10).then(pl.col("close") * 1.01).otherwise(pl.col("close")).alias("close"))
    atomic_write_parquet(edited, path, schema_name="dataset_v1", params={"edited": True})
    check = _check(certify(cfg, rebuild=True, network=False), "reproduce")
    assert check.severity == "FAIL"
    assert "changed content" in check.detail and "gold/time_1h/BTCUSDT_2024-05.parquet" in check.detail


def test_gold_proof_catches_a_cross_asset_value_from_the_next_bar(tmp_path, monkeypatch):
    real = chunked.add_cross_asset

    def leaky(frames, **kw):
        return {a: df.with_columns(pl.col("xd_cross_return_mean").shift(-1)) for a, df in real(frames, **kw).items()}

    monkeypatch.setattr(chunked, "add_cross_asset", leaky)
    cfg = _crypto_root(tmp_path)
    check = _check(certify(cfg, rebuild=False, network=False), "causality:gold_xd:time_1h")
    assert check.severity == "FAIL"
    assert "xd_cross_return_mean" in check.detail


def test_rebuild_is_refused_when_the_dollar_chain_started_earlier(tmp_path):
    days = [date(2024, 4, 29) + timedelta(days=i) for i in range(5)]  # Apr 29 .. May 3
    cfg = _crypto_root(tmp_path, days=days, bar_types=("time_1h", "dollar"), constant_dollar=False)
    before = _shas(tmp_path, "silver")
    # May 1 is a month boundary, so no stored May bar is outside the range; only the chain moves.
    check = _check(certify(dataclasses.replace(cfg, start=date(2024, 5, 1)), rebuild=True, network=False), "reproduce")
    assert check.severity == "FAIL"
    assert "calibrated on 2024-04-29" in check.detail and "calibrates on 2024-05-01" in check.detail
    assert _shas(tmp_path, "silver") == before


def test_rebuild_is_refused_for_a_start_inside_a_stored_month(tmp_path):
    cfg = _crypto_root(tmp_path)
    before = _shas(tmp_path, "silver") | _shas(tmp_path, "gold")
    check = _check(certify(dataclasses.replace(cfg, start=MAY[1]), rebuild=True, network=False), "reproduce")
    assert check.severity == "FAIL"
    assert "bars before 2024-05-02" in check.detail
    assert _shas(tmp_path, "silver") | _shas(tmp_path, "gold") == before


def test_rebuild_is_refused_for_an_asset_list_that_changes_the_gold_cohort(tmp_path):
    cfg = _crypto_root(tmp_path)
    before = _shas(tmp_path, "gold")
    report = certify(dataclasses.replace(cfg, assets=["BTCUSDT"]), rebuild=True, network=False)
    check = _check(report, "reproduce")
    assert check.severity == "FAIL" and "cohort ['BTCUSDT', 'ETHUSDT']" in check.detail
    assert _shas(tmp_path, "gold") == before
    # The proof recomputes the cohort the month was built with, so it still holds.
    assert _check(report, "causality:gold_xd:time_1h").severity == "PASS"


def test_fx_feature_proof_passes_and_catches_a_strength_value_from_the_next_bar(tmp_path, monkeypatch):
    clean = _fx_root(tmp_path / "clean")
    report = certify(clean, rebuild=True, network=False)
    assert _check(report, "reproduce").severity == "PASS", _check(report, "reproduce").detail
    proof = _check(report, "causality:fx_features:time_1h")
    assert proof.severity == "PASS", proof.detail

    real = features.add_currency_strength

    def leaky(frames):
        return {p: df.with_columns(pl.col("base_strength_ret").shift(-1)) for p, df in real(frames).items()}

    monkeypatch.setattr(features, "add_currency_strength", leaky)
    leaked = _fx_root(tmp_path / "leaked")
    check = _check(certify(leaked, rebuild=False, network=False), "causality:fx_features:time_1h")
    assert check.severity == "FAIL" and "base_strength_ret" in check.detail


def test_a_cut_frame_is_never_written(tmp_path):
    cfg = RunConfig(start=MAY[0], end=MAY[-1], assets=["BTCUSDT"], data_root=tmp_path)
    with pytest.raises(ValueError, match="cut_ms needs a sink"):
        chunked.write_dataset_month(cfg, ["BTCUSDT"], "time_1h", 2024, 5, {}, cut_ms=1)
    with pytest.raises(ValueError, match="cut_ms needs a sink"):
        chunked.write_fx_features_month(cfg, ["EURUSD"], "time_1h", 2024, 5, cut_ms=1)


def test_code_identity_names_the_commit_and_the_changes():
    ident = code_identity()
    assert set(ident) == {"commit", "branch", "dirty", "changed"}
    if ident["commit"] is not None:
        assert len(ident["commit"]) == 40 and isinstance(ident["dirty"], bool)
        assert ident["dirty"] == bool(ident["changed"])


def test_certify_cli_needs_assets_and_keeps_its_flags(capsys):
    args = build_parser().parse_args(["certify", "--start", "2020-01-01", "--end", "2020-12-31", "--universe", "u10",
                                      "--no-rebuild", "--refresh"])
    assert (args.cmd, args.no_rebuild, args.refresh, args.skip_network) == ("certify", True, True, False)
    assert main(["certify", "--start", "2020-01-01", "--end", "2020-01-02"]) == 2
    assert "certify needs --assets or --universe" in capsys.readouterr().err
