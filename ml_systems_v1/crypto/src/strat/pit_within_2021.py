"""src/strat/pit_within_2021.py -- POINT-IN-TIME 2021 within-year: the universe is NOT fixed u10; it admits
every coin that LISTED by/during 2021 (data-derived listing date), and tracks WHICH 2021 window (TRAIN/VAL/OOS)
each new listing falls into.

USER /orc 2026-06-17: "the asset universe does not remain u10 in 2021 -- there are new listings we can track,
whether they fall in the 2021 TRAIN/VAL/OOS window." This re-runs the 2021 within-year 6/3/3 deep-dive over the
EXPANDED point-in-time universe (survivorship-clean: listing date = first chimera 1d bar; admit if listed before
2022-01-01) and produces a NEW-LISTINGS register (coin -> listing date -> which window -> bars per window), plus a
PIT-vs-u10 candidate comparison (does expanding the universe change the per-family/type candidate picture?).

Reuses deep_ti_within_year.run_year (6/3/3, fixed-EW, ironed sleeve) with the universe overridden to the PIT set.
A coin listing mid-2021 appears only from its listing date (fixed-EW: cash before) -> if it lists AFTER the TRAIN
window it cannot clear the TRAIN&VAL>0 robust bar (flagged). Long-only, maker, NO 2022. No emoji.

RWYB: python -m strat.pit_within_2021 --tfs 1d,4h,2h,1h
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.deep2020_ti_pipeline as TI                                      # noqa: E402
import strat.deep_ti_within_year as WY                                       # noqa: E402
from strat.ma_2020_breakdown import _panel                                   # noqa: E402

DEST = ROOT.parent / "runs" / "periods" / "TRAIN" / "2021" / "DEEP_DIVE"
U10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
WIN_SPLITS = {"TRAIN": ("2021-01-01", "2021-07-01"), "VAL": ("2021-07-01", "2021-10-01"),
              "OOS": ("2021-10-01", "2022-01-01")}
LIST_CUTOFF = "2022-01-01"                                                   # admit coins listed by/during 2021


def _available():
    import glob
    syms = set()
    for f in glob.glob("data/processed/chimera/1d/*usdt*.parquet"):
        syms.add(Path(f).stem.split("_")[0].upper())
    return sorted(syms)


def pit_universe():
    """Coins listed by/during 2021 (first 1d bar < 2022). Returns (admitted [(sym, first_ms)], excluded, register)."""
    cut = pd.Timestamp(LIST_CUTOFF).value // 10**6
    admitted, excluded, register = [], [], []
    for sym in _available():
        try:
            o, h, l, c, ms = _panel(sym, "1d")
        except Exception:
            continue
        if len(ms) < 5:
            continue
        first = int(ms[0])
        if first >= cut:
            excluded.append((sym, first)); continue
        admitted.append((sym, first))
        # which 2021 window does the listing fall in? (bars per window in 2021)
        fd = pd.Timestamp(first, unit="ms")
        win = "pre-2021" if fd < pd.Timestamp("2021-01-01") else (
            "TRAIN" if fd < pd.Timestamp("2021-07-01") else (
                "VAL" if fd < pd.Timestamp("2021-10-01") else "OOS"))
        bars = {}
        for wk, (lo, hi) in WIN_SPLITS.items():
            lo_ms = pd.Timestamp(lo).value // 10**6; hi_ms = pd.Timestamp(hi).value // 10**6
            bars[wk] = int(((ms >= lo_ms) & (ms < hi_ms)).sum())
        register.append({"sym": sym, "listing": str(fd.date()), "lists_in": win, "is_new_2021": sym not in U10
                         and fd >= pd.Timestamp("2021-01-01"), "bars_2021": bars})
    admitted.sort(key=lambda x: x[1])
    return admitted, excluded, register


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.pit_within_2021")
    ap.add_argument("--tfs", default="1d,4h,2h,1h")
    a = ap.parse_args(argv)
    tfs = [t.strip() for t in a.tfs.split(",") if t.strip()]
    admitted, excluded, register = pit_universe()
    pit_syms = [s for s, _ in admitted]
    new_2021 = [r for r in register if r["is_new_2021"]]
    print(f"## POINT-IN-TIME 2021 universe: {len(pit_syms)} admitted (listed by 2021), {len(excluded)} excluded "
          f"(listed 2022+). u10 baseline = {len(U10)}.")
    print(f"   NEW 2021 listings ({len(new_2021)}): " +
          ", ".join(f"{r['sym']}({r['listing']},{r['lists_in']})" for r in new_2021[:30]) +
          (" ..." if len(new_2021) > 30 else ""))
    # window distribution of new listings
    from collections import Counter
    wd = Counter(r["lists_in"] for r in new_2021)
    print(f"   new-listing window distribution: {dict(wd)} "
          f"(coins listing in VAL/OOS have NO TRAIN data -> cannot clear the TRAIN&VAL>0 robust bar -- flagged)")
    # OVERRIDE the within-year universe to the PIT set + run 2021 (tag _pit so u10 within_2021.json is preserved)
    TI.SYMS = pit_syms
    print(f"\n## running 2021 within-year 6/3/3 over the PIT universe ({len(pit_syms)} coins), tfs={tfs} ...\n")
    WY.run_year(2021, tfs, tag="_pit")
    # save the universe register
    DEST.mkdir(parents=True, exist_ok=True)
    json.dump({"admitted": [{"sym": s, "listing": str(pd.Timestamp(m, unit="ms").date())} for s, m in admitted],
               "excluded": [s for s, _ in excluded], "new_2021_register": register,
               "u10": U10, "splits": WIN_SPLITS},
              open(DEST / "pit_universe_2021.json", "w"), indent=1, default=str)
    print(f"\n[universe register] {DEST / 'pit_universe_2021.json'}")
    print(f"[within-year PIT] runs/strat/within_year/within_2021_pit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
