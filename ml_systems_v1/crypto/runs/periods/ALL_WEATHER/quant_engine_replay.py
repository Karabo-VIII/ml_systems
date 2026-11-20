"""Replay the 4 named engine cells (ADX/MACD/DONCHIAN/SUPERTREND 1d) CONTINUOUSLY over
2020-10-01..2022-12-31, gated, taker -- to confirm the finding's '+184..200% / -13..-19% DD'.

Band selection mirrors run_ti_cell: configs positive on 2020 TRAIN & VAL (gated), EW ensemble;
then the FROZEN band is replayed continuously. Reports wealth + maxDD on the continuous span.
No emoji (cp1252).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(ROOT))

import strat.ma_strat_builder as msb
import strat.ti_capture_sweep as tcs
from strat.deep2020_ti_pipeline import INDICATORS

SPAN = ("2020-10-01", "2023-01-01")
COST = tcs.COST


def _ms(span):
    return (int(pd.Timestamp(span[0]).value // 10**6), int(pd.Timestamp(span[1]).value // 10**6))


def _maxdd(book):
    if book is None:
        return None
    x = book.dropna().to_numpy()
    if len(x) < 2:
        return 0.0
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return round(float(((eq - pk) / pk).min() * 100.0), 1)


def select_band(ti, cad="1d", gate=True):
    """mirror run_ti_cell band selection on 2020 TRAIN+VAL; return (variant, band list)."""
    spec = INDICATORS[ti]
    grid = spec["grid"]()
    native_mh = int(spec.get("minhold", 12))
    a2020 = msb._load_all(cad, "2020-01-01", "2021-01-01")
    tr = _ms(msb.TRAIN); vl = _ms(msb.VAL)
    best = None
    for variant in ("base", "iron"):
        held_fn = spec[variant]
        cfg_nets = []
        for params in grid:
            for cd in tcs.COOLDOWN_GRID:
                btr = tcs._cfg_book(a2020, held_fn, params, gate, cd, native_mh, *tr)
                bvl = tcs._cfg_book(a2020, held_fn, params, gate, cd, native_mh, *vl)
                ntr, nvl = tcs._netpct(btr), tcs._netpct(bvl)
                if ntr is None or nvl is None:
                    continue
                cfg_nets.append({"params": params, "cd": cd, "ntr": ntr, "nvl": nvl})
        band = [x for x in cfg_nets if x["ntr"] > 0 and x["nvl"] > 0]
        if not band:
            band = sorted(cfg_nets, key=lambda x: -(x["ntr"] + x["nvl"]))[:3]
        if not band:
            continue
        score = sum(b["ntr"] + b["nvl"] for b in band)
        cand = {"variant": variant, "band": band, "score": score}
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def replay_continuous(ti, cad="1d", gate=True):
    spec = INDICATORS[ti]
    native_mh = int(spec.get("minhold", 12))
    sel = select_band(ti, cad, gate)
    if sel is None:
        return None, None, 0, "?"
    held_fn = spec[sel["variant"]]
    af = msb._load_all(cad, SPAN[0], SPAN[1])
    lo, hi = _ms(SPAN)
    bk = tcs._ew([tcs._cfg_book(af, held_fn, b["params"], gate, b["cd"], native_mh, lo, hi)
                  for b in sel["band"]])
    return tcs._netpct(bk), _maxdd(bk), len(sel["band"]), sel["variant"]


def main():
    print("## Engine cells replayed CONTINUOUSLY 2020-10..2022-12 (gated, taker)")
    print(f"  {'cell':12s}{'var':6s}{'nband':>6s}{'wealth%':>10s}{'maxDD%':>9s}")
    for ti in ("ADX", "MACD", "DONCHIAN", "SUPERTREND"):
        w, dd, nb, var = replay_continuous(ti, "1d", True)
        print(f"  {ti:12s}{var:6s}{nb:>6d}{('' if w is None else f'{w:>10.1f}')}{('' if dd is None else f'{dd:>9.1f}')}")


if __name__ == "__main__":
    main()
