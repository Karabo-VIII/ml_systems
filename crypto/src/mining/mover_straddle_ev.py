"""MOVER-STRADDLE EV -- the options monetization go/no-go (data IN HAND, pre-ingest).

THE DECISIVE QUESTION (answerable before spending ~$700/mo on Deribit options ingest):
  The mover sprint converged: the DIRECTION of daily movers is information-bound dead
  (4 ways), but MAGNITUDE is capturable -- a fizzle-filter (OOS AUC ~0.70) raises the
  +1.5%-trigger -> >=5%-mover conversion 31% -> ~58%, and magnitude-continuation is
  predictable (OOS AUC ~0.73). Both predict |move|, SIGN-AGNOSTIC. The only vehicle for a
  sign-agnostic magnitude bet is a STRADDLE. Perp-straddles are a proven trap (D64/D65:
  spot+perp has ZERO gamma + cost), so the real vehicle is OPTIONS, which the project
  lacks. Buying volatility normally LOSES (VRP: IV>RV ~71-81% of days -> sellers win).

  So: does the FILTERED big-move cohort's REALIZED |move| exceed a FAIR option premium --
  i.e. is BUYING magnitude (an ATM straddle) +EV on the movers, when it is -EV on a
  random day? If yes on FILTERED and -EV UNFILTERED, the filter adds the edge and the
  ingest is justified (buy-magnitude-on-movers). If realized < implied even on the
  filtered movers, buying movers is -EV and the only options edge is SELLING VRP on calm
  names.

THE MODEL (conservative + adversarial; the default expectation is buyers LOSE):
  - Cohort entry = the lane-#4 +1.5%-from-day-open trigger (reuses
    mover_burst_timing.trigger_events: causal features at trigger close + the fizzle
    label is_mover = day-runup>=5%). The fizzle-filter (HGB on those features, fit
    TRAIN-only) gives a per-event magnitude SCORE; FILTERED = top-quantile score.
  - Horizon H = the natural mover hold (1 and 3 calendar days from the trigger fill).
  - ATM STRADDLE PAYOFF (held to expiry, sign-agnostic) = |S_{t+H}/S_t - 1|.
  - FAIR PREMIUM, the honest anchor (Brenner-Subrahmanyam ATM approx):
        premium ~= 0.7979 * IV_annual * sqrt(H/365)
    where IV_annual is the at-entry implied vol:
      * BTC/ETH: REAL DVOL (Deribit DVOL index, daily) on the trigger date -- the actual
        option market's price. No estimation error.
      * the other 8 assets: IV = RV30_annual * (1 + m), RV30 = trailing-30d realized
        annualized vol at the trigger; m = the empirical VRP markup, CALIBRATED from the
        BTC/ETH (DVOL / RV30) ratio (so the modeled-IV alt premium reproduces the known
        VRP). The estimation error is stated; an alt-only and a BTC/ETH-only verdict are
        both reported so the conclusion never rests on the modeled leg alone.
  - STRADDLE EV = E[|move|] - premium - option_spread_cost. Crypto option bid/ask is wide;
    spread modeled as a fraction of premium (5% / 10% / 20% sensitivity).

HONEST HELD-OUT:
  Filter + z-stats + quantile cutoffs + the alt VRP markup m fit on TRAIN (<2024-01-01)
  ONLY. OOS 2024-01-01..2025-07-01 and UNSEEN >=2025-07-01 scored separately; the verdict
  reports OOS and UNSEEN side by side. Random-day VRP baseline (the floor the filter must
  beat) computed on BTC/ETH with real DVOL.

Run:
  python -m mining.mover_straddle_ev --universe u10
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mining.cascade_oracle import load_panel  # noqa: E402
from mining.mover_burst_timing import (  # noqa: E402
    trigger_events, FIZZLE_FEATS, _zscore_per_asset, _norm_sym,
    TRAIN_END_MS, OOS_END_MS,
)

OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)
DVOL_PATH = ROOT / "data" / "raw_external" / "deribit" / "dvol_daily.parquet"

__contract__ = {
    "kind": "research",
    "inputs": {
        "liq_subbar_1m": "via cascade_oracle.load_panel (1m grid)",
        "trigger_cohort": "via mover_burst_timing.trigger_events (fizzle features + label)",
        "dvol_daily": "data/raw_external/deribit/dvol_daily.parquet (BTC/ETH real IV)",
    },
    "outputs": {"study_json": "runs/mining/mover_straddle_ev_<tag>_<stamp>.json"},
    "invariants": {
        "train_only_fit": "filter, z-stats, quantile cutoffs, alt VRP markup m ALL TRAIN-only",
        "real_iv_where_available": "BTC/ETH premium uses REAL DVOL on the trigger date",
        "modeled_iv_stated": "alt IV = RV30*(1+m); m from BTC/ETH DVOL/RV; alt-only verdict given",
        "premium_priced_at_entry": "IV is the at-entry vol -- the honest VRP-vs-realized test",
        "held_out": "OOS and UNSEEN scored separately; filter never sees them in fit",
        "vrp_baseline": "random-day buyer EV on BTC/ETH (real DVOL) = the floor to beat",
    },
}

DAY_MS = 86_400_000
BS_ATM = 0.7978845608  # sqrt(2/pi): Brenner-Subrahmanyam ATM straddle coefficient
HORIZONS_D = [1, 3]            # natural mover hold (calendar days from trigger fill)
SPREAD_FRACS = [0.05, 0.10, 0.20]  # option bid/ask as a fraction of premium (sensitivity)
TOP_QUANTILES = [0.30, 0.10]  # FILTERED = top-q fizzle-filter score
RV_WIN_D = 30                 # trailing realized-vol window for alt IV (days)
SEED = 7

EPOCH = dt.date(1970, 1, 1)


def split_of(ms: int) -> str:
    return "TRAIN" if ms < TRAIN_END_MS else ("OOS" if ms < OOS_END_MS else "UNSEEN")


# ----------------------------------------------------------------- DVOL (real IV)

def load_dvol() -> dict:
    """{asset -> {day_index -> dvol_close/100}} for BTC/ETH (real annualized IV)."""
    if not DVOL_PATH.exists():
        return {}
    df = pl.read_parquet(DVOL_PATH).with_columns(pl.col("date").cast(pl.Date))
    out: dict[str, dict[int, float]] = {}
    for asset in ["BTC", "ETH"]:
        sub = df.filter(pl.col("asset") == asset)
        m = {}
        for r in sub.iter_rows(named=True):
            di = int((r["date"] - EPOCH).days)
            if r["dvol_close"] is not None:
                m[di] = float(r["dvol_close"]) / 100.0
        out[asset] = m
    return out


# ----------------------------------------------------------------- daily price/RV

def daily_series(sym: str):
    """Daily last-close + trailing-30d annualized realized vol, keyed by UTC day-index.
    RV at day d uses returns strictly < day d (causal: trailing, no same-day)."""
    df = load_panel(sym)
    d = (df.with_columns((pl.col("minute_ts") // DAY_MS).alias("d"))
         .group_by("d").agg(pl.col("close").last().alias("close"))
         .sort("d"))
    dd = d["d"].to_numpy().astype(np.int64)
    dc = d["close"].to_numpy().astype(float)
    logret = np.full(len(dc), np.nan)
    logret[1:] = np.diff(np.log(dc))
    rv = np.full(len(dc), np.nan)  # annualized RV usable AT day dd[i] (uses returns up to i-1)
    for i in range(RV_WIN_D + 1, len(dc)):
        seg = logret[i - RV_WIN_D:i]          # returns ending the prior day (strictly < i)
        seg = seg[np.isfinite(seg)]
        if len(seg) >= RV_WIN_D // 2:
            rv[i] = float(np.std(seg)) * np.sqrt(365.0)
    close_at = {int(dd[i]): dc[i] for i in range(len(dc))}
    rv_at = {int(dd[i]): rv[i] for i in range(len(dc)) if np.isfinite(rv[i])}
    return close_at, rv_at, dd, dc


def fwd_abs_move(close_at: dict, day_idx: int, H: int) -> float | None:
    """|S_{d+H}/S_d - 1| using daily closes (the ATM straddle terminal payoff)."""
    c0 = close_at.get(day_idx)
    cH = close_at.get(day_idx + H)
    if c0 is None or cH is None or c0 <= 0:
        return None
    return abs(cH / c0 - 1.0)


def premium_from_iv(iv_annual: float, H: int) -> float:
    """Brenner-Subrahmanyam ATM straddle premium as a fraction of spot."""
    return BS_ATM * iv_annual * np.sqrt(H / 365.0)


# ----------------------------------------------------------------- the fizzle filter score

def fit_filter_scores(events: list[dict], seed: int) -> None:
    """Fit the HGB magnitude fizzle-filter on TRAIN ONLY; write a 'filt_score' onto every
    event (TRAIN/OOS/UNSEEN) -- TRAIN out-of-fold via 5-fold, OOS/UNSEEN by the full-TRAIN
    model. Mirrors mover_burst_timing's modeling (same features, per-asset z, early stop)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    tr = [e for e in events if e["split"] == "TRAIN"]
    rest = [e for e in events if e["split"] != "TRAIN"]
    ytr = np.array([e["is_mover"] for e in tr], dtype=int)

    def new_hgb(rs):
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=20, random_state=rs)

    # per-asset z (TRAIN stats) for TRAIN and for the held-out rest
    Ztr, Zrest = _zscore_per_asset(tr, rest, FIZZLE_FEATS)

    # OOS/UNSEEN scores from a model fit on ALL train
    full = new_hgb(seed)
    full.fit(Ztr, ytr)
    p_rest = full.predict_proba(Zrest)[:, 1] if len(rest) else np.array([])
    for e, p in zip(rest, p_rest):
        e["filt_score"] = float(p)

    # TRAIN out-of-fold scores (so a FILTERED-on-TRAIN view is honest too)
    oof = np.full(len(tr), np.nan)
    if len(np.unique(ytr)) >= 2:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (i_in, i_out) in enumerate(skf.split(Ztr, ytr)):
            mdl = new_hgb(seed + fold)
            mdl.fit(Ztr[i_in], ytr[i_in])
            oof[i_out] = mdl.predict_proba(Ztr[i_out])[:, 1]
    for e, s in zip(tr, oof):
        e["filt_score"] = float(s) if np.isfinite(s) else np.nan


# ----------------------------------------------------------------- EV engine

def cohort_ev(rows: list[dict], H: int, spread_frac: float) -> dict:
    """rows carry 'absmove' (realized payoff) + 'premium' (fair). Buyer EV = move - prem -
    spread; spread = spread_frac * premium (round-trip-ish; conservative)."""
    if not rows:
        return {"n": 0}
    move = np.array([r["absmove"] for r in rows])
    prem = np.array([r["premium"] for r in rows])
    cost = spread_frac * prem
    ev = move - prem - cost
    return {
        "n": len(rows),
        "mean_abs_move": float(move.mean()),
        "mean_premium": float(prem.mean()),
        "mean_spread_cost": float(cost.mean()),
        "buyer_ev": float(ev.mean()),
        "buyer_ev_pct_of_premium": float(ev.mean() / prem.mean()) if prem.mean() > 0 else None,
        "buyer_win_rate": float(np.mean(ev > 0)),
        "median_abs_move": float(np.median(move)),
        "median_premium": float(np.median(prem)),
        "rv_realized_over_implied": float(move.mean() / prem.mean()) if prem.mean() > 0 else None,
    }


def build_priced_rows(events: list[dict], close_at_by: dict, rv_at_by: dict,
                      dvol: dict, alt_markup: float, H: int) -> list[dict]:
    """Attach realized |move| over H and the fair premium (real DVOL for BTC/ETH, modeled
    for alts) to each event that has the forward window + an IV. Drops events w/o either."""
    rows = []
    for e in events:
        sym = e["sym"]
        base = sym[:-4]  # strip USDT
        day_idx = int(e["ms"] // DAY_MS)
        close_at = close_at_by[sym]
        absmove = fwd_abs_move(close_at, day_idx, H)
        if absmove is None:
            continue
        iv = None
        iv_src = None
        if base in ("BTC", "ETH") and base in dvol and day_idx in dvol[base]:
            iv = dvol[base][day_idx]
            iv_src = "dvol_real"
        else:
            rv = rv_at_by[sym].get(day_idx)
            if rv is not None and np.isfinite(rv):
                iv = rv * (1.0 + alt_markup)
                iv_src = "rv_modeled"
        if iv is None or not np.isfinite(iv) or iv <= 0:
            continue
        rows.append({
            "sym": sym, "split": e["split"], "is_mover": e["is_mover"],
            "filt_score": e.get("filt_score", np.nan),
            "absmove": absmove, "premium": premium_from_iv(iv, H), "iv_src": iv_src,
        })
    return rows


def calibrate_alt_markup(close_at_by: dict, rv_at_by: dict, dvol: dict) -> dict:
    """m = median(DVOL / RV30) over BTC/ETH TRAIN days (the empirical VRP markup applied to
    alts). TRAIN-only, both assets pooled. Returns the markup + diagnostics."""
    ratios = []
    per_asset = {}
    for base, sym in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]:
        if base not in dvol or sym not in rv_at_by:
            continue
        rr = []
        for di, ivd in dvol[base].items():
            ms = di * DAY_MS
            if ms >= TRAIN_END_MS:        # TRAIN-only calibration
                continue
            rv = rv_at_by[sym].get(di)
            if rv is not None and np.isfinite(rv) and rv > 0:
                rr.append(ivd / rv)
        if rr:
            per_asset[base] = {"n": len(rr), "median_ratio": float(np.median(rr)),
                               "mean_ratio": float(np.mean(rr)),
                               "iv_gt_rv_frac": float(np.mean(np.array(rr) > 1.0))}
            ratios.extend(rr)
    m = float(np.median(ratios)) - 1.0 if ratios else 0.15
    return {"alt_markup_m": m, "n_calib_days": len(ratios), "per_asset": per_asset,
            "note": "m = median(DVOL/RV30) - 1 over BTC+ETH TRAIN days; applied to alts"}


def random_day_vrp_baseline(close_at_by: dict, dvol: dict, H: int) -> dict:
    """The floor the filter must beat: buy an ATM straddle on a RANDOM day on BTC/ETH at
    REAL DVOL, hold H days. Buyer EV (no spread). Per split."""
    out = {}
    for split in ["OOS", "UNSEEN"]:
        moves, prems = [], []
        for base, sym in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]:
            if base not in dvol or sym not in close_at_by:
                continue
            close_at = close_at_by[sym]
            for di, ivd in dvol[base].items():
                ms = di * DAY_MS
                if split_of(ms) != split:
                    continue
                mv = fwd_abs_move(close_at, di, H)
                if mv is None:
                    continue
                moves.append(mv)
                prems.append(premium_from_iv(ivd, H))
        if moves:
            moves = np.array(moves); prems = np.array(prems)
            ev = moves - prems
            out[split] = {"n": len(moves), "mean_abs_move": float(moves.mean()),
                          "mean_premium": float(prems.mean()), "buyer_ev_no_spread": float(ev.mean()),
                          "buyer_win_rate": float(np.mean(ev > 0))}
    return out


# ----------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser(description="Mover-straddle EV: buy-magnitude go/no-go (held-out)")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.assets:
        syms = [_norm_sym(a) for a in args.assets]
        tag = args.tag or "_".join(s[:-4] for s in syms[:4]).lower()
    elif args.universe:
        spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{args.universe}.yaml"))
        syms = [a["symbol"] for a in spec["assets"]]
        tag = args.tag or args.universe
    else:
        ap.error("provide --assets or --universe")

    t0 = time.time()
    dvol = load_dvol()
    print(("DVOL loaded: " + "+".join(dvol.keys()) + " (real IV anchor)") if dvol else "DVOL MISSING")

    # gather trigger cohort + daily series per asset
    all_trig: list[dict] = []
    close_at_by: dict = {}
    rv_at_by: dict = {}
    for sym in syms:
        try:
            tev = trigger_events(sym)
            close_at, rv_at, _, _ = daily_series(sym)
        except FileNotFoundError as e:
            print(f"[{sym}] SKIP: {e}")
            continue
        close_at_by[sym] = close_at
        rv_at_by[sym] = rv_at
        all_trig.extend(tev)
        n_tr = sum(1 for e in tev if e["split"] == "TRAIN")
        n_oo = sum(1 for e in tev if e["split"] == "OOS")
        n_un = sum(1 for e in tev if e["split"] == "UNSEEN")
        print(f"[{sym}] triggers T/O/U={n_tr}/{n_oo}/{n_un} ({time.time()-t0:.0f}s)")

    # NOTE: trigger_events() drops UNSEEN by design (split=='UNSEEN' -> continue). To price
    # the UNSEEN cohort we re-include it via a parallel pass with the UNSEEN guard relaxed.
    all_trig.extend(_unseen_triggers(syms))

    if not all_trig:
        print("no triggers -- abort")
        return 1

    n_assets = len({e["sym"] for e in all_trig})
    print(f"\ntotal triggers: {len(all_trig)} over {n_assets} assets")

    # fit the magnitude fizzle-filter (TRAIN-only) and score every event
    fit_filter_scores(all_trig, args.seed)

    # calibrate the alt VRP markup from BTC/ETH TRAIN (DVOL/RV)
    calib = calibrate_alt_markup(close_at_by, rv_at_by, dvol)
    m = calib["alt_markup_m"]
    print(f"alt VRP markup m = {m:+.3f} (from {calib['n_calib_days']} BTC/ETH TRAIN days)")

    results = {}
    for H in HORIZONS_D:
        priced = build_priced_rows(all_trig, close_at_by, rv_at_by, dvol, m, H)
        baseline = random_day_vrp_baseline(close_at_by, dvol, H)
        per_split = {}
        for split in ["OOS", "UNSEEN"]:
            sp_rows = [r for r in priced if r["split"] == split]
            if not sp_rows:
                continue
            # rank by filter score within this split
            scored = [r for r in sp_rows if np.isfinite(r["filt_score"])]
            order = sorted(scored, key=lambda r: -r["filt_score"])
            split_block = {"n_total": len(sp_rows), "n_scored": len(scored)}
            for sf in SPREAD_FRACS:
                key = f"spread_{int(sf*100)}pct"
                blk = {"UNFILTERED_all_triggers": cohort_ev(sp_rows, H, sf)}
                for q in TOP_QUANTILES:
                    k = max(1, int(round(q * len(order))))
                    blk[f"FILTERED_top_{int(q*100)}pct"] = cohort_ev(order[:k], H, sf)
                # also: the calm/low-magnitude bottom cohort (the SELL-VRP candidate)
                kbot = max(1, int(round(0.30 * len(order))))
                blk["BOTTOM_30pct_calm"] = cohort_ev(order[-kbot:], H, sf)
                split_block[key] = blk
            # conversion check: mover-rate by cohort (sanity the filter is selecting movers)
            if scored:
                k30 = max(1, int(round(0.30 * len(order))))
                k10 = max(1, int(round(0.10 * len(order))))
                split_block["conversion"] = {
                    "all": float(np.mean([r["is_mover"] for r in sp_rows])),
                    "top30": float(np.mean([r["is_mover"] for r in order[:k30]])),
                    "top10": float(np.mean([r["is_mover"] for r in order[:k10]])),
                }
            per_split[split] = split_block
        results[f"H{H}d"] = {
            "random_day_vrp_baseline": baseline,
            "n_priced_total": len(priced),
            "iv_src_counts": _count_iv_src(priced),
            "cohorts": per_split,
        }

    # ------------------------------------------------------------- verdict
    verdict = build_verdict(results)

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT).stdout.strip()
    except Exception:
        sha = "unknown"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "tool": "mover_straddle_ev", "git_sha": sha, "seed": args.seed,
        "params": {"horizons_d": HORIZONS_D, "spread_fracs": SPREAD_FRACS,
                   "top_quantiles": TOP_QUANTILES, "rv_win_d": RV_WIN_D,
                   "bs_atm_coef": BS_ATM, "fizzle_feats": FIZZLE_FEATS},
        "n_assets": n_assets, "alt_markup_calib": calib,
        "results": results, "verdict": verdict,
        "caveats": [
            "BTC/ETH premium uses REAL DVOL (no estimation error); alts use IV=RV30*(1+m), "
            "m calibrated from BTC/ETH -- alt premia carry model error (alt-only verdict given)",
            "ATM straddle premium via Brenner-Subrahmanyam (0.7979*IV*sqrt(T)); a full BS "
            "ATM straddle is within ~1-2% of this for short T -- conservative for the buyer test",
            "payoff = |daily-close move over H| (held-to-expiry straddle); intraday gamma "
            "scalping / early exit NOT modeled -- a pure terminal-payoff lower bound for the buyer",
            "option spread modeled 5/10/20% of premium; real crypto-option spreads can exceed "
            "20% on alts -> the alt buyer case is OPTIMISTIC at 5-10%",
            "u10 = CURRENT membership (survivorship); per-asset z; UNSEEN scored once here",
            "DVOL coverage BTC/ETH from 2021-03-24; pre-date events use modeled IV if alt, dropped if BTC/ETH",
        ],
    }
    out_path = OUT / f"mover_straddle_ev_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    _print_story(results, verdict, calib, out_path, time.time() - t0)
    return 0


def _unseen_triggers(syms: list[str]) -> list[dict]:
    """Re-run the trigger extraction but KEEP the UNSEEN split (trigger_events drops it).
    Identical mechanics; only the split guard differs. Kept separate so the TRAIN/OOS path
    is byte-for-byte the audited trigger_events()."""
    import mining.mover_burst_timing as mbt
    out = []
    for sym in syms:
        try:
            P = mbt._panel_arrays(sym)
        except FileNotFoundError:
            continue
        ms, opens, closes, highs, real = P["ms"], P["open"], P["close"], P["high"], P["real"]
        n = len(ms)
        day_ids = ms // DAY_MS
        day_starts = np.flatnonzero(np.diff(day_ids, prepend=day_ids[0] - 1))
        day_ends = np.append(day_starts[1:] - 1, n - 1)
        for ds, de in zip(day_starts, day_ends):
            if de - ds < 1200 or real[ds:de + 1].mean() < 0.90:
                continue
            d_open = opens[ds]
            if not np.isfinite(d_open) or d_open <= 0:
                continue
            if split_of(int(ms[ds])) != "UNSEEN":
                continue
            rel = closes[ds:de + 1] / d_open - 1.0
            hit = np.flatnonzero(rel >= mbt.T_TRIG)
            if len(hit) == 0:
                continue
            mm = ds + int(hit[0]); f = mm + 1
            if de - f < mbt.MIN_MINUTES_LEFT or not (real[mm] and real[f]):
                continue
            entry = opens[f]
            if not np.isfinite(entry) or entry <= 0:
                continue
            day_runup = float(np.max(highs[ds:de + 1]) / d_open - 1.0)
            is_mover = int(day_runup >= mbt.MOVER_RUNUP)
            feats = mbt._micro_features(P, mm, int(ds))
            feats["overshoot"] = float(rel[mm - ds] - mbt.T_TRIG)
            feats["t2trig"] = float(mm - ds)
            feats["day_run_pace"] = float(rel[mm - ds] / max(mm - ds, 1))
            ret30 = float(closes[mm] / closes[max(ds, mm - 30)] - 1.0)
            feats["run_accel"] = (ret30 / 30.0) / feats["day_run_pace"] if feats["day_run_pace"] != 0 else np.nan
            dvol_s = float(np.nanstd(P["ret1m"][ds:mm + 1])) if mm - ds >= 30 else np.nan
            pv = P["pre_vol"][mm]
            feats["dayvol_ratio"] = (dvol_s / pv) if (np.isfinite(pv) and pv > 0 and np.isfinite(dvol_s)) else np.nan
            out.append({"sym": sym, "ms": int(ms[mm]), "split": "UNSEEN",
                        "is_mover": is_mover, "day_runup": day_runup,
                        "ride_gross": 0.0, "feats": feats})
    return out


def _count_iv_src(priced: list[dict]) -> dict:
    c = {"dvol_real": 0, "rv_modeled": 0}
    for r in priced:
        c[r["iv_src"]] = c.get(r["iv_src"], 0) + 1
    return c


def build_verdict(results: dict) -> dict:
    """Decisive go/no-go. At the canonical 10% spread:
      buy-magnitude-on-movers is +EV iff FILTERED top-10% buyer EV > 0 on OOS AND UNSEEN.
      The filter ADDS edge iff FILTERED EV > UNFILTERED EV (and ideally UNFILTERED <= 0).
      sell-VRP-on-calm is +EV iff the BOTTOM-30pct calm cohort buyer EV < 0 (seller wins)."""
    v = {}
    for hk, hb in results.items():
        cohorts = hb.get("cohorts", {})
        row = {}
        for split in ["OOS", "UNSEEN"]:
            sb = cohorts.get(split, {}).get("spread_10pct")
            if not sb:
                continue
            top10 = sb.get("FILTERED_top_10pct", {})
            top30 = sb.get("FILTERED_top_30pct", {})
            allc = sb.get("UNFILTERED_all_triggers", {})
            calm = sb.get("BOTTOM_30pct_calm", {})
            row[split] = {
                "filtered_top10_buyer_ev": top10.get("buyer_ev"),
                "filtered_top30_buyer_ev": top30.get("buyer_ev"),
                "unfiltered_buyer_ev": allc.get("buyer_ev"),
                "calm_bottom30_buyer_ev": calm.get("buyer_ev"),
                "filter_adds_edge": (top10.get("buyer_ev") is not None and allc.get("buyer_ev") is not None
                                     and top10["buyer_ev"] > allc["buyer_ev"]),
                "buy_magnitude_positive": bool((top10.get("buyer_ev") or -1) > 0),
                "sell_vrp_on_calm_positive": bool((calm.get("buyer_ev") or 1) < 0),
            }
        # cross-split decisiveness
        oos = row.get("OOS", {}); uns = row.get("UNSEEN", {})
        row["DECISION"] = {
            "buy_magnitude_GO": bool(oos.get("buy_magnitude_positive") and uns.get("buy_magnitude_positive")),
            "filter_adds_edge_both": bool(oos.get("filter_adds_edge") and uns.get("filter_adds_edge")),
            "sell_vrp_GO": bool(oos.get("sell_vrp_on_calm_positive") and uns.get("sell_vrp_on_calm_positive")),
        }
        v[hk] = row
    return v


def _print_story(results, verdict, calib, out_path, secs):
    print("\n" + "=" * 84)
    print("MOVER-STRADDLE EV -- is BUYING magnitude (a straddle) +EV on the movers? (held-out)")
    print("=" * 84)
    print(f"alt VRP markup m = {calib['alt_markup_m']:+.3f} (BTC/ETH DVOL/RV TRAIN); "
          f"BTC/ETH use REAL DVOL")
    for hk, hb in results.items():
        print(f"\n----- {hk}  (n_priced={hb['n_priced_total']}, "
              f"IV src real/modeled={hb['iv_src_counts'].get('dvol_real',0)}/"
              f"{hb['iv_src_counts'].get('rv_modeled',0)}) -----")
        base = hb.get("random_day_vrp_baseline", {})
        for split in ["OOS", "UNSEEN"]:
            b = base.get(split)
            if b:
                print(f"  [VRP floor] random-day buyer EV ({split}, real DVOL, no spread): "
                      f"{b['buyer_ev_no_spread']*100:+.3f}%  (|move| {b['mean_abs_move']*100:.2f}% "
                      f"vs prem {b['mean_premium']*100:.2f}%, win {b['buyer_win_rate']*100:.0f}%)")
        for split in ["OOS", "UNSEEN"]:
            sb = hb.get("cohorts", {}).get(split, {})
            s10 = sb.get("spread_10pct")
            if not s10:
                continue
            conv = sb.get("conversion", {})
            print(f"  [{split}] (10% option spread)  conversion all/top30/top10 = "
                  f"{conv.get('all',0)*100:.0f}%/{conv.get('top30',0)*100:.0f}%/{conv.get('top10',0)*100:.0f}%")
            for name in ["UNFILTERED_all_triggers", "FILTERED_top_30pct", "FILTERED_top_10pct", "BOTTOM_30pct_calm"]:
                c = s10.get(name, {})
                if c.get("n"):
                    print(f"      {name:<26} n={c['n']:<5} |move| {c['mean_abs_move']*100:5.2f}%  "
                          f"prem {c['mean_premium']*100:5.2f}%  EV {c['buyer_ev']*100:+6.3f}%  "
                          f"(RV/IV {c['rv_realized_over_implied']:.2f}, win {c['buyer_win_rate']*100:.0f}%)")
        dec = verdict.get(hk, {}).get("DECISION", {})
        print(f"  ==> {hk} DECISION: buy-magnitude GO (OOS&UNSEEN +EV)? {dec.get('buy_magnitude_GO')} | "
              f"filter-adds-edge both? {dec.get('filter_adds_edge_both')} | "
              f"sell-VRP-on-calm GO? {dec.get('sell_vrp_GO')}")
    print(f"\n({secs:.0f}s)  JSON -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
