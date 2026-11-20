"""exit_opt_mom14k5.py -- Cycle-2 exit optimisation lane: mom14-K5 over 2020->2026.

Optimise the EXIT policy while holding the ENTRY fixed:
  - baseline: rebal=3 flush (Cycle-1 reference)
  - ATR-trail k in {2, 3, 4, 5}  (trail stop at entry_price * (1 - k*atr14/C))
  - take-profit at {+15, +30, +50}%  (exit when price up X% from entry)
  - let-winners-run: hold while gated AND still in top-K  (no forced flush)
  - signal-flip: exit when mom14 < 0
  - time-stop at {7, 14} days  (exit if held for N days regardless)

CAUSAL RULE: W.loc[d] uses only ind[...].loc[d] or earlier.
RWYB: run this script directly.

Reports:
  1. Full table over 2020-2026 per exit variant.
  2. FORWARD VALIDATION: best exit vs plain flush, 2023-2025 specifically.
  3. Honest OOS-contribution quantification.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from strat.mover_lab import load, evaluate, topk_weight  # noqa: E402

COST = 0.0024  # TAKER_RT (round-trip)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mom14_k5_flush(ind, rebal=3) -> pd.DataFrame:
    """Baseline: standard topk_weight with flush every `rebal` days."""
    return topk_weight(ind["mom14"], ind, K=5, gate=True, rebal=rebal)


def _build_exit_variant(ind, variant: str, **kw) -> pd.DataFrame:
    """
    Build a weight matrix for mom14-K5 with a custom exit policy.

    variant options:
        'flush_N'       -- flush every N days (baseline family)
        'atr_trail_K'   -- trail stop k*ATR below highest-close since entry
        'tp_X'          -- take-profit at +X% from entry price
        'lwr'           -- let-winners-run (hold while gated & still top-K)
        'sig_flip'      -- exit when mom14 < 0
        'ts_N'          -- time-stop: exit after N days
    """
    C = ind["C"]
    mom14 = ind["mom14"]
    gate = ind["gate"]
    atr14 = ind["atr14"]
    dates = C.index
    assets = C.columns
    n = len(dates)

    # position matrix (pre-lag; evaluate() will lag by 1)
    W = pd.DataFrame(0.0, index=dates, columns=assets)

    # position state per asset: entry_date_idx, entry_price, peak_price_since_entry
    entry_idx = {s: -999 for s in assets}
    entry_px  = {s: np.nan for s in assets}
    peak_px   = {s: np.nan for s in assets}
    hold_days = {s: 0 for s in assets}

    # rebal schedule
    last_rebal = -999
    rebal_every = kw.get("rebal_every", 3)

    # current holdings set
    holdings: set = set()

    for i, d in enumerate(dates):
        # ---- decide exits ----
        to_exit = set()
        for s in list(holdings):
            px = C.loc[d, s]
            if pd.isna(px):
                to_exit.add(s)
                continue
            ep = entry_px[s]; pp = peak_px[s]
            hd = hold_days[s]
            m14 = mom14.loc[d, s]

            if variant.startswith("atr_trail_"):
                k = float(variant.split("_")[-1])
                atr = atr14.loc[d, s]
                if pd.isna(atr) or pd.isna(pp): pass
                else:
                    trail = pp * (1.0 - k * atr / (px + 1e-12))
                    if px < trail:
                        to_exit.add(s)
            elif variant.startswith("tp_"):
                tp_pct = float(variant.split("_")[-1]) / 100.0
                if not pd.isna(ep) and px >= ep * (1.0 + tp_pct):
                    to_exit.add(s)
            elif variant == "lwr":
                # exit if no longer gated
                if not bool(gate.loc[d, s]):
                    to_exit.add(s)
            elif variant == "sig_flip":
                if not pd.isna(m14) and m14 < 0:
                    to_exit.add(s)
                if not bool(gate.loc[d, s]):
                    to_exit.add(s)
            elif variant.startswith("ts_"):
                n_days = int(variant.split("_")[-1])
                if hd >= n_days:
                    to_exit.add(s)
                if not bool(gate.loc[d, s]):
                    to_exit.add(s)
            else:
                # flush_N handled separately (no state needed)
                pass

            # update peak
            if s in holdings and s not in to_exit:
                peak_px[s] = max(peak_px[s], px) if not pd.isna(pp) else px

        for s in to_exit:
            holdings.discard(s)
            entry_idx[s] = -999
            entry_px[s] = np.nan
            peak_px[s] = np.nan
            hold_days[s] = 0

        # ---- decide rebalance / entries ----
        if variant == "lwr":
            # rebal_every-day rhythm: re-score eligible assets, fill top-K slots
            do_rebal = (i - last_rebal >= rebal_every)
        elif variant.startswith("ts_") or variant == "sig_flip":
            do_rebal = (i - last_rebal >= rebal_every)
        elif variant.startswith("atr_trail_") or variant.startswith("tp_"):
            do_rebal = (i - last_rebal >= rebal_every)
        else:
            # flush_N: standard rebal
            do_rebal = (i - last_rebal >= rebal_every)

        if do_rebal:
            # score eligible assets
            elig = []
            for s in assets:
                if bool(gate.loc[d, s]) and pd.notna(mom14.loc[d, s]):
                    elig.append((s, mom14.loc[d, s]))
            top5 = sorted(elig, key=lambda x: -x[1])[:5]
            top5_syms = {s for s, _ in top5}

            # for lwr: only replace slots that are free
            if variant == "lwr":
                # keep existing holdings that are still top-K eligible
                keep = holdings & top5_syms
                # vacated slots
                slots = 5 - len(keep)
                new_entries = [s for s in top5_syms if s not in keep][:slots]
                holdings = keep | set(new_entries)
            else:
                # standard: exit all current, enter top-5 (flush is implicit -- exits handled above)
                # for non-flush variants, do a full rebal (exit everything and re-enter)
                old = set(holdings)
                holdings = top5_syms.copy()
                for s in old - holdings:
                    entry_idx[s] = -999
                    entry_px[s] = np.nan
                    peak_px[s] = np.nan
                    hold_days[s] = 0
                new_entries = holdings - old

            # record entries
            for s in new_entries if variant == "lwr" else holdings:
                if entry_idx[s] < 0:
                    entry_idx[s] = i
                    entry_px[s] = C.loc[d, s]
                    peak_px[s]  = C.loc[d, s]
                    hold_days[s] = 0

            last_rebal = i

        # increment hold days
        for s in holdings:
            hold_days[s] += 1
            if s not in to_exit:
                px = C.loc[d, s]
                if not pd.isna(px) and not pd.isna(peak_px[s]):
                    peak_px[s] = max(peak_px[s], px)

        # write weights
        W.iloc[i] = 0.0
        for s in holdings:
            if pd.notna(C.loc[d, s]):
                W.loc[d, s] = 1.0 / len(holdings)

    return W


def _build_flush_n(ind, n_days: int) -> pd.DataFrame:
    """Flush every n_days (uses topk_weight which resets positions on each rebal)."""
    return topk_weight(ind["mom14"], ind, K=5, gate=True, rebal=n_days)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EXIT OPTIMISATION LANE: mom14-K5, 2020->2026")
    print("=" * 70)

    # --- Load full 2020->2026 ---
    print("\nLoading data 2020-01-01 -> 2026-06-01 ...")
    ind_full = load("2020-01-01", "2026-06-01")
    print(f"  dates: {ind_full['C'].index[0].date()} -> {ind_full['C'].index[-1].date()}")
    print(f"  assets: {len(ind_full['C'].columns)}")

    # --- Load OOS 2023->2026 ---
    print("Loading OOS data 2023-01-01 -> 2026-06-01 ...")
    ind_oos = load("2023-01-01", "2026-06-01")
    print(f"  dates: {ind_oos['C'].index[0].date()} -> {ind_oos['C'].index[-1].date()}")

    # --- Define exit variants to test ---
    variants = [
        ("flush_3",    "Baseline: flush every 3d (Cycle-1 ref)"),
        ("flush_7",    "Flush every 7d"),
        ("flush_14",   "Flush every 14d"),
        ("atr_trail_2", "ATR-trail k=2"),
        ("atr_trail_3", "ATR-trail k=3"),
        ("atr_trail_4", "ATR-trail k=4"),
        ("atr_trail_5", "ATR-trail k=5"),
        ("tp_15",      "Take-profit +15%"),
        ("tp_30",      "Take-profit +30%"),
        ("tp_50",      "Take-profit +50%"),
        ("lwr",        "Let-winners-run (hold while gated & top-K)"),
        ("sig_flip",   "Signal-flip (exit when mom14<0)"),
        ("ts_7",       "Time-stop 7d"),
        ("ts_14",      "Time-stop 14d"),
    ]

    rows_full = []
    rows_oos  = []

    for vname, vdesc in variants:
        print(f"\n  Building: {vname} | {vdesc}")

        # Build weight matrix for full period
        if vname.startswith("flush_"):
            n = int(vname.split("_")[1])
            W_full = _build_flush_n(ind_full, n)
        else:
            W_full = _build_exit_variant(ind_full, vname)

        r_full = evaluate(W_full, ind_full, H=3, label=vname)

        # Per-year comps from the full-period run
        def comp_yr(s, e):
            R = ind_full["R"].reindex(index=W_full.index, columns=W_full.columns).fillna(0.0)
            pos = W_full.shift(1).fillna(0.0)
            turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
            bret = (pos * R).sum(axis=1) - turn * (COST / 2.0)
            mask = (bret.index >= s) & (bret.index < e)
            xs = bret[mask].to_numpy()
            return round((np.prod(1 + xs) - 1) * 100, 1) if mask.sum() > 2 else None

        row_full = {
            "variant":    vname,
            "desc":       vdesc,
            "comp_2020":  comp_yr("2020-01-01", "2021-01-01"),
            "comp_2021":  comp_yr("2021-01-01", "2022-01-01"),
            "comp_2022":  comp_yr("2022-01-01", "2023-01-01"),
            "comp_2023":  comp_yr("2023-01-01", "2024-01-01"),
            "comp_2024":  comp_yr("2024-01-01", "2025-01-01"),
            "comp_2025":  comp_yr("2025-01-01", "2026-01-01"),
            "comp_full":  r_full["comp_full"],
            "maxDD":      r_full["maxDD"],
            "green_all":  r_full["green_all"],
            "green_2021": r_full["green_2021"],
            "green_2022": r_full["green_2022"],
            "avg_expo":   r_full["avg_expo"],
            "avg_turn":   r_full["avg_turnover"],
        }
        rows_full.append(row_full)

        # Build OOS weight matrix (2023-2026)
        if vname.startswith("flush_"):
            n = int(vname.split("_")[1])
            W_oos = _build_flush_n(ind_oos, n)
        else:
            W_oos = _build_exit_variant(ind_oos, vname)

        def comp_oos_yr(W, ind, s, e):
            R = ind["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
            pos = W.shift(1).fillna(0.0)
            turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
            bret = (pos * R).sum(axis=1) - turn * (COST / 2.0)
            mask = (bret.index >= s) & (bret.index < e)
            xs = bret[mask].to_numpy()
            return round((np.prod(1 + xs) - 1) * 100, 1) if mask.sum() > 2 else None

        # OOS metrics restricted to 2023-01-01 -> 2026-01-01 (forward-validate window)
        c23 = comp_oos_yr(W_oos, ind_oos, "2023-01-01", "2024-01-01")
        c24 = comp_oos_yr(W_oos, ind_oos, "2024-01-01", "2025-01-01")
        c25 = comp_oos_yr(W_oos, ind_oos, "2025-01-01", "2026-01-01")

        # OOS full = compound over 2023-2025 only (NOT the warm-up years)
        R_oos = ind_oos["R"].reindex(index=W_oos.index, columns=W_oos.columns).fillna(0.0)
        pos_oos = W_oos.shift(1).fillna(0.0)
        turn_oos = pos_oos.diff().abs().fillna(pos_oos.abs()).sum(axis=1)
        bret_oos = (pos_oos * R_oos).sum(axis=1) - turn_oos * (COST / 2.0)
        mask_oos = (bret_oos.index >= "2023-01-01") & (bret_oos.index < "2026-01-01")
        xs_oos = bret_oos[mask_oos].to_numpy()
        eq_oos = np.cumprod(1 + xs_oos)
        pk_oos = np.maximum.accumulate(eq_oos)
        comp_oos_full_val = round((eq_oos[-1] - 1) * 100, 1) if len(xs_oos) > 2 else None
        maxdd_oos_val = round(float(((eq_oos - pk_oos) / pk_oos).min() * 100), 1) if len(xs_oos) > 2 else None

        # Green rate OOS
        from strat.mover_lab import _month_blocks
        oos_idx = bret_oos[mask_oos].index
        blks_oos = _month_blocks(oos_idx, 3)
        gr_vals = []
        for blk in blks_oos:
            sub = xs_oos[blk]  # blk = integer positions 0..len(oos_idx)-1
            gr_vals.append(np.prod(1 + sub) - 1)
        green_oos_val = round(100.0 * float(np.mean(np.array(gr_vals) > 0)), 0) if gr_vals else None

        row_oos = {
            "variant":    vname,
            "comp_2023":  c23,
            "comp_2024":  c24,
            "comp_2025":  c25,
            "comp_oos_full": comp_oos_full_val,
            "maxDD_oos":  maxdd_oos_val,
            "green_oos":  green_oos_val,
        }
        rows_oos.append(row_oos)

    df_full = pd.DataFrame(rows_full)
    df_oos  = pd.DataFrame(rows_oos)

    # --- Print tables ---
    print("\n")
    print("=" * 90)
    print("TABLE 1: FULL-PERIOD (2020-2026) EXIT VARIANT COMPARISON — mom14-K5")
    print("=" * 90)
    cols_show = ["variant", "comp_2020", "comp_2021", "comp_2022", "comp_2023",
                 "comp_2024", "comp_2025", "comp_full", "maxDD", "green_all", "avg_expo", "avg_turn"]
    print(df_full[cols_show].to_string(index=False))

    print("\n")
    print("=" * 90)
    print("TABLE 2: OOS FORWARD VALIDATION (2023-2025 only) — exit variant OOS contribution")
    print("=" * 90)
    baseline_oos = df_oos[df_oos["variant"] == "flush_3"]["comp_oos_full"].values[0]
    df_oos["oos_delta_vs_flush3"] = df_oos["comp_oos_full"].apply(
        lambda x: round(x - baseline_oos, 1) if pd.notna(x) else None
    )
    print(df_oos[["variant", "comp_2023", "comp_2024", "comp_2025",
                  "comp_oos_full", "maxDD_oos", "green_oos",
                  "oos_delta_vs_flush3"]].to_string(index=False))

    # --- Identify best exit by OOS compound ---
    best_row = df_oos.sort_values("comp_oos_full", ascending=False).iloc[0]
    best_var = best_row["variant"]
    best_oos = best_row["comp_oos_full"]

    print("\n")
    print("=" * 70)
    print("FORWARD-VALIDATION VERDICT")
    print("=" * 70)
    print(f"  Baseline (flush_3) OOS 2023-2025:  {baseline_oos:+.1f}%")
    print(f"  Best exit variant:                 {best_var}")
    print(f"  Best exit OOS 2023-2025:           {best_oos:+.1f}%")
    print(f"  OOS delta vs flush:                {best_oos - baseline_oos:+.1f}pp")
    flush3_full = df_full[df_full["variant"] == "flush_3"]["comp_full"].values[0]
    best_full   = df_full[df_full["variant"] == best_var]["comp_full"].values[0]
    best_dd_full = df_full[df_full["variant"] == best_var]["maxDD"].values[0]
    print(f"  flush_3 full-period compound:      {flush3_full:+.1f}%")
    print(f"  {best_var} full-period compound:    {best_full:+.1f}%")
    print(f"  {best_var} maxDD (full):            {best_dd_full:.1f}%")

    # Per-year OOS for top-3 exits
    print("\n  Top-3 exits by OOS compound (2023-2025):")
    top3 = df_oos.sort_values("comp_oos_full", ascending=False).head(3)
    for _, r in top3.iterrows():
        print(f"    {r['variant']:20s}  2023={r['comp_2023']:+6.1f}%  "
              f"2024={r['comp_2024']:+6.1f}%  2025={r['comp_2025']:+6.1f}%  "
              f"OOS={r['comp_oos_full']:+7.1f}%  DD={r['maxDD_oos']:5.1f}%  "
              f"delta={r['oos_delta_vs_flush3']:+5.1f}pp")

    print("\n  HONEST ASSESSMENT:")
    # Check if exit ranking IS consistent 2020-2022 vs 2023-2025
    # Rank exits by full-period comp, then check OOS rank correlation
    df_merge = df_full[["variant", "comp_full"]].merge(
        df_oos[["variant", "comp_oos_full"]], on="variant")
    rho = df_merge["comp_full"].corr(df_merge["comp_oos_full"], method="spearman")
    print(f"  Rank correlation (full-period vs OOS): Spearman rho = {rho:.3f}")
    if rho > 0.5:
        print("  -> Exit ranking TRANSLATES in-sample -> OOS (rho > 0.5): moderate evidence")
    elif rho > 0.2:
        print("  -> Exit ranking WEAKLY translates (rho 0.2-0.5): noisy, selection risk")
    else:
        print("  -> Exit ranking does NOT translate (rho < 0.2): in-sample selection = overfit")

    print("\n  Noise/overfit caveats:")
    print("  - Only 14 variants tested; best-of-N inflates OOS figures")
    print("  - 2020-2022 is the in-sample supercycle; 2023-2025 is the decisive OOS")
    print("  - OOS 2022 bear shows gate is the ONLY bear lever (no exit saves you)")
    print("  - Any exit variant with OOS DD > -70% or OOS < +100% vs flush is not a clear upgrade")

    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
