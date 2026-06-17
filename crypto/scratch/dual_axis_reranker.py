"""
Dual-axis strategy re-ranker: SHARPE-axis + WEALTH-axis.

Takes documented metrics for every tested strategy and scores each under:
  A) SHARPE AXIS (institutional, leverage-capable, path-smoothness)
     Rank by: Sharpe. Target: smooth equity curve, lever-optimal.

  B) WEALTH AXIS (growth mandate, unlevered, terminal-wealth-max)
     Rank by: CAGR_at_DD_budget, where each strategy is position-size-scaled
     to fit a target portfolio-level max DD budget (user's choice).
     Effective CAGR = raw_CAGR * min(1, dd_budget / raw_dd).

Assumption: linear sizing (scale all positions proportionally). This is
the standard approximation; in practice non-linearities exist (stop-loss
interactions, correlation shifts at size) but for ranking purposes the
approximation is robust.

User's premise (2026-04-22): "DD is less tolerable the more it amplifies,
but I can accept any level with proper sizing + stops".
So both axes matter — pick the right one per deployment context.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# =======================================================================
# ALL DOCUMENTED STRATEGIES (from session memory + scratch outputs)
# All 18mo combined WF unless noted. Ann returns computed assuming 540-day.
# =======================================================================

# Fields: name, total_pct, sharpe, max_dd_pct, period_days, class, notes
STRATEGIES = [
    # -- Tier 1: xsec rankers (WF-validated) --
    ('xsec_K5+5_FULL_dneut',    314, 3.70,  -8, 540, 'XSEC_RANKER',
     'Champion so far. 3/3 WF windows positive (Sh 3.13-4.02).'),
    ('xgb_K5+5_dneut',          227, 3.36, -20, 540, 'XSEC_RANKER',
     '3/3 WF positive. DD estimated from Sharpe/vol.'),
    ('xgb_K3+3_dneut',          269, 2.89, -20, 540, 'XSEC_RANKER',
     '3/3 WF positive. Highest Sh among pure XGB.'),
    ('xgb_K3_long',             545, 1.92, -40, 540, 'XSEC_RANKER',
     '3/3 WF positive. Long-only.'),
    ('xgb_K1_long',             630, 1.57, -60, 540, 'XSEC_RANKER',
     'FAILED WF1 (-19%). Combined still strong.'),
    ('cat_K1_stop_no_macro',    778, 1.69, -78, 540, 'XSEC_RANKER',
     '3/3 WF positive. Aggressive.'),
    ('cat_K1_FULL_long',        537, 1.53, -59, 540, 'XSEC_STACKED',
     'FAILED WF1 (-29%). K=1 concentration fragile.'),
    ('cat_K5_FULL_long',        533, 2.05, -53, 540, 'XSEC_STACKED',
     '3/3 WF positive. Long-only with FULL stack.'),

    # -- MoE gate candidates --
    ('moe_v1_4champs_argmax',   382, 2.73, -38, 240, 'MOE_GATE',
     'CONCEDED on Sharpe, NOT YET WF-validated. 10mo test only.'),
    ('moe_v2i_3champs_argmax',  177, 4.55, -11, 240, 'MOE_GATE',
     '3 d-neut champs. Tested on 8mo subset.'),

    # -- Paper_trader profiles (post-fix J1 run, 13mo) --
    ('prod_meta_combined',       94, 3.80,  -4, 396, 'PAPER_TRADER',
     'Post-fix +94% on cost_val_maker seed. Real fills calibrated.'),
    ('prod_meta_combined_k15',   95, 3.84,  -4, 396, 'PAPER_TRADER',
     'Kelly 1.5x variant. Marginal beat over k10.'),
    ('prod_meta_full_3eng',      77, 2.96,  -6, 396, 'PAPER_TRADER',
     '3-engine variant — vpin_flow DILUTES the 2-engine baseline.'),
    ('regime_routed_full',       24, 0.88, -21, 396, 'PAPER_TRADER',
     'Downgraded post-fix from Sh 3.82 to 0.88.'),
    ('prod_floor_combined',      19, 0.68, -22, 396, 'PAPER_TRADER',
     'Downgraded post-fix.'),
    ('perp_dna_long_short',      14, 0.65, -16, 396, 'PAPER_TRADER',
     'Downgraded from Sh 4.15. Bear claim unsubstantiated.'),
    ('prod_meta_combined_v8',     9, 0.42, -26, 396, 'PAPER_TRADER',
     'CatBoost v8 meta. Was claimed +738% pre-fix. 82x de-rating.'),
    ('prod_combined_no_meta',     6, 0.33, -24, 396, 'PAPER_TRADER',
     'Non-meta baseline. Reference point.'),
]


def compute_axes(strategies, dd_budget_pct: float = 20.0):
    """Compute both axis scores for each strategy."""
    results = []
    for name, total, sharpe, dd, days, cls, notes in strategies:
        # Annualized CAGR
        days_per_year = 365
        eq_multiple = 1 + total / 100
        if eq_multiple > 0:
            cagr = (eq_multiple ** (days_per_year / days) - 1) * 100
        else:
            cagr = -100
        # Calmar (CAGR / |DD|)
        calmar = abs(cagr / dd) if dd != 0 else 0
        # Position-size factor to fit DD budget
        size_factor = min(1.0, dd_budget_pct / abs(dd)) if dd != 0 else 1.0
        # Effective CAGR at DD budget (linear sizing approx)
        scaled_cagr = cagr * size_factor
        # Effective DD at chosen sizing
        scaled_dd = dd * size_factor
        results.append({
            'name': name, 'class': cls, 'period_days': days, 'notes': notes,
            'total_pct': total, 'sharpe': sharpe, 'raw_dd': dd,
            'cagr': round(cagr, 1), 'calmar': round(calmar, 2),
            'size_factor_at_dd_budget': round(size_factor, 3),
            'scaled_cagr_at_dd_budget': round(scaled_cagr, 1),
            'scaled_dd_at_budget': round(scaled_dd, 1),
            # Heuristic Kelly fraction cap (kelly = Sharpe^2 / 2 for geometric, capped)
            'kelly_f_star_est': round(min(1.0, sharpe ** 2 / 4), 2),
        })
    df = pd.DataFrame(results)
    return df


def print_ranking(df, axis, top_n=15):
    sort_col = {'SHARPE': 'sharpe', 'WEALTH_20': 'scaled_cagr_at_dd_budget',
                'WEALTH_40': 'scaled_cagr_at_dd_budget_40',
                'WEALTH_unconstrained': 'cagr'}[axis]
    if axis == 'WEALTH_40':
        sort_col = 'scaled_cagr_at_dd_budget_40'
    df_sorted = df.sort_values(sort_col, ascending=False).head(top_n)
    print(f'\n{"=" * 95}')
    if axis == 'SHARPE':
        print(f'AXIS A: SHARPE RANKING (institutional sleeve — smooth path, leverage-capable)')
        cols = ['name', 'class', 'sharpe', 'cagr', 'raw_dd', 'calmar', 'notes']
    elif axis == 'WEALTH_20':
        print(f'AXIS B: WEALTH RANKING @ DD budget 20% (unlevered, portfolio-stop guardrail)')
        cols = ['name', 'class', 'scaled_cagr_at_dd_budget', 'cagr', 'raw_dd',
                'size_factor_at_dd_budget', 'sharpe', 'notes']
    elif axis == 'WEALTH_40':
        print(f'AXIS B2: WEALTH RANKING @ DD budget 40% (aggressive growth, hard portfolio stop)')
        cols = ['name', 'class', 'scaled_cagr_at_dd_budget_40', 'cagr', 'raw_dd',
                'size_factor_at_dd_budget_40', 'sharpe', 'notes']
    elif axis == 'WEALTH_unconstrained':
        print(f'AXIS B3: WEALTH RANKING UNCONSTRAINED (raw CAGR at 1x sizing — NO DD limit)')
        cols = ['name', 'class', 'cagr', 'raw_dd', 'sharpe', 'calmar', 'notes']
    print('=' * 95)
    for _, r in df_sorted.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (int, float)) and c != 'period_days':
                vals.append(f'{v:+7.1f}' if abs(v) >= 10 or isinstance(v, int) else f'{v:+6.2f}')
            else:
                vals.append(str(v)[:60])
        print(f"  {' | '.join(vals)}")


if __name__ == '__main__':
    df = compute_axes(STRATEGIES, dd_budget_pct=20.0)

    # Also compute WEALTH_40
    df['size_factor_at_dd_budget_40'] = df['raw_dd'].abs().apply(
        lambda dd: round(min(1.0, 40 / dd), 3) if dd != 0 else 1.0)
    df['scaled_cagr_at_dd_budget_40'] = round(
        df['cagr'] * df['size_factor_at_dd_budget_40'], 1)

    # SHARPE axis
    print_ranking(df, 'SHARPE', top_n=18)
    # WEALTH axis @ DD budget 20% (conservative)
    print_ranking(df, 'WEALTH_20', top_n=18)
    # WEALTH axis @ DD budget 40% (aggressive)
    print_ranking(df, 'WEALTH_40', top_n=18)
    # WEALTH unconstrained (pure CAGR race)
    print_ranking(df, 'WEALTH_unconstrained', top_n=18)

    # Recommendation
    print(f'\n{"=" * 95}')
    print('RECOMMENDATION (pick the right axis per your deployment context)')
    print('=' * 95)

    # Find champions
    top_sharpe = df.sort_values('sharpe', ascending=False).iloc[0]
    top_wealth_20 = df.sort_values('scaled_cagr_at_dd_budget', ascending=False).iloc[0]
    top_wealth_40 = df.sort_values('scaled_cagr_at_dd_budget_40', ascending=False).iloc[0]
    top_unconstrained = df.sort_values('cagr', ascending=False).iloc[0]

    print(f'\nSHARPE champion: {top_sharpe["name"]} (Sh {top_sharpe["sharpe"]}, CAGR {top_sharpe["cagr"]}%/yr)')
    print(f'  Rationale: smooth curve, lever-optimal. Pick this for external-partner / pro-desk use.')

    print(f'\nWEALTH_20 champion: {top_wealth_20["name"]} '
          f'(scaled CAGR {top_wealth_20["scaled_cagr_at_dd_budget"]}%/yr @ size {top_wealth_20["size_factor_at_dd_budget"]:.2f}x)')
    print(f'  Rationale: max CAGR under -20% portfolio DD stop. Conservative growth.')

    print(f'\nWEALTH_40 champion: {top_wealth_40["name"]} '
          f'(scaled CAGR {top_wealth_40["scaled_cagr_at_dd_budget_40"]}%/yr @ size {top_wealth_40["size_factor_at_dd_budget_40"]:.2f}x)')
    print(f'  Rationale: max CAGR under -40% portfolio DD stop. Aggressive growth.')

    print(f'\nUNCONSTRAINED champion: {top_unconstrained["name"]} '
          f'(raw CAGR {top_unconstrained["cagr"]}%/yr @ 1x sizing, raw DD {top_unconstrained["raw_dd"]}%)')
    print(f'  Rationale: pure CAGR max. Only deploy with hard portfolio-level kill switch.')

    out = ROOT / 'logs' / 'dual_axis_ranking.csv'
    df.to_csv(out, index=False)
    print(f'\n[ranking] written to {out}')
