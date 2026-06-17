"""reports -- generate REPORT.md from audit results.

__contract__:
  inputs: audit_result dict from walk_forward.n_seed_audit + optional baseline
  outputs: markdown string + optional file write
"""
from __future__ import annotations

__contract__ = {
    "kind": "report_generator",
    "owner": "wealth_bot/framework/reports",
    "purpose": "Markdown report from N-seed audit results",
}

from pathlib import Path


def render_segment_table(summary: dict, segments: list[str] = None) -> str:
    """Render per-segment summary as markdown table."""
    if segments is None:
        segments = list(summary.keys())
    lines = []
    lines.append("| Segment | n_seeds_pos | median % | mean % | std pp | p05 % | p95 % | min % | max % | med trades | med DD% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seg in segments:
        s = summary[seg]
        lines.append(
            f"| **{seg}** | {s['positive_seeds']}/{int(round(s['positive_seeds'] / (s['positive_seeds_pct']/100.0) if s['positive_seeds_pct']>0 else 10))} "
            f"| {s['compound_median']:+.1f} | {s['compound_mean']:+.1f} | {s['compound_std']:.1f} "
            f"| {s['compound_p05']:+.1f} | {s['compound_p95']:+.1f} "
            f"| {s['compound_min']:+.1f} | {s['compound_max']:+.1f} "
            f"| {s['mean_trades']:.0f} | {s['median_max_dd']:.1f} |"
        )
    return "\n".join(lines)


def render_ablation_table(ablation: dict) -> str:
    """Render configuration ablation comparison table."""
    lines = []
    lines.append("| Config | UNSEEN median % | UNSEEN p05 % | UNSEEN p95 % | seeds positive | std pp |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, audit in ablation.items():
        s = audit["summary"]["UNSEEN"]
        lines.append(
            f"| **{name}** | {s['compound_median']:+.1f} | {s['compound_p05']:+.1f} | {s['compound_p95']:+.1f} "
            f"| {s['positive_seeds']}/{audit['n_seeds']} | {s['compound_std']:.1f} |"
        )
    return "\n".join(lines)


def generate_report(
    cfg_summary: dict,
    audit_baseline: dict | None = None,
    audit_ensemble: dict | None = None,
    audit_threshold: dict | None = None,
    audit_full: dict | None = None,
    static_baseline_unseen_pct: float | None = None,
    title: str = "PEPE x EMA Wealth Bot — N-Seed Audit Report",
    extra_sections: list[tuple[str, str]] | None = None,
) -> str:
    """Generate the full REPORT.md content."""
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for k, v in cfg_summary.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    win_threshold = "UNSEEN compound (10-seed median) > +50% AND p05 > +5% AND all 10 seeds positive"
    lines.append(f"## Win threshold")
    lines.append("")
    lines.append(f"> {win_threshold}")
    lines.append("")

    if static_baseline_unseen_pct is not None:
        lines.append(f"**Static baseline UNSEEN compound** (4h EMA 7/15 + whale>0): **+{static_baseline_unseen_pct:.1f}%** (19 trades).")
        lines.append("")

    # Ablation
    ablation = {}
    if audit_baseline:
        ablation["baseline (LGBM picker, per-seed)"] = audit_baseline
    if audit_ensemble:
        ablation["+ U1 ensemble"] = audit_ensemble
    if audit_threshold:
        ablation["+ U2 threshold"] = audit_threshold
    if audit_full:
        ablation["+ U1 + U2 (combined)"] = audit_full

    if ablation:
        lines.append("## Ablation: architecture upgrades")
        lines.append("")
        lines.append(render_ablation_table(ablation))
        lines.append("")

    # Detail per audit
    for name, audit in ablation.items():
        if audit is None:
            continue
        lines.append(f"### Detail: {name}")
        lines.append("")
        lines.append(render_segment_table(audit["summary"]))
        lines.append("")
        if audit.get("ensemble"):
            lines.append(f"**Ensemble pathway** (mean of {audit['n_seeds']} seeds, single deploy candidate):")
            ens = audit["ensemble"]
            lines.append("")
            lines.append("| Segment | compound % | n_trades | WR | max DD % | sharpe | thr |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for seg in ["TRAIN", "VAL", "OOS", "UNSEEN"]:
                if seg in ens:
                    r = ens[seg]
                    lines.append(
                        f"| **{seg}** | {r['compound_pct']:+.1f} | {r['n_trades']} | {r['win_rate']:.2f} "
                        f"| {r['max_dd_pct']:.1f} | {r['sharpe']:.2f} "
                        f"| {ens.get('best_threshold', 0.0):+.4f} |"
                    )
            lines.append("")

    # Verdict
    if audit_full and static_baseline_unseen_pct is not None:
        s = audit_full["summary"]["UNSEEN"]
        win_met = (
            s["compound_median"] > 50.0
            and s["compound_p05"] > 5.0
            and s["positive_seeds"] >= audit_full["n_seeds"]
        )
        refute_met = (
            s["compound_median"] < 35.0 or s["compound_p05"] < -25.0
        )
        lines.append("## Verdict")
        lines.append("")
        if win_met:
            lines.append("**WIN MET** ✓ Bot beats static baseline under 10-seed median + p05 > +5% + all seeds positive.")
        elif refute_met:
            lines.append("**REFUTED** — Bot fails win threshold AND triggers refutation criteria. Static rule remains primary.")
        else:
            lines.append("**INCONCLUSIVE** — Bot does not clearly beat static under all gates.")
        lines.append("")
        lines.append(f"- UNSEEN median: {s['compound_median']:+.1f}% (gate >+50%)")
        lines.append(f"- UNSEEN p05:    {s['compound_p05']:+.1f}% (gate >+5%)")
        lines.append(f"- UNSEEN min:    {s['compound_min']:+.1f}% (gate: all >+0%)")
        lines.append(f"- Static baseline: +{static_baseline_unseen_pct:.1f}% (single-window deterministic)")
        lines.append("")

    if extra_sections:
        for header, body in extra_sections:
            lines.append(f"## {header}")
            lines.append("")
            lines.append(body)
            lines.append("")

    return "\n".join(lines)


def save_report(content: str, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
