# Project state / opportunity / risk map (2026-06-18)

Read-only 6-lane research sweep (workflow `wgyldyi6s`) across crypto + games + harness + archive + docs + integrity,
run alongside the MA strat-builder. Synthesis below; per-lane detail in the workflow output.

## Recurring themes (cross-lane)

1. **Most pulled data is going to waste — confirmed quantitatively.** The chimera carries ~181 features / 22 families,
   but deployed strategies use only `fund_*` (funding satellite) + price/regime (daily_engine). **ETF, stablecoin, DVOL,
   transfer-entropy, basis/premium, LOB, book-depth — all UNUSED by any deployed strategy.** (The user's "almost all go
   to waste" is literally true.)
2. **Live data-integrity + staleness issues.** All external sources 18–21d stale (last ~2026-05-28..31); `basis/premium`
   49d stale; **77/104 1d chimera files still carry the `xex_` name-collision** (registry fixed 2026-06-18 but parquets
   NOT rebuilt — 3 ghost `_right` cols + 47–64% null on the primary xex spreads); `lob_` 94% null. The pre-commit CDAP
   hook is **not installed** (staleness/regressions not auto-caught).
3. **The genuine open frontier = EXTERNAL event data — and there's pre-reset evidence + a half-built probe.** Multiple
   lanes converge: (a) Coinbase/Upbit listing-announcement signal (the stated open frontier; `coinbase_effect_probe.py`
   exists from the crashed Wave-7 but has a **CRITICAL API bug** — all 77 Coinbase↔Binance products return NOT_FOUND);
   (b) **archive P8: Binance/Bybit/OKX new-FUTURES-listing h1-momentum** — pre-reset OOS evidence (t=3.61, n=85,
   +8.64%/event, Bonferroni-surviving), data ON DISK (`multi_venue_listings.parquet`, 1,482 events, 3 venues), NOT on the
   dead-list, never re-tested post-reset. This is the strongest concrete untested avenue.
4. **Canonical docs are stale (monotonic-memory hazard).** `05_OPEN_THREADS.md` is 8+ days stale (resolved June-17
   threads still listed HIGH-priority open). **Dead-list count cited as "D01–D63" in README/STATE/CLAUDE.md but the list
   runs to D76** — D64–D76 are invisible to a new instance (it could re-mine a re-killed vein).

## Opportunities (EV-ranked)

| EV | Item | Why |
|----|------|-----|
| HIGH | **Test the UNUSED chimera families as regime conditioners on `daily_engine`** — ETF-flow, basis/premium, transfer-entropy. Fully on disk, never tested, ~1h each. | Directly attacks "data going to waste"; could improve the deployed book's UNSEEN DD protection. |
| HIGH | **External event-data frontier:** fix `coinbase_effect_probe.py`'s API bug + run it; AND re-test the **P8 multi-exchange futures-listing h1-momentum** under the post-reset apparatus (strong pre-reset evidence, data on disk). | The only genuinely-open avenues after internal-data exhaustion; P8 has real prior evidence + is event-driven. |
| HIGH | **Rebuild the 77 `xex_`-collision 1d chimera files** with the fixed registry (zero new data). | Recovers the only cross-venue microstructure signal; the collision silently poisons any xex_-reading strategy. |
| MED | Refresh stale data (external 21d, basis 49d) before any deploy or new conditioning run. | Stale conditioning data invalidates recent/forward runs. |
| MED | Wire `lob_bgf_*` (77 assets, on disk) into the bar-grain attach (registry addition, not new data). | Intraday LOB microstructure currently ghost-null in chimera. |
| LOW | WM: run the V1.1 upgrade probes (SAM/PCGrad/MTP) — but deltas all INFERRED (0% verified) + V1 checkpoints incompatible (need retrain). Compute-bound. | Lower priority; the WM regime-gate already lost to the cheap SMA on UNSEEN. |

## Risks / gaps (high severity)

1. **`daily_engine --today` runs on 21-day-stale data** (last bar 2026-05-28) — the deployable book's live-readiness gap.
   Refresh before any deploy.
2. **Pre-commit CDAP hook not installed** — commits bypass all gates + the strat regression checks.
3. **`xex_` collision live in 77/104 chimera files.**
4. **Doc staleness** — `05_OPEN_THREADS` 8d stale; dead-list count D63-vs-actual-D76 → misleads new instances.
5. **WM regime-gate UNSEEN reversal** (known) — do NOT wire WM as a regime gate; cheap SMA dominates held-out.

## Cheapest safe correct-as-you-go fixes (offered)
- Fix the dead-list count refs (D63→D76) across README/STATE/CLAUDE.md + refresh `05_OPEN_THREADS` with the June-17
  outcomes (pure knowledge hygiene, zero-risk).
- Install the pre-commit CDAP hook.
These are knowledge/infra hygiene; the data-refresh + xex_ rebuild are heavier (pipeline runs); the external-data + unused-
feature tests are the real research EV.
