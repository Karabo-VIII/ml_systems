# Per-Cadence × Bar-Type CEILING / COVERAGE Map (meta2)

Built 2026-06-06 by the sol-synthesis worker. **RWYB**: every number below was recomputed from a
command against real data (the two capturable catalogs + `oracle_holdtime_prefilter.json` + on-disk
chimera file counts), not quoted from a rig headline. LONG-ONLY, taker cost 0.0024 round-trip.

## TL;DR (for meta2 routing)
1. **Capturable-move SUPPLY is NOT the bottleneck.** Every cadence has 24–49 % of bars *inside* a
   net-positive oracle up-leg, and ~33 % of 4h bars / ~30 % of daily bars have a ≥5 % move ahead. The
   "abundant opportunity" premise holds at the ceiling. **No cell is dry of *moves*.**
2. **The dry axis is COVERAGE (strategy-grade chimera features), not supply.** `runs_volume` and
   `adaptive_vol` have **0** chimera builds; `range`/`dib` exist for **3 assets only** (BTC/ETH/PEPE);
   `30m` is partial (77). Only **15m/1h/4h/1d/dollar** are fully built on ~u100 (104 files).
3. **On the fully-covered clock cadences the MA *trigger* is refuted** (4h+1d, two-sided-sound, 0/77 beat
   the regime-matched firewall). Supply is there; an MA can't time it (median oracle hold ≤1–2 bars vs the
   ~4.5-bar MA lag). Those cells need a **non-MA trigger**, not a different cadence.
4. **Highest-EV *under-explored* cell = `adaptive_vol`**: best oracle MA-timeability (median hold 10.5 bars
   ≫ 4.5-bar lag = "YES") AND high supply (41 % time-in-move) — but **zero chimera coverage**. A chimera
   build is the single unlock. `dib` is 2nd (MARGINAL timeability) but only BTC/ETH/PEPE.
5. **`dollar` is granularity-degenerate** for high-capture (4.1M bars → DP excluded by design); don't chase
   it for capture-ceiling work.

## The map

| cadence | SUPPLY: time-in-move %¹ | SUPPLY: win@5% (catalog)² | trades/1k-bar¹ | median oracle hold (bars)¹ | MA-timeable?³ | chimera coverage (u100)⁴ | verdict |
|---|--:|--:|--:|--:|:--|:--|:--|
| **15m**         | 49.0 | — (no catalog) | 98  | 5.0  | MARGINAL | ✅ FULL (104) | high supply, MA laggy → non-MA trigger |
| **30m**         | 43.8 | — | 146 | 3.0  | NO | ⚠️ PARTIAL (77) | mineable; MA dead |
| **1h**          | 39.2 | — | 196 | 2.0  | NO | ✅ FULL (104) | mineable; MA dead |
| **4h**          | 23.8 | **32.9 %** (UNSEEN 29.6) | 238 | 1.0 | NO | ✅ FULL (104) | ceiling proven; **MA REFUTED 0/77** |
| **1d / 3d**     | 26.2 | **29.9 %** (UNSEEN 24.8) | 262 | 1.0 | NO | ✅ FULL (104) | ceiling proven; **MA REFUTED** |
| **dollar**      | n/a (DP-excluded) | — | n/a | n/a | n/a | ✅ FULL (104) | **degenerate** for high-capture |
| **range**       | 34.4 | — | 86  | 4.0  | NO | ❌ DRY (3; loader schema-broken) | build + fix loader prereq |
| **dib**         | 30.1 | — | 50  | 6.0  | MARGINAL | ❌ DRY (3: BTC/ETH/PEPE) | only timeable+built cell, but no SOL/u100 |
| **runs_volume** | 33.3 | — | 333 | 1.0  | NO | ❌ DRY (0 built) | not built |
| **adaptive_vol**| 40.5 | — | 39  | 10.5 | **YES** | ❌ DRY (0 built) | **best MA-timeability, zero coverage → BUILD = unlock** |

¹ Oracle perfect-foresight high-capture DP (`runs/research/oracle_ceiling_builder.py`), **u20, raw bars**,
net of 0.24 % cost, 1h≤hold≤7d. `time-in-move %` = n_trades×median_hold / n_bars = fraction of bars inside
a captured up-leg. `dollar` excluded (4.1M bars infeasible for O(n·H) DP).
² `win@5%` = fraction of bars with a realized ≥5 % forward move (oracle SUPPLY base rate), from the
pre-computed catalogs (**87 assets**): 4h = `win_5pct_18bar` (≥5 % within 72h); 1d = `win_5pct` (≥5 % within
3d). Only 4h + daily have a per-split win-rate catalog; 15m/30m/1h/dollar/alts have oracle-density only.
³ vs the ~4.5-bar MA decision lag (from `rank_by_hold_bars`). YES=hold≫lag, MARGINAL=hold≈lag, NO=too sharp.
⁴ On-disk feature-enriched chimera parquet count under `data/processed/chimera/<cad>/`. ~u100 ≈ 104 files.

## Supply ceiling, full catalog detail (pooled / UNSEEN, 87 assets)

| catalog | win@3% | win@5% | win@7% | win@10% | median fwd max-gain |
|---|--:|--:|--:|--:|--:|
| **4h** (12-bar/48h gain, 18-bar/72h for ≥5%) — pooled | 41.0 % | 32.9 % | — | — | 3.7 % (18-bar) |
| **4h** — UNSEEN                                        | 38.8 % | 29.6 % | — | — | 3.1 % |
| **1d/3d** — pooled                                     | 40.2 % | 29.9 % | 22.2 % | 14.2 % | 2.3 % (3d) |
| **1d/3d** — UNSEEN                                     | — | 24.8 % | — | 11.0 % | 1.4 % |

→ The **median** 4h bar's best 72h exit is ~3.7 % gross (~3.4 % net) — squarely inside the user's
2–5 %/move target band. **The target band is reachable at the supply ceiling; the gap is realization
(timing), not opportunity.** (Realized: the MA family captured 0/77 vs the firewall — supply ≠ skill.)

## Concentration vs dry — the honest read
- **Where capturable moves concentrate:** *everywhere by supply* (no cell <24 % time-in-move). Highest raw
  oracle trade-density at 1d/runs_volume/4h; highest *time-in-move* at 15m/adaptive_vol/30m.
- **Where it's genuinely dry:** **coverage**, not moves — `runs_volume`/`adaptive_vol` (0 features),
  `range` (3 + schema-broken), `dib` (3), `30m` (partial). `dollar` is supply-degenerate.
- **The one cell that pairs good supply + good MA-timeability + is unbuilt = `adaptive_vol`** → if any
  bar-type deserves a chimera build, it is that one (then `dib` for SOL/u100). For the already-built clock
  cadences, the lever is the **trigger** (liquidation-cascade / momentum-accel per RESEARCHER_REPORT_2),
  not the cadence.

## Caveats (do not over-read)
- `win@X%` catalog rates are **oracle SUPPLY base rates** (any bar with a forward move), an upper bound a
  perfect timer could see — **NOT realizable** by a live signal. Realized capture from the refuted MA work
  was 0/77 beating the regime-matched null.
- Oracle-density (time-in-move, hold) = **u20**, not u100. Catalogs = **87** assets. Coverage counts = file
  presence, not feature-validity (`range`'s 3 files are loader-broken). Apples-to-apples per-split win-rates
  exist **only** for 4h + daily; the faster clock cadences and alts have density-only ceilings.
- `MA-timeable` is an **MA-specific** lens (4.5-bar lag); a window-free trigger (e.g. `norm_momentum_accel`)
  is not bound by it, so "NO" cells are not closed to *all* triggers — only to the MA family.
