# 3-Layer Problem-Solving Engine — Adversarial Audit (2026-06-10)

> Triggered by the user asking to "confirm each layer is correct, audited, and has no gaps." Four RED-team auditors
> (presumed-broken-until-proven, every claim RWYB-verified against the actual code, not build reports). Honest verdict:
> **it was NOT gap-free** — 2 CRITICAL + several HIGH found and fixed; the scope was also an overclaim, now corrected.

## The honest scope (audit-confirmed)

This is **not three engines.** It is **routing + plumbing over one real generic engine, one adapter-stub, and one
hardened-but-modest model cohort:**

| Layer | What it actually is | Verdict |
|---|---|---|
| **A — Games** | A generic `GameAdapter` + UCT **proven only on TicTacToe** (W37 D3 L0). The real chess engine pre-existed and does **not** use `GameAdapter`; **no `ChessGameAdapter` exists.** | Generic adapter **works**; "chess plugs in" was an **OVERCLAIM** → corrected in `solve.py`. |
| **B — General WM** | An **adapter (format converter) + a routing label.** No model, no trainer, no training loop. | **STUB.** "General WM" was an **OVERCLAIM** → honestly labeled. |
| **C — Crypto WM** | Pre-existing V0–V25, **hardened + scaffolded** the prior session. | **The solid part** — all 4 components HELD under adversarial probing (no look-ahead, no behavior change, no critical). |
| **Router / bus / solve** | The plumbing tying them together. | **Works as a router/planner**; had 2 CRITICAL + several HIGH (now fixed). |

## Findings + status

### Fixed (committed)
| # | Sev | Finding | Fix | Commit |
|---|-----|---------|-----|--------|
| 1 | CRITICAL | Router misrouted crypto/time-series/science **decision** problems (with `has_exact_simulator`) to Layer-A/AlphaZero | domain-gated the simulator-implies-games branch | `bca6e2d` |
| 2 | CRITICAL | `split_four_way` silently produced **empty val/OOS/unseen** on small data (no error) | raises a clear min-bars error; crypto-size unaffected | `bca6e2d` |
| 3 | HIGH | `solve.py` "IMPORTABLE" misleading for the `GameAdapter` ABC | `IMPORTABLE_ABSTRACT (needs concrete subclass)` | `bca6e2d` |
| 4 | HIGH | structured-prediction NOT-IMPLEMENTED bypassed the Python `warnings` channel | routed through `_warn` | `bca6e2d` |
| 5 | HIGH | `GeneralAdapter` emitted 6-digit timestamps (violates 13-digit-ms invariant; breaks dated splits) | valid base-epoch ms + range-check warn | `bca6e2d` |
| 6 | HIGH | Overclaim language ("chess plugs in", Layer-B "anti-fragile loop") | corrected to honest text | `bca6e2d` |
| 7 | MED | `GeneralAdapter` silently zeroed NaN features | warns (count + columns) | `bca6e2d` |
| 8 | MED | `MultiAssetDataset` silently dropped duplicate `asset_idx` | warns on duplicate | `bca6e2d` |
| 9 | MED | Cross-pollination bus **concurrent-write data loss** (fail-open wrote UNLOCKED; Windows PermissionError defeated stale-detection) | block-or-raise lock + real PID-liveness reclaim; strict 50/50 + 20-way test | `(bus commit)` |

### Held under adversarial probing (no fix needed — the genuinely solid core)
- **Layer-C WM hardening (all 4):** the OOM numpy-index fix (behavior-preserving, verified across stride/weights/short-seg edge cases); `regime_targets.py` (no-look-ahead independently re-proven — perturbing future-beyond-horizon and past bars leaves the label invariant); `forward_regime_head.py` (truly OFF-by-default, aux loss correctly masked); `multi_asset_dataset.py` (causal asof-backward, mask zeroes absent assets, no leak). **No look-ahead anywhere.**
- **Cross-pollination bus:** persists cross-process; the 3 seeded lessons are **accurate** to CLAUDE.md/MEMORY.md; `read_for_layer` filters correctly; dedup is idempotent.
- **Layer-A generic UCT:** the TicTacToe proof passes now (W37 D3 L0); the routing import resolves.
- **CryptoAdapter.feature_families():** works at call-time (the prior build report's claimed failure did not reproduce).

### Deferred (honest open gaps — low severity / rare path / by-design)
- **MED** Router never reads `data_budget` → the EfficientZero-V2 sub-branch is unreachable (always MuZero for no-sim). Completeness gap.
- **MED** `structured_prediction + domain=games` silently discards the domain (no mismatch warning).
- **LOW** `forward_regime_aux_loss` returns a non-grad leaf when ALL labels are NaN (safe in the intended `base + w*aux` wiring; unsafe only if used standalone).
- **LOW** `MultiAssetDataset` absent-asset slots are zeros, not a NaN sentinel — a trainer that ignores the mask trains toward 0.0 (contract is documented, not mechanically enforced).
- **LOW** `GeneralAdapter` is single-instrument by default (multi-instrument needs a subclass).
- **NOTE** `scripts/autonomy/git_commit_safe.py` has the **same** lock anti-pattern as the bus bug (FileExistsError-only + age-only stale + fail-open). `git`'s own `index.lock` backstops it, but it's worth the same fix.

## Bottom line
The plumbing now works correctly as a **router/planner** (criticals fixed, overclaims corrected, bus race truly closed),
and the **Layer-C WM hardening is clean**. But "three correct, gap-free engines" was never the reality: Layer B is a stub,
Layer A's chess is not wired, and the value is the routing + the (modest, untrained) crypto cohort + the validation spine.
The honest deliverable is an **honest router that states its own ceiling**, not three solved engines.
