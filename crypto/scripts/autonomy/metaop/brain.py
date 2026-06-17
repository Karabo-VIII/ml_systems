"""Operator BRAIN -- crypto-consumer SHIM over the canonical harness.metaop.brain (G-A dedup 2026-06-07).

The pluggable intelligence (decide/act/work + every backend: Mock / AgentSdk / Anthropic / Cli / PersistentCli /
Ollama) now lives ONCE in harness/metaop/brain.py. That engine is project-AGNOSTIC: its system prompts carry a
{domain} slot. This shim:

  1. RE-EXPORTS every brain class + find_claude (so `from metaop.brain import PersistentCliBrain` etc. still work).
  2. Injects the crypto DOMAIN -- make_brain(kind) here calls the harness make_brain(kind, domain=CRYPTO_DOMAIN), so
     the loop's decide/act/work prompts carry the crypto-quant framing the live loop has always used. Crypto callers
     pass ONLY a kind (manager.py / worker.py / probes), so the wrapper keeps that 1-arg contract and supports the
     opt-in 'persistent' kind (PersistentCliBrain -- one claude session carried across nodes).
  3. Marks this PROCESS as a metaop LOOP worker (METAOP_LOOP=1) at import time. The harness brains spawn their
     claude -p / SDK children with env={**os.environ, ...}, so the marker PROPAGATES into every worker child, where
     .claude/hooks/permission_gate.py auto-allows the worker's work but hard-denies commit/push + control-surface
     writes. This preserves the exact pre-dedup live-loop safety contract without coupling the harness to crypto.

No emoji (Windows cp1252).
"""
from __future__ import annotations

import os

# mark THIS process (the metaop loop) a worker so the permission gate governs its claude -p / SDK children. The
# harness brains inherit os.environ into the child env, so this single set covers every worker the loop spawns.
os.environ.setdefault("METAOP_LOOP", "1")

from harness.metaop import brain as _h  # the canonical engine
# re-export the full brain surface (classes + helpers) so existing imports are unchanged. LiteLLMBrain (+ its
# helpers) are re-exported so the live crypto loop can brain-swap through the unified LiteLLM gateway, and so the
# copy-parity firewall (which requires LiteLLMBrain in BOTH copies) holds.
from harness.metaop.brain import (  # noqa: F401
    Brain, MockBrain, AgentSdkBrain, AnthropicBrain, CliBrain, PersistentCliBrain, OllamaBrain, LiteLLMBrain,
    find_claude, _extract_json, _first_balanced_object, _litellm_available,
)

# The crypto-quant framing injected into every brain's {domain} slot (the live loop's historical contract: LONG-ONLY
# spot/perp; objective = robust held-out COMPOUND return; the unit of trading is a SETUP across a multi-candle MOVE,
# so per-bar IC is NOT the target).
CRYPTO_DOMAIN = ("a crypto-quant trading-research project (LONG-ONLY spot/perp; objective = robust held-out COMPOUND "
                 "return; the unit of trading is a SETUP across a multi-candle MOVE, so per-bar IC is NOT the target)")


def make_brain(kind: str = "auto", model: str | None = None) -> Brain:
    """Crypto entry point: select a backend with the crypto DOMAIN injected. Callers historically pass ONLY a kind
    (the 1-arg contract is preserved); the domain is fixed to CRYPTO_DOMAIN here. Supports the same kinds as the
    harness PLUS the opt-in 'persistent' (PersistentCliBrain -- one claude session carried across graph nodes) and
    'litellm' (the unified gateway). `model` (opt) overrides the model string for litellm/ollama backends."""
    return _h.make_brain(kind, domain=CRYPTO_DOMAIN, model=model)


if __name__ == "__main__":
    b = make_brain()
    print("brain:", b.name)
    print("plan :", b.decide("plan", {"objective": "characterize SOL"}))
    print("act  :", b.act("inspect repo", "run_shell,...", []))
