"""Install/availability check for Kronos.

Kronos is distributed as a github repo (shiyu-coder/Kronos), not a pip
package. The model weights live on HuggingFace under NeoQuasar/.

This script:
    1. Verifies HuggingFace + transformers are installed
    2. Tries to import the kronos module from a local clone
       (default: ./external/Kronos/, override with KRONOS_PATH env var)
    3. If not found, prints clear install instructions
    4. Smoke-tokenizes a fake OHLCV bar to verify the tokenizer loads

Run:
    python -m src.frontier_ml.kronos_baseline.install_check
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KRONOS_LOCAL_DEFAULT = PROJECT_ROOT / "external" / "Kronos"


def announce(msg: str) -> None:
    print(f"[kronos-check] {msg}", flush=True)


def main():
    announce("checking dependencies...")

    # 1. transformers + huggingface_hub
    try:
        import transformers, huggingface_hub
        announce(f"  transformers      = {transformers.__version__}  (need >=4.40)")
        announce(f"  huggingface_hub   = {huggingface_hub.__version__}")
    except ImportError as e:
        announce(f"FAIL: missing dependency: {e}")
        announce("    install: pip install -U transformers huggingface_hub")
        sys.exit(1)

    # 2. Local Kronos clone or pip package
    kronos_path = Path(os.environ.get("KRONOS_PATH", str(KRONOS_LOCAL_DEFAULT)))
    kronos_module = None
    src_path = kronos_path / "model"  # the repo's importable layout
    candidates = [
        kronos_path / "model",          # shiyu-coder/Kronos repo layout
        kronos_path,                     # if user dropped it directly
    ]

    if not kronos_path.exists():
        announce(f"FAIL: local clone not found at {kronos_path}")
        announce("    To install:")
        announce(f"      mkdir -p {kronos_path.parent}")
        announce(f"      cd {kronos_path.parent}")
        announce(f"      git clone https://github.com/shiyu-coder/Kronos.git")
        announce("    Then re-run this script.")
        sys.exit(2)

    for cand in candidates:
        if cand.exists():
            sys.path.insert(0, str(cand))
            try:
                from model import Kronos, KronosTokenizer, KronosPredictor   # type: ignore
                kronos_module = (Kronos, KronosTokenizer, KronosPredictor)
                announce(f"  kronos package    = importable from {cand}")
                break
            except Exception as e:
                announce(f"  WARN: import from {cand} failed: {type(e).__name__}: {e}")

    if kronos_module is None:
        announce("FAIL: could not import Kronos / KronosTokenizer / KronosPredictor")
        announce(f"    Confirm the repo at {kronos_path} has model/__init__.py "
                 f"exposing Kronos + KronosTokenizer + KronosPredictor.")
        sys.exit(3)

    Kronos, KronosTokenizer, KronosPredictor = kronos_module

    # 3. Try to load the tokenizer (no model download yet -- just tokenizer
    # is small enough that the smoke is cheap).
    try:
        announce("loading NeoQuasar/Kronos-Tokenizer-base from HF...")
        tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        announce(f"  tokenizer loaded: {type(tok).__name__}")
    except Exception as e:
        announce(f"FAIL: tokenizer load failed: {type(e).__name__}: {e}")
        announce("    This may be a network issue (huggingface.co reachable?)")
        announce("    or a kronos repo version mismatch.")
        sys.exit(4)

    # 4. Tokenize a fake bar (smoke)
    try:
        import torch
        # Kronos expects (open, high, low, close, volume, amount) per bar.
        # Construct a fake 32-bar window of synthetic OHLCV.
        n = 32
        close = torch.linspace(100.0, 110.0, n)
        high = close + 0.5
        low = close - 0.5
        open_ = close - 0.1
        volume = torch.full((n,), 1000.0)
        amount = volume * close
        # Stack to (n, 6)
        ctx = torch.stack([open_, high, low, close, volume, amount], dim=-1).float()
        announce(f"  fake context shape: {tuple(ctx.shape)}")
        # Most KronosTokenizer impls expect a (B, T, 6) or (B, T, 5) tensor.
        x = ctx.unsqueeze(0)
        try:
            tokens = tok.encode(x)
            t_shape = tuple(tokens.shape) if hasattr(tokens, "shape") else tokens
            announce(f"  tokens after encode: shape/repr={t_shape}")
        except AttributeError:
            announce("  tokenizer has no .encode() method -- API may have changed; "
                     "we'll discover the right call in eval_kronos.py")
    except Exception as e:
        announce(f"WARN: tokenizer smoke failed but tokenizer loaded: {e}")

    announce("PASS: kronos available; ready to run eval_kronos.py")


if __name__ == "__main__":
    main()
