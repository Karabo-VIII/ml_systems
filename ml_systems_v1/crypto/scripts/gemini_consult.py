"""gemini_consult.py -- external Gemini co-companion wrapper.

PURPOSE
-------
Shells out to Google Gemini for second-opinion / sanity-check / alternative-
implementation reviews. The wrapper is lazy: only imports google-generativeai
when invoked; gives helpful error messages if missing.

SETUP (one-time, user-side):
    pip install google-genai            # NEW SDK (google.generativeai is deprecated)
    # Set the API key — see scripts/gemini_setup_key.ps1 helper for safe entry:
    #   Windows PowerShell:  $env:GEMINI_API_KEY = "AIza..."  (session-only)
    #   Or persistent:       [Environment]::SetEnvironmentVariable("GEMINI_API_KEY","AIza...","User")
    #   bash/zsh:            export GEMINI_API_KEY=AIza...

USAGE:
    python scripts/gemini_consult.py "Review this diff: <paste diff>"
    python scripts/gemini_consult.py --file /tmp/code_to_review.py
    python scripts/gemini_consult.py --model gemini-1.5-pro "your prompt"

OUTPUT (printed to stdout):
    Gemini's response text. Designed for the Claude Code Bash tool to capture
    and feed into the foreground reasoning.

CONTRACT
--------
- Stateless (each invocation is fresh; no chat history)
- Streaming OFF (full response in one return)
- Default model: gemini-1.5-flash (fast + cheap)
- 30s default timeout

CAVEATS
-------
- Output flows back into the Claude session's context window — large
  responses consume Claude tokens. Prefer prompting Gemini for SHORT,
  STRUCTURED outputs.
- Gemini is a different model with different biases. Use for second
  opinions, NOT as a primary source of truth.
- Cost: Gemini Flash is ~$0.075 per 1M input tokens, $0.30 per 1M output.
  Pro is ~10x. Budget accordingly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", nargs="?", default=None,
                     help="Prompt text (or use --file)")
    ap.add_argument("--file", "-f", default=None,
                     help="Read prompt from file path")
    ap.add_argument("--model", "-m", default="gemini-1.5-flash",
                     help="Gemini model name (gemini-1.5-flash, gemini-1.5-pro)")
    ap.add_argument("--max-tokens", type=int, default=2048,
                     help="Max response tokens")
    ap.add_argument("--temperature", type=float, default=0.4,
                     help="Sampling temperature 0.0-1.0")
    ap.add_argument("--system", default=None,
                     help="Optional system-style preamble")
    args = ap.parse_args()

    # Resolve prompt
    prompt = args.prompt
    if args.file:
        prompt = Path(args.file).read_text(encoding="utf-8")
    if not prompt:
        print("ERROR: provide a prompt (positional arg or --file)",
                file=sys.stderr)
        return 2

    if args.system:
        prompt = f"{args.system}\n\n{prompt}"

    # API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY (or GOOGLE_API_KEY) not set in env.\n"
              "Setup:\n"
              "  PowerShell:  $env:GEMINI_API_KEY = 'AIza...'\n"
              "  bash/zsh:    export GEMINI_API_KEY=AIza...\n"
              "Get a key at https://aistudio.google.com/apikey",
              file=sys.stderr)
        return 2

    # Lazy import — new google-genai SDK
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError:
        print("ERROR: google-genai not installed.\n"
              "Install:  pip install google-genai",
              file=sys.stderr)
        return 2

    client = genai.Client(api_key=api_key)
    # Map old model names to new ones
    model = args.model
    if model == "gemini-1.5-flash":
        model = "gemini-2.0-flash"      # latest stable Flash
    elif model == "gemini-1.5-pro":
        model = "gemini-2.5-pro"        # latest stable Pro
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=args.max_tokens,
                temperature=args.temperature,
            ),
        )
    except Exception as e:
        print(f"ERROR: Gemini API call failed: {e}", file=sys.stderr)
        return 1

    # Extract text robustly
    try:
        text = resp.text
    except Exception:
        text = str(resp)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
