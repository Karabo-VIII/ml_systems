"""wire_model.py -- turn a local GGUF in the project's ./models/ folder into a runnable ollama model,
ready for scripts/autonomy/benchmark_brain.py. The workflow for testing a candidate brain:

    1. put a .gguf in  <project>/models/     (download there: hf download <repo> <file> --local-dir models)
    2. python scripts/autonomy/wire_model.py models/<file>.gguf   (or no arg = newest .gguf in models/)
    3. python scripts/autonomy/benchmark_brain.py --models <printed-name> qwen2.5-coder:7b

It writes a tiny Modelfile (FROM <abs gguf> + a num_ctx cap so big-context MoE models don't OOM your KV
cache) and runs `ollama create`. num_ctx default 32768 (lower with --num-ctx if you hit OOM on 8GB VRAM /
32GB RAM). Windowless subprocess (no console-flash on Windows). models/ is gitignored (GGUFs are multi-GB)."""
import argparse, os, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
NW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
try:  # ollama prints Braille spinner chars; the Windows cp1252 console can't encode them -> make stdout safe
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def clean_name(stem: str) -> str:
    """Derive a short ollama tag from a GGUF filename, e.g.
    'Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M' -> 'qwen3.6-35b-a3b:q4_k_m'."""
    s = stem.lower()
    qm = re.search(r"(iq\d_[a-z0-9]+|q\d_k_[a-z]|q\d_k|q\d_[a-z0-9]+|iq\d_[a-z]+|f16|bf16|q\d)", s)
    quant = qm.group(1) if qm else "gguf"
    base = s[:qm.start()] if qm else s
    base = re.sub(r"[^a-z0-9.]+", "-", base).strip("-.")[:42] or "model"
    return f"{base}:{quant}"


def find_gguf(arg):
    if arg:
        p = Path(arg)
        for cand in (p, MODELS / arg, MODELS / Path(arg).name):
            if cand.exists():
                return cand
        sys.exit(f"GGUF not found: {arg}")
    ggufs = sorted(MODELS.glob("**/*.gguf"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not ggufs:
        inc = list(MODELS.glob("**/*.incomplete"))
        sys.exit(f"no .gguf in {MODELS}" + (f"  ({len(inc)} unfinished .incomplete partial(s) -- finish the download first)" if inc else ""))
    return ggufs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", nargs="?", help="path to a .gguf (default: newest in models/)")
    ap.add_argument("--name", help="ollama model name (default: derived from filename)")
    ap.add_argument("--num-ctx", type=int, default=32768, help="context cap (KV memory); lower if OOM on 8GB VRAM")
    ap.add_argument("--list", action="store_true", help="just list candidate GGUFs in models/ and exit")
    a = ap.parse_args()
    MODELS.mkdir(exist_ok=True)

    if a.list:
        ggufs = sorted(MODELS.glob("**/*.gguf"))
        print(f"GGUFs in {MODELS}:")
        for g in ggufs:
            print(f"  {g.stat().st_size/1e9:6.1f} GB  {g.relative_to(MODELS)}  -> would wire as '{clean_name(g.stem)}'")
        if not ggufs:
            print("  (none)")
        return 0

    gguf = find_gguf(a.gguf)
    name = a.name or clean_name(gguf.stem)
    size_gb = gguf.stat().st_size / 1e9
    print(f"wiring  {gguf.name}  ({size_gb:.1f} GB)  ->  ollama '{name}'  (num_ctx={a.num_ctx})")
    modelfile = f"FROM {gguf.resolve().as_posix()}\nPARAMETER num_ctx {a.num_ctx}\n"
    mf = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False, dir=str(MODELS), encoding="utf-8") as fh:
            fh.write(modelfile); mf = fh.name
        r = subprocess.run(["ollama", "create", name, "-f", mf], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1800, creationflags=NW)
        out = (r.stdout or "").strip() or (r.stderr or "").strip()
        print(out[-600:])
        if r.returncode != 0:
            print(f"\nFAILED (ollama create exit {r.returncode}).")
            sys.exit(2)
    finally:
        if mf:
            try: os.remove(mf)
            except OSError: pass
    print(f"\nWIRED: '{name}'. Benchmark it:\n  python scripts/autonomy/benchmark_brain.py --models {name} qwen2.5-coder:7b")


if __name__ == "__main__":
    main()
