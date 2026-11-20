"""Canonical symbol form. One conversion, at the edge."""

from __future__ import annotations


def canonical_symbol(raw: str) -> str:
    """BTC, btcusdt, BTC/USDT, and BTC-USDT all become BTCUSDT."""
    s = raw.strip().upper().replace(" ", "")
    if "/" in s:
        base, quote = s.split("/", 1)
        s = base + quote
    s = s.replace("-", "").replace("_", "")
    if not s.endswith("USDT"):
        s = s + "USDT"
    if not s.endswith("USDT") or len(s) <= 4:
        raise ValueError(f"cannot canonicalise symbol {raw!r}")
    return s


def parse_symbol_list(items: list[str] | None) -> list[str]:
    if not items:
        return []
    out: list[str] = []
    for item in items:
        for part in item.split(","):
            part = part.strip()
            if part:
                out.append(canonical_symbol(part))
    # preserve order, drop duplicates
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq
