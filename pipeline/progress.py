"""Terminal progress and a run log under ``runs/``. Windows PowerShell compatible."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from pipeline.paths import repo_root

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tests install requirements
    tqdm = None


def _now_stamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_runs_root() -> Path:
    return repo_root() / "runs"


class RunLog:
    """Append-only log plus stderr. tqdm bars also go to stderr so the log stays clean."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("a", encoding="utf-8", newline="\n")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None  # type: ignore[assignment]

    def write(self, message: str, *, also_stderr: bool = True) -> None:
        line = message if message.endswith("\n") else message + "\n"
        stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        recorded = f"{stamp} {line}"
        if self._fh:
            self._fh.write(recorded)
            self._fh.flush()
        if also_stderr:
            sys.stderr.write(line)
            sys.stderr.flush()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Counters:
    def __init__(self) -> None:
        self.cached = 0
        self.written = 0
        self.failed = 0
        self.skipped = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "cached": self.cached,
            "written": self.written,
            "failed": self.failed,
            "skipped": self.skipped,
        }


class StageBar:
    """A tqdm bar for one stage (or one asset×stage)."""

    def __init__(self, description: str, total: int, unit: str, log: RunLog | None):
        self.description = description
        self.total = max(int(total), 0)
        self.unit = unit
        self.log = log
        self.counters = Counters()
        bar_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        if tqdm is not None and self.total >= 0:
            self._bar = tqdm(
                total=self.total or None,
                desc=description,
                unit=unit,
                file=sys.stderr,
                dynamic_ncols=True,
                mininterval=0.3,
                bar_format=bar_format,
                ascii=False,
            )
        else:
            self._bar = None

    def _postfix(self) -> str:
        c = self.counters
        return f"cached={c.cached} wrote={c.written} failed={c.failed}"

    def update(self, n: int = 1, *, cached: bool = False, written: bool = False, failed: bool = False, skipped: bool = False) -> None:
        if cached:
            self.counters.cached += n
        if written:
            self.counters.written += n
        if failed:
            self.counters.failed += n
        if skipped:
            self.counters.skipped += n
        if self._bar is not None:
            self._bar.set_postfix_str(self._postfix(), refresh=False)
            self._bar.update(n)

    def close(self) -> str:
        if self._bar is not None:
            self._bar.set_postfix_str(self._postfix(), refresh=True)
            self._bar.close()
        c = self.counters
        line = (
            f"[done] {self.description}  {self.total} {self.unit}s  "
            f"cached={c.cached} wrote={c.written} failed={c.failed}"
        )
        if self.log:
            self.log.write(line, also_stderr=True)
        else:
            sys.stderr.write(line + "\n")
        return line

    def __enter__(self) -> "StageBar":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RunSession:
    """One pipeline invocation: a log file, stage bars, and a final summary."""

    def __init__(
        self,
        *,
        market: str,
        start,
        end,
        stages: list[str],
        data_root: Path | None = None,
        runs_root: Path | None = None,
    ):
        runs_root = Path(runs_root) if runs_root is not None else default_runs_root()
        name = f"{market}_{start}_{end}_{_now_stamp()}.log"
        self.log = RunLog(runs_root / name)
        self.market = market
        self.start = start
        self.end = end
        self.stages = list(stages)
        self.asset_lines: list[str] = []
        self.errors: list[str] = []
        self._t0 = datetime.now(tz=timezone.utc)
        self.log.write(
            f"[run] market={market} {start} -> {end} stages={','.join(stages)} log={self.log.path}"
        )

    def bar(self, description: str, total: int, unit: str = "item") -> StageBar:
        return StageBar(description, total, unit, self.log)

    def info(self, message: str) -> None:
        self.log.write(message)

    def error(self, asset: str, when: str, exc: BaseException) -> None:
        detail = f"[error] {asset} {when}: {type(exc).__name__}: {exc}"
        self.errors.append(detail)
        self.log.write(detail)
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        self.log.write("".join(tb), also_stderr=False)

    def asset_summary(self, line: str) -> None:
        self.asset_lines.append(line)
        self.log.write(line)

    def finish(self, code: int) -> int:
        elapsed = (datetime.now(tz=timezone.utc) - self._t0).total_seconds()
        n_err = len(self.errors)
        line = (
            f"[summary] market={self.market} {self.start} -> {self.end}  "
            f"assets={len({(ln.split()[1] if len(ln.split())>1 else ln) for ln in self.asset_lines})} errors={n_err} exit={code} "
            f"elapsed_s={elapsed:.1f} log={self.log.path}"
        )
        self.log.write(line)
        if self.errors:
            self.log.write("[summary] failed units:")
            for err in self.errors[:20]:
                self.log.write(f"  {err}")
        self.log.close()
        return code

    def __enter__(self) -> "RunSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is KeyboardInterrupt:
            self.log.write("[run] interrupted (Ctrl+C). Re-run the same command to resume.")
        if self.log._fh:
            self.log.close()
