"""
Shared logging utility -- Tees stdout to both terminal and a log file.

Usage (in any training or validation script):
    from log_utils import setup_logging
    setup_logging(LOG_DIR, "v1_train")       # -> logs/v1/v1_train_20260216_143022.log
    setup_logging(LOG_DIR, "v1_validate")    # -> logs/v1/v1_validate_20260216_143022.log

All subsequent print() calls will be captured to both the terminal and the log file.
Call teardown_logging() at the end to flush and restore stdout (optional but clean).
"""
import sys
import io
from pathlib import Path
from datetime import datetime


class TeeLogger:
    """Duplicates writes to both the original stdout and a log file."""

    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_path = log_path
        self.log_file = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.flush()
        self.log_file.close()

    # Support isatty() for tqdm compatibility
    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

    # Proxy all other attributes to terminal (for tqdm, encoding, etc.)
    def __getattr__(self, name):
        return getattr(self.terminal, name)


_active_logger = None


def setup_logging(log_dir: Path, prefix: str) -> Path:
    """
    Start tee-logging to a timestamped file in log_dir.

    Args:
        log_dir: Directory for log files (e.g., LOG_DIR from settings)
        prefix: Filename prefix (e.g., "v1_train", "v3_validate")

    Returns:
        Path to the created log file
    """
    global _active_logger
    if _active_logger is not None:
        teardown_logging()

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{prefix}_{timestamp}.log"

    _active_logger = TeeLogger(log_path)
    sys.stdout = _active_logger
    return log_path


def teardown_logging():
    """Restore original stdout and close the log file."""
    global _active_logger
    if _active_logger is not None:
        sys.stdout = _active_logger.terminal
        _active_logger.close()
        _active_logger = None
