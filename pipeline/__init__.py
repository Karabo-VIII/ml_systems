"""Crypto data pipeline.

One orchestrator: ``python -m pipeline.cli``. The pre-rebuild project,
including the old pipeline, is the read-only snapshot at ``ml_systems_v1/``.
"""

from pipeline.constants import CONTRACT_VERSION

__version__ = CONTRACT_VERSION
