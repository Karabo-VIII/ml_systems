"""Pipeline v2: fetch, bars, features, alignment, datasets, validation.

One orchestrator (``python -m pipeline_v2.cli``). The previous tree is
reference-only at ``crypto/archive/pipeline_v1``.
"""

from pipeline_v2.constants import CONTRACT_VERSION

__version__ = CONTRACT_VERSION
