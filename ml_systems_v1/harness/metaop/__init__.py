"""harness.metaop -- a standalone, project-agnostic autonomous engine-builder loop.

A small, model-portable plan->dispatch->judge->reflect->route graph with a pluggable Brain. Nothing here imports
any host project; all run artifacts live under a configurable WORKSPACE (see config.py). See harness/README.md.
"""

from .brain import (make_brain, Brain, MockBrain, AgentSdkBrain, AnthropicBrain, CliBrain,  # noqa: F401
                    PersistentCliBrain, OllamaBrain, find_claude)
from .graph import build, make_nodes, OpState  # noqa: F401
from .tools import Tools  # noqa: F401

__all__ = ["make_brain", "Brain", "MockBrain", "AgentSdkBrain", "AnthropicBrain", "CliBrain",
           "PersistentCliBrain", "OllamaBrain", "find_claude",
           "build", "make_nodes", "OpState", "Tools"]
