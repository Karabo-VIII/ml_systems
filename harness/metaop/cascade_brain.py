"""CASCADE ROUTER -- a cheap->strong escalation Brain (cost-aware routing behind the model-portability seam).

Motivation: most nodes are EASY (a weak/cheap local model solves them); a FEW are hard (need the strong model).
A flat single-brain harness either over-pays (strong on everything) or under-delivers (cheap on everything).
CascadeBrain runs the CHEAP brain FIRST and only ESCALATES to the STRONG brain when the cheap result is judged
INADEQUATE -- so the strong model is spent ONLY where it earns its cost.

It is itself a Brain (decide/act/work), so it drops into the SAME seam every other backend uses (graph.build,
manager.launch, make_brain). It wraps an ORDERED list [cheap, strong, ...]; `work()` is where the cascade lives
(that is the node-execution path the graph dispatches), and decide/act delegate to the cheap brain (escalating
decide/act would need a separate adequacy signal; the value is in node EXECUTION, where a mechanical verifier
exists).

ESCALATION TRIGGER (mechanical-first, three tiers, most-trusted first):
  (1) MECHANICAL (ground truth): if a verify_cmd is known for the node (wired via set_node_context(...) by a
      caller that has the node, e.g. a thin graph shim), the cheap result is ACCEPTED iff that verify_cmd exits 0
      when run from the build cwd -- reusing the harness's OWN verifier (graph._run_verify, which also screens
      destructive/trivial cmds). exit!=0 (or a screen-rejected cmd) => ESCALATE. This is the SAME ground truth the
      judge uses, so "the cascade accepted it" == "the judge will pass it" for that artifact. REAL, not heuristic.
  (2) WORKER-OK + QUALITY heuristic (no verify_cmd available): ESCALATE when the cheap result is structurally
      inadequate -- worker reported not-ok, OR the result is empty / a brain-error / "hit max steps" / explicitly
      REFUTED / a near-verbatim echo of the prompt / below a min length / contains a self-declared failure marker.
      This is a HEURISTIC (best-effort when no external check exists) and is documented as such.
  (3) The caller can also force escalation per node via set_node_context(needs_strong=True) (e.g. the planner
      flagged a hard node).

LOCAL + graceful (never crash if the strong model is absent):
  - make_cascade(...) builds [cheap, strong]; if only ONE ollama model is pulled, cheap IS strong (documented):
    the cascade still runs, escalation just re-tries the same model with a sharper prompt (a cheap second attempt).
  - If the strong brain ERRORS at escalation time, we DEGRADE to the cheap result rather than raising into the
    graph -- a provider/model outage can never stall the loop.

OBSERVABILITY: n_cheap (nodes the cheap brain satisfied), n_escalated (nodes that needed the strong brain),
n_escalation_failed (escalations where strong errored -> degraded to cheap), and a per-node `trail`. Exposed via
.stats() so the router's behavior is inspectable from a run.

No emoji (Windows cp1252). decide/act/work signatures match Brain exactly (drop-in).
"""
from __future__ import annotations

import threading

from .brain import Brain, MockBrain, DEFAULT_DOMAIN, make_brain


# markers that, appearing in a cheap result, signal a STRUCTURAL inadequacy (heuristic tier only)
_FAIL_MARKERS = (
    "hit max steps", "hit max_steps", "brain error", "_error", "refused", "i cannot", "i can't",
    "unable to", "no tool/final",
)


def _looks_refuted(result: str) -> bool:
    r = (result or "").strip().upper()
    return r.startswith("REFUTED") or "RESULT: REFUTED" in r


class CascadeBrain(Brain):
    """Cheap->strong escalation router. Wraps an ordered [cheap, strong, ...] list of Brains."""
    name = "CascadeBrain"

    def __init__(self, brains: list[Brain], domain: str = DEFAULT_DOMAIN,
                 min_result_len: int = 24, escalate_turns_note: bool = True, strong_decide: bool = True):
        if not brains:
            raise ValueError("CascadeBrain needs at least one brain")
        super().__init__(domain)
        self.brains = brains
        self.cheap = brains[0]
        self.strong = brains[-1]                 # == cheap when only one brain is supplied (single-model local case)
        self.single_model = len(brains) == 1 or (self.cheap is self.strong)
        self.min_result_len = min_result_len
        self.escalate_turns_note = escalate_turns_note
        # ROLE-SPLIT (default ON): route the high-leverage GENERATIVE-REASONING roles (plan + replan) to the STRONG
        # brain, because a bad PLAN/replan caps the whole run and -- unlike node EXECUTION -- has no per-result
        # mechanical verifier to recover it. judge + reflect stay cheap (judge is already mechanical-verifier-backed
        # for verify_cmd nodes; reflect is low-stakes). Cost: plan/replan fire ~once per run, so strong planning is
        # near-free vs the per-node execution. No-op when there is no distinct strong brain (single_model).
        self.strong_decide = strong_decide
        self.n_strong_decide = 0
        # per-node context the cascade consults for its MECHANICAL trigger (set by a caller that holds the node).
        # THREAD-LOCAL: graph dispatch (parallel>1) runs work() CONCURRENTLY on ONE shared cascade instance; with
        # instance-level context, node B's set_node_context would clobber node A's before A's work() reads it (a wrong
        # accept/escalate decision). Thread-local keeps each dispatch thread's (set_node_context -> work) pair
        # consistent. Accessed via the _verify_cmd/_verify_cwd/_force_strong properties below. (Observability counters
        # below stay shared -- concurrent increments are GIL-safe appends/adds and any tiny drift is cosmetic.)
        self._tl = threading.local()
        # observability
        self.n_cheap = 0
        self.n_escalated = 0
        self.n_escalation_failed = 0
        self.trail: list[dict] = []

    # --- per-node context, backed by the thread-local so concurrent dispatch threads never cross-contaminate -----
    @property
    def _verify_cmd(self) -> "str | None":
        return getattr(self._tl, "verify_cmd", None)

    @_verify_cmd.setter
    def _verify_cmd(self, v: "str | None") -> None:
        self._tl.verify_cmd = v

    @property
    def _verify_cwd(self) -> "str | None":
        return getattr(self._tl, "verify_cwd", None)

    @_verify_cwd.setter
    def _verify_cwd(self, v: "str | None") -> None:
        self._tl.verify_cwd = v

    @property
    def _force_strong(self) -> bool:
        return getattr(self._tl, "force_strong", False)

    @_force_strong.setter
    def _force_strong(self, v: bool) -> None:
        self._tl.force_strong = bool(v)

    # --- caller hook: wire the per-node MECHANICAL trigger (optional; absent => heuristic tier) ------------------
    def set_node_context(self, verify_cmd: str | None = None, cwd: str | None = None,
                         needs_strong: bool = False) -> None:
        """Tell the cascade about the NEXT node it will work(): its verify_cmd (=> mechanical accept/escalate),
        the build cwd to run it in, and/or a planner force-escalate flag. Call right before brain.work(...). The
        graph itself does not pass nodes into work(), so a thin caller that HAS the node wires this; without it the
        cascade falls back to the worker-ok+quality heuristic (still safe, just not ground-truth)."""
        self._verify_cmd = verify_cmd
        self._verify_cwd = cwd
        self._force_strong = bool(needs_strong)

    def _clear_node_context(self) -> None:
        self._verify_cmd = None
        self._verify_cwd = None
        self._force_strong = False

    def stats(self) -> dict:
        return {"n_cheap": self.n_cheap, "n_escalated": self.n_escalated,
                "n_escalation_failed": self.n_escalation_failed,
                "n_strong_decide": self.n_strong_decide, "strong_decide": self.strong_decide,
                "single_model": self.single_model,
                "cheap": self.cheap.name, "strong": self.strong.name}

    # --- the ADEQUACY decision (the escalation trigger) ---------------------------------------------------------
    def _mechanically_passes(self, cwd: str | None) -> tuple[bool, str]:
        """Run the wired verify_cmd via the harness's OWN mechanical verifier (ground truth). Returns
        (passed, reason). Lazy import of graph avoids any import cycle (graph receives a brain, doesn't import one)."""
        from . import graph as _graph
        from .config import build_cwd
        run_cwd = cwd or self._verify_cwd or str(build_cwd())
        code, tail = _graph._run_verify(self._verify_cmd, run_cwd)
        return (code == 0, f"verify_exit={code}" + ("" if code == 0 else f" :: {str(tail)[:160]}"))

    def _heuristic_adequate(self, res: dict) -> tuple[bool, str]:
        """No verify_cmd => best-effort QUALITY signal. Returns (adequate, reason)."""
        if not isinstance(res, dict) or not res.get("ok"):
            return (False, "worker_ok=False")
        result = str(res.get("result", ""))
        low = result.lower()
        if not result.strip():
            return (False, "empty result")
        if _looks_refuted(result):
            return (False, "result is REFUTED")
        if any(m in low for m in _FAIL_MARKERS):
            return (False, "result carries a failure marker")
        if len(result.strip()) < self.min_result_len:
            return (False, f"result too short (<{self.min_result_len} chars)")
        return (True, "worker_ok + quality heuristic")

    def _adequate(self, res: dict, cwd: str | None) -> tuple[bool, str, str]:
        """Decide if the CHEAP result is good enough. Returns (adequate, tier, reason).
        Priority: forced-strong > mechanical(verify_cmd) > worker-ok+quality heuristic."""
        if self._force_strong:
            return (False, "forced", "caller set needs_strong=True")
        if self._verify_cmd:
            ok, reason = self._mechanically_passes(cwd)
            return (ok, "mechanical", reason)
        ok, reason = self._heuristic_adequate(res)
        return (ok, "heuristic", reason)

    # --- Brain interface ---------------------------------------------------------------------------------------
    def decide(self, role: str, payload: dict, persona: str = "") -> dict:
        # ROLE-SPLIT: the GENERATIVE-REASONING roles (plan + replan) go to the STRONG brain when strong_decide is on
        # and a distinct strong brain exists -- a weak planner caps the whole run and can't be mechanically recovered
        # the way EXECUTION can. judge + reflect stay cheap (judge is verifier-backed; reflect is low-stakes). Strong
        # plan/replan fire ~once per run, so this buys planning quality at near-zero marginal cost. Degrades to cheap
        # if the strong brain errors (never stall).
        if self.strong_decide and not self.single_model and role in ("plan", "replan"):
            try:
                out = self.strong.decide(role, payload, persona)
                self.n_strong_decide += 1
                return out
            except Exception:
                return self.cheap.decide(role, payload, persona)
        return self.cheap.decide(role, payload, persona)

    def act(self, task: str, tools_schema: str, history: list) -> dict:
        return self.cheap.act(task, tools_schema, history)

    def work(self, task: str, persona: str = "", cwd: str | None = None) -> dict:
        """Run the node on the CHEAP brain; if INADEQUATE, escalate to the STRONG brain. Returns the SAME
        {ok, result} contract every Brain.work returns, plus cascade telemetry under 'cascade'."""
        verify_cmd = self._verify_cmd          # snapshot (cleared at the end so the next work() is clean)
        cheap_res = self.cheap.work(task, persona=persona)
        adequate, tier, reason = self._adequate(cheap_res, cwd)

        if adequate:
            self.n_cheap += 1
            self.trail.append({"escalated": False, "tier": tier, "reason": reason, "by": self.cheap.name})
            self._clear_node_context()
            out = dict(cheap_res)
            out["cascade"] = {"escalated": False, "tier": tier, "reason": reason, "by": self.cheap.name}
            return out

        # ---- ESCALATE -----------------------------------------------------------------------------------------
        # When cheap IS strong (single local model), re-attempt with a sharper prompt (a cheap 2nd try). When a
        # distinct strong brain exists, hand it the task PLUS the cheap attempt + why it was inadequate (gradient).
        if self.single_model:
            esc_task = task + ("\n\n[CASCADE retry on the same local model: the first attempt was INADEQUATE ("
                               f"{reason}). Be concrete, run a real tool, and verify by running.]")
        else:
            esc_task = task + (
                f"\n\n[CASCADE ESCALATION: a cheaper model attempted this and its result was INADEQUATE ({reason}). "
                f"Its attempt:\n{str(cheap_res.get('result',''))[:600]}\nProduce a CORRECT, verified artifact.]")
        if verify_cmd:
            esc_task += f"\n[A mechanical verifier will run `{verify_cmd}`; make it exit 0.]"

        try:
            strong_res = self.strong.work(esc_task, persona=persona)
        except Exception as e:
            # strong model absent/errored -> DEGRADE to cheap (never stall the loop)
            self.n_escalation_failed += 1
            self.trail.append({"escalated": True, "tier": tier, "reason": reason, "by": self.cheap.name,
                               "escalation_failed": f"{type(e).__name__}: {str(e)[:120]}"})
            self._clear_node_context()
            out = dict(cheap_res)
            out["cascade"] = {"escalated": True, "degraded_to_cheap": True, "tier": tier, "reason": reason,
                              "escalation_error": f"{type(e).__name__}: {str(e)[:120]}"}
            return out

        self.n_escalated += 1
        self.trail.append({"escalated": True, "tier": tier, "reason": reason,
                           "from": self.cheap.name, "to": self.strong.name})
        self._clear_node_context()
        out = dict(strong_res)
        out["cascade"] = {"escalated": True, "tier": tier, "reason": reason,
                          "from": self.cheap.name, "to": self.strong.name}
        return out


def make_cascade(cheap_kind: str = "ollama", strong_kind: str | None = None, domain: str = DEFAULT_DOMAIN,
                 cwd: str | None = None, cheap_model: str | None = None, strong_model: str | None = None) -> CascadeBrain:
    """Build a [cheap, strong] CascadeBrain via make_brain. LOCAL-graceful:
      - strong_kind=None  -> strong = cheap (single model; escalation = a sharper retry on the same model).
      - strong_model set with the SAME kind -> a LARGER local model as strong (e.g. cheap=qwen3b, strong=qwen7b),
        used only if present; if make_brain can't build it, the cascade still works (degrades to cheap at runtime).
    Both brains share the domain + build cwd so their workers write where the mechanical verifier runs."""
    cheap = make_brain(cheap_kind, domain=domain, cwd=cwd, model=cheap_model)
    if strong_kind is None and strong_model is None:
        return CascadeBrain([cheap], domain=domain)            # single-model local case (documented)
    strong = make_brain(strong_kind or cheap_kind, domain=domain, cwd=cwd, model=strong_model)
    return CascadeBrain([cheap, strong], domain=domain)
