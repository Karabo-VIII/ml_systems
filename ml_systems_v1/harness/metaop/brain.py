"""Harness BRAIN -- the pluggable intelligence. The GRAPH is the awake loop; the BRAIN is what thinks per node.

This is the model-portability seam: the harness OUTLIVES any one model. Every backend implements the SAME small
interface, so you can drop in a new model (e.g. an OllamaBrain) without touching the graph:

  decide(role, payload, persona="") -> dict   : one-shot structured reasoning (plan/judge/reflect)
  act(task, tools_schema, history)  -> dict    : one ReAct step -> {"action":"tool",tool,args} | {"action":"final",result}
  work(task, persona="")            -> dict    : DO a node's task end-to-end -> {"ok":bool, "result":str}

Backends shipped:
  MockBrain       (default, NO creds)  -- deterministic + role/task-aware; drives the FULL loop incl. real tool work
  AnthropicBrain  ANTHROPIC_API_KEY    -- real Claude via the anthropic SDK (retried, JSON-coerced); ReAct work loop
  AgentSdkBrain   claude-agent-sdk     -- in-process Claude agent (fast); the Claude-specific impl behind the seam
  CliBrain        `claude` on PATH     -- real Claude via the headless CLI (uses existing auth, no API key)
  LiteLLMBrain    `pip install litellm` -- UNIFIED gateway: ANY provider via one API (anthropic/openai/ollama/...),
                                          auto-fallback chain (primary -> local ollama -> MockBrain); ReAct work loop

DOMAIN-AGNOSTIC: the system prompts carry a `{domain}` slot. Default domain = a generic "engine-builder" framing;
pass `domain=` to make/Brain to specialize (e.g. "a crypto-quant trading-research repo"). To add an OllamaBrain:
subclass Brain, implement decide/act/work against your local model, register it in make_brain. No emoji (cp1252).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess


def _first_balanced_object(text: str) -> str | None:
    """Return the first complete, brace-balanced {...} object in text (string/escape aware), else None.
    Weaker local models often emit valid JSON then trail extra prose or a SECOND object; a greedy `{.*}` would
    swallow that and break json.loads ('Extra data'). Scanning for the first balanced object is model-portable."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _extract_json(text: str) -> dict:
    if not text:
        return {"_error": "empty"}
    # prefer a fenced ```json block; else the FIRST brace-balanced object (tolerates trailing prose / a 2nd object).
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = m.group(1) if m else _first_balanced_object(text)
    if candidate is None:
        return {"_error": "no json", "_raw": text[:300]}
    try:
        return json.loads(candidate)
    except Exception as e:
        return {"_error": f"json parse: {e}", "_raw": text[:300]}


# --------------------------------------------------------------------------- domain + system prompts
# The harness is project-agnostic. The {domain} slot is the ONLY place a caller injects task-flavor; everything
# else is a generic, transferable engine-builder contract. Default domain is deliberately neutral.
DEFAULT_DOMAIN = "a software engine-builder project (build verifiable artifacts in any domain)"

# --------------------------------------------------------------------------- the PLANNER INSTRUCTION (DSPy seam)
# This is the ONE planner-quality knob, isolated as a single named string so a future DSPy MIPROv2 pass can optimize
# it against the eval harness (eval_harness.solve_rate) WITHOUT touching the graph or the rest of the prompt. It is
# the body of the `plan` role contract inside _DECIDE_SYS_T below (interpolated at {plan_instruction}); keep the
# OUTPUT-CONTRACT line first (the strict JSON schema the schema/tests depend on) and let the rest be the optimizable
# reasoning recipe. DSPy-READY: an optimizer mutates _PLAN_INSTRUCTION (or supplies a replacement string), re-runs
# eval_harness_run.py, and keeps the variant with the higher solve_rate. NOT integrated now (no DSPy dependency).
# NOTE: this string is `.format()`-ed inside _DECIDE_SYS_T, so any literal brace MUST be doubled ({{ }}).
_PLAN_INSTRUCTION = (
    "{{\"frontier\":[{{\"id\":str,\"task\":str,\"ev\":0..1,\"kind\":\"build|verify|diverge\",\"status\":\"open\","
    "\"verify_cmd\":str}}]}}\n"
    "   Output EXACTLY that JSON object -- 3-6 concrete, independently-checkable nodes. PLAN WELL (do not commit to "
    "the first idea):\n"
    "   0. VERIFY_CMD IS MANDATORY on every build node (and any node that produces an artifact): a SHELL COMMAND, run "
    "from the build cwd, that exits 0 ONLY IF the artifact is correct -- e.g. "
    "`python -c \"from mod import fn; assert fn(x)==y; print('ok')\"` or `python test_x.py`. This is how a build is "
    "MECHANICALLY confirmed (exit 0 = ground-truth pass, overriding any opinion). A build node WITHOUT a verify_cmd "
    "cannot be credited and will be refuted -- so ALWAYS include one that genuinely tests the required behavior. Make "
    "the task and its verify_cmd agree on the exact artifact name + interface (file name, function name, signature).\n"
    "   1. APPROACHES: silently consider 2-3 DISTINCT candidate approaches to the objective, then SELECT the best "
    "(most robust + most checkable) -- the frontier should reflect the chosen approach, not the first thing you "
    "thought of.\n"
    "   2. DECOMPOSE n+-k: the frontier MUST contain (a) the primary objective node(s) [the 'n', kind=build]; (b) at "
    "least ONE FALSIFIER [the '-k', kind=verify] -- a node that AUDITS soundness / tries to REFUTE the premise "
    "(is the method/data/spec sound? does a positive control pass? is there a leak/look-ahead?); (c) at least ONE "
    "GENERALIZATION [the '+k', kind=diverge] -- the adjacent / more-general case the solution should also handle. A "
    "single-path frontier with no falsifier and no generalization is REJECTED.\n"
    "   3. USE THE PAYLOAD: if PAYLOAD.framing is present, draw breadth from its axes/jolts and OBEY its "
    "anti-impossible rail (never declare the objective impossible -- find the smallest checkable step). If "
    "PAYLOAD.recall / similar_past_cycles / mem0_recall / prior_project_learnings are present, REUSE their validated "
    "assets + open hypotheses and do NOT re-propose anything on their dead/refuted list (never re-mine a refuted "
    "vein). BUILD ON prior learnings -- never re-settle what they already settled.\n"
    "   4. SELF-CRITIQUE then revise ONCE: before returning, check your drafted frontier -- does it cover the "
    "objective? is it single-path (add an alternative)? is the falsifier (kind=verify) present? is the generalization "
    "(kind=diverge) present? are any nodes re-mining a refuted vein? Fix the gaps, THEN emit the final JSON.\n"
    "   ev = expected value 0..1. If PAYLOAD.expert_mode is set, ALSO add \"expert\":<one of available_experts> to "
    "each node (the specialist best suited)."
)

# Optional, CHEAP second-pass critique role (graph.plan calls it only when the drafted frontier is missing a
# falsifier/generalization; default ON, degradable -- any error leaves the frontier untouched). Kept tiny so it adds
# at most one short LLM call. Its output is MERGED into the frontier by the graph, never replaces it.
_PLAN_CRITIQUE_INSTRUCTION = (
    "{{\"add\":[{{\"id\":str,\"task\":str,\"ev\":0..1,\"kind\":\"build|verify|diverge\",\"status\":\"open\"}}]}}\n"
    "   PAYLOAD.frontier is a DRAFT plan + PAYLOAD.missing names what it lacks ('falsifier' (kind=verify) and/or "
    "'generalization' (kind=diverge)). Return ONLY the MINIMAL extra node(s) that fill the named gap(s) for "
    "PAYLOAD.objective -- a real FALSIFIER (kind=verify, audits soundness / a positive control / a leak check) when "
    "'falsifier' is missing, and a real GENERALIZATION (kind=diverge, the adjacent/general case) when 'generalization' "
    "is missing. Add NOTHING else (return {{\"add\":[]}} if nothing is missing). Do not re-list existing nodes; use "
    "fresh ids. These are breadth GUARDS, so do not pad."
)

# _DECIDE_SYS_T is now built by a FACTORY so the plan-role contract is a single substitutable slot (the DSPy /
# evolve seam): _build_decide_sys(plan_instruction) interpolates ANY plan instruction in place of _PLAN_INSTRUCTION,
# leaving every other role contract byte-identical. The module global below is the DEFAULT (built with
# _PLAN_INSTRUCTION) so all existing callers/prompts are unchanged; a Brain with `self.plan_instruction` set rebuilds
# the template with the override (see Brain._decide_sys). `plan_instruction` MUST use doubled braces for literal
# braces (same .format() contract as _PLAN_INSTRUCTION) -- evolve_planner mutates copies of _PLAN_INSTRUCTION, so the
# brace convention is preserved automatically.
def _build_decide_sys(plan_instruction: str = _PLAN_INSTRUCTION) -> str:
    return (
    "You are a reasoning node in an autonomous meta-orchestration graph for {domain}. Respond with EXACTLY ONE "
    "JSON object and nothing else. Be rigorous and HONEST -- every claim must be checkable by running code (verify "
    "by running, do not assert); never claim what wasn't verified; a refutation/null is a valid, valuable result. "
    "Role contracts:\n"
    " plan    -> " + plan_instruction + "\n"
    " plan_critique -> " + _PLAN_CRITIQUE_INSTRUCTION + "\n"
    " judge   -> {{\"verdict\":\"pass|refuted|inconclusive\",\"reason\":str}}\n"
    "   REFUSE false victory: 'pass' ONLY when the result shows VERIFIED evidence meeting the success criteria. A "
    "result that asserts success without running/showing evidence is NOT a pass. Default 'inconclusive' if unproven; "
    "'refuted' if evidence contradicts the claim.\n"
    " reflect -> {{\"lesson\":str,\"adjacent\":[{{\"id\":str,\"task\":str,\"ev\":0..1,\"kind\":str,\"status\":\"open\"}}]}}\n"
    "   lesson = ONE durable, transferable takeaway (stored project-wide). adjacent = NEW problems THIS cycle "
    "genuinely opened; return [] when the neighborhood is exhausted (never pad to look busy). If "
    "PAYLOAD.external_guidance contains cross-cutting guidance, PRIORITIZE it -- turn it into concrete adjacent nodes."
    "\n replan  -> {{\"frontier\":[{{\"id\":str,\"task\":str,\"ev\":0..1,\"kind\":\"build|verify|diverge\",\"status\":\"open\"}}]}}\n"
    "   The current plan is FAILING (PAYLOAD.replan_reason says how: stall / repeated-failure / approach-wrong). "
    "Return a REVISED frontier, NOT an append. Look at PAYLOAD.current_frontier (each node's status/verdict/"
    "verify_error) + PAYLOAD.lessons and: PRUNE the doomed/refuted nodes (OMIT them -- do not re-list a node that "
    "exhausted its retries with the SAME task), KEEP the genuinely-open promising ones (re-list them by id), and ADD "
    ">=1 NEW-APPROACH node that attacks the objective DIFFERENTLY (a new method/decomposition, not a reword of a "
    "refuted node). At least one node must be genuinely new. ev = expected value 0..1."
    )


_DECIDE_SYS_T = _build_decide_sys()  # the DEFAULT system template (built with _PLAN_INSTRUCTION) -- unchanged contract


_ACT_SYS_T = (
    "You are a tool-using worker agent executing ONE task in {domain}. You have these tools:\n"
    "  {tools}\n"
    "Each turn respond with ONE JSON object: {{\"thought\":str, \"action\":\"tool\"|\"final\", "
    "\"tool\":name, \"args\":{{...}}, \"result\":str}}. Use 'tool' to run a step (then you'll see its output), "
    "'final' when the task is done (put the verified answer/evidence in 'result'). Verify by RUNNING, don't assert. "
    "Stay within the safety fence (destructive ops are blocked)."
)
_WORK_SYS_T = (
    "You are an autonomous worker agent in {domain} (cwd = the target project root). Verify by RUNNING -- never "
    "report a number/result you did not produce from a command. Be honest about null/negative results (a refutation "
    "is valuable). SAFETY: do NOT commit/push/deploy or take any irreversible action. Finish with exactly one line: "
    "'RESULT: <evidence-backed answer, or REFUTED: <why>>'."
)


def find_claude() -> str | None:
    """Locate the claude CLI: PATH first, else the VS Code extension's bundled native binary (latest version)."""
    p = shutil.which("claude")
    if p:
        return p
    import glob
    home = os.path.expanduser("~")
    cands = sorted(glob.glob(os.path.join(home, ".vscode", "extensions", "anthropic.claude-code-*",
                                          "resources", "native-binary", "claude.exe")))
    return cands[-1] if cands else None


class Brain:
    """The model-portability interface. Implement these three to wire ANY model into the harness."""
    name = "Brain"

    def __init__(self, domain: str = DEFAULT_DOMAIN):
        self.domain = domain or DEFAULT_DOMAIN
        # OPTIONAL plan-instruction OVERRIDE (the DSPy / evolve seam). None -> the default _PLAN_INSTRUCTION contract
        # (byte-identical prompt; agnostic path unchanged). When set to a string, _decide_sys rebuilds the system
        # template with THIS plan instruction in the `plan` role slot -- letting evolve_planner inject a mutated
        # planner prompt without touching the graph. Set per-instance via set_plan_instruction(...).
        self.plan_instruction: str | None = None

    def set_plan_instruction(self, plan_instruction: str | None) -> "Brain":
        """Inject (or clear with None) the planner-prompt override. Returns self for chaining. This is the single
        DSPy/evolve hook: an optimizer sets a mutated _PLAN_INSTRUCTION here, re-runs the eval, keeps the winner."""
        self.plan_instruction = plan_instruction
        return self

    def decide(self, role: str, payload: dict, persona: str = "") -> dict: raise NotImplementedError
    def act(self, task: str, tools_schema: str, history: list) -> dict: raise NotImplementedError
    def work(self, task: str, persona: str = "") -> dict: raise NotImplementedError
    # persona (optional) = an expert's system prompt, prepended in expert mode

    # prompt builders (use the injected domain) -----------------------------
    def _decide_sys(self) -> str:
        # default: the module template (built with _PLAN_INSTRUCTION). When a per-instance plan_instruction override is
        # set, rebuild the template with it -> only the `plan` role contract changes; every other role is identical.
        tmpl = _DECIDE_SYS_T if not self.plan_instruction else _build_decide_sys(self.plan_instruction)
        return tmpl.format(domain=self.domain)

    def _act_sys(self, tools_schema: str) -> str:
        return _ACT_SYS_T.format(domain=self.domain, tools=tools_schema)

    def _work_sys(self) -> str:
        return _WORK_SYS_T.format(domain=self.domain)


class AnthropicBrain(Brain):
    name = "AnthropicBrain"

    def __init__(self, model: str | None = None, max_retries: int = 3, domain: str = DEFAULT_DOMAIN):
        super().__init__(domain)
        import anthropic  # lazy
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("HARNESS_MODEL", "claude-opus-4-8")
        self.max_retries = max_retries

    def _call(self, system: str, user: str, max_tokens: int = 2000) -> str:
        import anthropic
        last = ""
        for i in range(self.max_retries):
            try:
                m = self.client.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                                 messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in m.content if getattr(b, "type", "") == "text")
            except anthropic.APIError as e:
                last = str(e)
                if i == self.max_retries - 1:
                    raise
        return last

    def decide(self, role: str, payload: dict, persona: str = "") -> dict:
        sysp = (persona + "\n\n" + self._decide_sys()) if persona else self._decide_sys()
        out = self._call(sysp, f"ROLE: {role}\nPAYLOAD:\n{json.dumps(payload, default=str)[:8000]}")
        return _extract_json(out)

    def act(self, task: str, tools_schema: str, history: list) -> dict:
        sys = self._act_sys(tools_schema)
        hist = "\n".join(f"[{h['role']}] {json.dumps(h['content'], default=str)[:1200]}" for h in history[-8:])
        out = self._call(sys, f"TASK: {task}\nHISTORY:\n{hist}\nNext JSON step:")
        return _extract_json(out)

    def work(self, task: str, persona: str = "") -> dict:
        from .tools import Tools  # ReAct loop: the API brain is a raw LLM -> tools.py gives it hands.
        tools = Tools()
        history: list = []
        task = (persona + "\n\nTASK: " + task) if persona else task
        for _ in range(6):
            d = self.act(task, tools.schema(), history)
            history.append({"role": "assistant", "content": d})
            if d.get("_error"):
                return {"ok": False, "result": f"brain error: {d['_error']}"}
            if d.get("action") == "final":
                return {"ok": True, "result": d.get("result", "")}
            obs = tools.call(d.get("tool"), d.get("args", {}) or {})
            history.append({"role": "tool", "content": obs})
        return {"ok": False, "result": "hit max steps without final"}


class AgentSdkBrain(Brain):
    """FAST LOCAL brain via the Claude Agent SDK (claude-agent-sdk; in-process). This is the Claude-specific impl
    behind the model-agnostic Brain interface -- the HARNESS survives a model change; only this class is
    Claude-specific (swap in a local-model Brain to drop Claude entirely). setting_sources=[] so no host-project
    hooks/settings recurse into the worker. The worker's cwd is the harness build_cwd (the target project)."""
    name = "AgentSdkBrain"

    def __init__(self, model: str = "claude-opus-4-8", timeout: int = 300, work_turns: int = 16,
                 cwd: str | None = None, domain: str = DEFAULT_DOMAIN):
        super().__init__(domain)
        import claude_agent_sdk  # noqa: F401  -- lazy import => a clear error if the SDK isn't installed
        from .config import build_cwd
        self.model = model
        self.timeout = timeout
        self.work_turns = work_turns
        # CRITICAL: the worker must write to a REAL directory. Default cwd None -> tool writes land in a void.
        self.cwd = cwd or str(build_cwd())

    @staticmethod
    async def _safety_fence_hook(input_data, tool_use_id, context):
        """MECHANICAL worker fence as a PreToolUse HOOK: DENY irreversible/destructive Bash ops (commit/push/deploy/
        rm -rf/force). Hooks work with a plain string prompt (can_use_tool needs streaming-mode input)."""
        import re as _re
        if input_data.get("tool_name") == "Bash":
            cmd = (input_data.get("tool_input") or {}).get("command", "") or ""
            DENY = _re.compile(r"git\s+(commit|push|reset\s+--hard|clean\s+-\w*f|tag\b)|--no-verify|"
                               r"\bdeploy\b|rm\s+-rf\s+[~/]|\bshutdown\b|\bgh\s+(pr|release)\b", _re.I)
            if cmd and DENY.search(cmd):
                return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                        "permissionDecisionReason": "harness worker fence: commit/push/deploy/destructive DENIED -- "
                        "stage your work; the overseer reviews + commits."}}
        return {}

    def _ask(self, prompt: str, turns: int, tools: bool, sysp: str = "") -> str:
        import asyncio
        from claude_agent_sdk import query, ClaudeAgentOptions, HookMatcher
        common = dict(model=self.model, max_turns=turns, setting_sources=[], system_prompt=(sysp or None),
                      cwd=self.cwd, env={**os.environ, "HARNESS_WORKER": "1"})
        if tools:
            # bypassPermissions (NOT acceptEdits): the worker MUST be able to Write NEW artifact files unprompted --
            # acceptEdits only auto-accepts edits to EXISTING files, so a fresh Write prompts ("Write needs
            # permission") and the artifact never lands in the build cwd (every node then refutes). The Bash
            # PreToolUse safety fence below still fires under bypassPermissions and DENIES destructive ops.
            opts = ClaudeAgentOptions(permission_mode="bypassPermissions",
                                      allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
                                      hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[self._safety_fence_hook])]},
                                      **common)
        else:
            opts = ClaudeAgentOptions(permission_mode="bypassPermissions",
                                      disallowed_tools=["Bash", "Edit", "Write", "Read", "Glob", "Grep", "Task"], **common)

        async def _run() -> str:
            chunks: list = []
            async for m in query(prompt=prompt, options=opts):
                if type(m).__name__ == "AssistantMessage":
                    for b in getattr(m, "content", []) or []:
                        t = getattr(b, "text", None)
                        if t:
                            chunks.append(t)
            return "\n".join(chunks)

        try:
            return asyncio.run(asyncio.wait_for(_run(), self.timeout))
        except Exception as e:
            return json.dumps({"_error": f"sdk: {type(e).__name__}: {str(e)[:160]}"})

    def decide(self, role: str, payload: dict, persona: str = "") -> dict:
        sysp = (persona + "\n\n" + self._decide_sys()) if persona else self._decide_sys()
        prompt = ("Output ONLY the JSON object, no preamble, no tools -- pure reasoning.\n"
                  f"ROLE: {role}\nPAYLOAD: {json.dumps(payload, default=str)[:8000]}")
        return _extract_json(self._ask(prompt, turns=1, tools=False, sysp=sysp))

    def act(self, task: str, tools_schema: str, history: list) -> dict:
        return {"action": "final", "result": "AgentSdkBrain uses work() not act()"}

    def work(self, task: str, persona: str = "") -> dict:
        sysp = ((persona + "\n\n") if persona else "") + self._work_sys()
        out = self._ask(task, turns=self.work_turns, tools=True, sysp=sysp)
        return {"ok": bool(out.strip()) and "_error" not in out[:40], "result": out.strip()[:4000]}


class CliBrain(Brain):
    """Real Claude via the headless `claude -p` CLI (uses existing auth, no API key). `claude -p` is itself a full
    tool-using agent -> for `work` we delegate the whole task; for `decide` we instruct NO tools + JSON only."""
    name = "CliBrain"

    def __init__(self, exe: str | None = None, timeout: int = 600, domain: str = DEFAULT_DOMAIN,
                 cwd: str | None = None):
        super().__init__(domain)
        from .config import build_cwd
        self.exe = exe or find_claude()
        self.timeout = timeout
        # CRITICAL: run `claude -p` IN the build cwd. Without this the worker runs in the launching process's dir
        # (the host repo root), writes artifacts to the wrong place AND walks UP the tree into the host project's
        # CLAUDE.md -> it derails onto the host project's task instead of the node's. The build cwd should be an
        # ISOLATED target dir (its own tree, no parent CLAUDE.md) so the worker stays on-task + writes where the
        # mechanical verifier checks.
        self.cwd = cwd or str(build_cwd())
        if not self.exe:
            raise RuntimeError("claude CLI not found")

    def _run(self, prompt: str, retries: int = 1) -> str:
        child_env = {**os.environ, "HARNESS_WORKER": "1"}
        for attempt in range(retries + 1):
            try:
                # --setting-sources "" : skip CLAUDE.md auto-discovery (user/project/local) so the host's global
                # ~/.claude/CLAUDE.md does NOT hijack the worker onto the host's task (it loads regardless of cwd and
                # otherwise makes `claude -p` "investigate/report per the autonomy mandate" instead of the node task).
                # CREATE_NO_WINDOW (Windows): `claude` is claude.CMD -> running it via subprocess WITHOUT this flashes
                # a console window every call (capture_output redirects stdio but does NOT suppress the window for a
                # console app). With the loop on the cli/composite backend that fires repeatedly, the flashes steal
                # focus and pause the user's typing. This flag runs it truly headless (no window, no focus steal).
                _nw = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                r = subprocess.run([self.exe, "-p", "--setting-sources", "",
                                    "--permission-mode", "bypassPermissions", prompt],
                                   capture_output=True, text=True, cwd=self.cwd, creationflags=_nw,
                                   encoding="utf-8", errors="replace", timeout=self.timeout, env=child_env)
                out = r.stdout or ""
                if out.strip():
                    return out
            except subprocess.TimeoutExpired:
                if attempt >= retries:
                    return json.dumps({"_error": f"cli timeout after {self.timeout}s"})
            except Exception as e:
                if attempt >= retries:
                    return json.dumps({"_error": f"cli: {e}"})
        return json.dumps({"_error": "cli: empty output after retries"})

    def decide(self, role: str, payload: dict, persona: str = "") -> dict:
        head = (persona + "\n\n") if persona else ""
        # claude -p reasons well but adheres loosely to the contract schema (it invents its own keys). Pin the EXACT
        # output shape per role so _extract_json gets what the graph expects (e.g. plan -> {"frontier":[...]}).
        shapes = {
            "plan": '{"frontier":[{"id":"n1","task":"<concrete step>","kind":"build|verify|diverge","ev":0.9,'
                    '"status":"open","verify_cmd":"<shell cmd that exits 0 on success, optional>"}, ...]}',
            "judge": '{"verdict":"pass|refuted|inconclusive","why":"<one line>"}',
            "reflect": '{"lesson":"<one transferable line>","adjacent":[{"id":"a1","task":"...","kind":"build",'
                       '"ev":0.5,"status":"open"}],"status":"running|solved"}',
            "replan": '{"frontier":[{"id":"n2","task":"...","kind":"build","ev":0.8,"status":"open"}]}',
        }
        shape = shapes.get(role)
        prompt = (head + self._decide_sys()
                  + f"\n\nYour ROLE this turn is '{role}'. The INPUT to reason over is below -- this IS your task. "
                  "Do NOT say a task/objective is missing, do NOT inspect the environment or use any tools -- just "
                  "reason over the INPUT and output ONLY the single JSON object for this role (no preamble, no "
                  "markdown fences, no invented keys)."
                  + (f"\nOutput EXACTLY this shape: {shape}" if shape else "")
                  + f"\nINPUT ({role}):\n{json.dumps(payload, default=str)[:8000]}")
        return _extract_json(self._run(prompt))

    def act(self, task: str, tools_schema: str, history: list) -> dict:
        return {"action": "final", "result": "CliBrain uses work() not act()"}

    def work(self, task: str, persona: str = "") -> dict:
        head = (persona + "\n\n") if persona else ""
        # CLEAN TASK-FIRST prompt: `claude -p` is itself a full agent WITH its own system prompt, so prepending the
        # harness _work_sys() as -p TEXT makes it "inspect the environment to find the task" instead of doing it
        # (verified: the _work_sys preamble dominated and claude went looking for a task). A direct, task-first
        # build-imperative prompt is what made a real `claude -p` reliably Write the artifact. (decide() keeps the
        # _decide_sys framing because it wants pure-reasoning JSON, no tools -- different need.)
        prompt = (head + "TASK: " + task + "\n\n"
                  "Complete THIS task now in the current working directory. Use the Write/Edit tools to CREATE the "
                  "required file(s) as real artifacts on disk, run commands to verify them, and fix until they work. "
                  "Do NOT inspect for some other task, do NOT just analyze or report -- the TASK above is the entire "
                  "job. Stop once the artifact exists and works.")
        out = self._run(prompt)
        return {"ok": bool(out.strip()) and "_error" not in out[:40], "result": out.strip()[:4000]}


class PersistentCliBrain(CliBrain):
    """CliBrain that keeps ONE claude session ALIVE across graph nodes, so context CARRIES (no per-node cold-start).

    node-1 runs `claude -p PROMPT --output-format json` with NO --resume (a fresh session); we parse the returned
    `session_id` and store it. Every later node passes `--resume <session_id>` so the SAME conversation (and its
    accumulated context) continues. A 'rebirth' (drop the stored session_id -> the next call starts a brand-new
    session) is triggered ONLY on a context-limit / error response. EVERY call is wrapped in try/except that FALLS
    BACK to the plain-text CliBrain path on ANY failure, so the loop can never break.

    CliBrain is left byte-for-byte intact -- this subclass only overrides _run (the CLI invocation); decide/act/work
    (prompt construction + result handling) are INHERITED unchanged, so the persistent session is transparent to
    callers: _run still returns the model's inner `result` TEXT, exactly as plain CliBrain._run does. The domain seam
    is inherited from CliBrain/Brain, so the persistent session is also project-agnostic ({domain}-templated)."""
    name = "PersistentCliBrain"

    def __init__(self, exe: str | None = None, timeout: int = 600, domain: str = DEFAULT_DOMAIN,
                 cwd: str | None = None):
        super().__init__(exe, timeout, domain, cwd=cwd)
        self.session_id: str | None = None  # carried across nodes; None => next call is fresh (node-1 / post-rebirth)
        self.rebirths = 0                    # observability: how many times context was reset
        # B1 (2026-06-08): the graph shares ONE brain across plan/dispatch/judge and dispatch runs work() CONCURRENTLY
        # (parallel>1). session_id is then read+written by multiple threads, and two workers could --resume the SAME
        # session at once (claude -p cannot safely resume one session from 2 procs). This lock SERIALIZES the whole
        # claude -p call per brain instance so a session is resumed by at most one node at a time -> the documented
        # race is CONTAINED. (Context still carries node->node as designed; only true concurrency is serialized.)
        import threading
        self._session_lock = threading.Lock()

    @staticmethod
    def _is_error_envelope(env: dict) -> bool:
        """A context-limit OR error response (the ONLY rebirth trigger). `claude -p` surfaces context-window
        overflow as an API error (is_error / api_error_status), and turn/exec failures as a non-success subtype."""
        if not isinstance(env, dict):
            return True
        if env.get("is_error") or env.get("api_error_status"):
            return True
        return env.get("subtype") not in (None, "success")

    def _rebirth(self) -> None:
        """Drop the carried session so the NEXT _run omits --resume and starts a brand-new (cold) session."""
        if self.session_id is not None:
            self.rebirths += 1
        self.session_id = None

    def _run(self, prompt: str, retries: int = 1) -> str:
        """Persistent-session variant of CliBrain._run. Returns the model's inner `result` TEXT (same contract as
        CliBrain._run -> decide() feeds it to _extract_json, work() scans it). On ANY failure -> super()._run, i.e.
        the plain-text CliBrain path, so the loop never breaks.

        B1: the entire session-id read -> claude -p call -> session-id write is serialized under self._session_lock,
        so concurrent dispatch workers can never both --resume the same session at once (last-writer-wins race) and
        no session is silently orphaned. The lock is per-brain-instance; distinct brains/threads are unaffected."""
        with self._session_lock:
            return self._run_locked(prompt, retries)

    def _run_locked(self, prompt: str, retries: int = 1) -> str:
        child_env = {**os.environ, "HARNESS_WORKER": "1"}
        used_resume = self.session_id is not None
        try:
            cmd = [self.exe, "-p", prompt, "--output-format", "json"]
            if self.session_id:  # node-1 / post-rebirth has none -> a fresh session is created and its id parsed below
                cmd += ["--resume", self.session_id]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=self.timeout, env=child_env,
                               creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            out = (r.stdout or "").strip()
            if not out:
                raise RuntimeError("empty json envelope")
            env = json.loads(out)  # non-JSON stdout -> JSONDecodeError -> caught below -> CliBrain fallback
            if self._is_error_envelope(env):
                self._rebirth()  # context-limit/error -> reset; serve THIS node via the plain-text CliBrain path
                return super()._run(prompt, retries)
            sid = env.get("session_id")
            if sid:
                self.session_id = sid  # carry the live session forward to the next node
            text = env.get("result", "") or ""
            if not text.strip():
                raise RuntimeError("empty result text in json envelope")
            return text
        except Exception:
            # ANY failure (timeout, non-JSON stdout, subprocess error, ...) -> never break the loop.
            # If we were resuming, the session is suspect -> rebirth so we don't get stuck re-resuming a dead session.
            if used_resume:
                self._rebirth()
            return super()._run(prompt, retries)


def _run_react_work(brain, task: str, persona: str, work_turns: int, cwd: str | None) -> dict:
    """The ReAct worker loop shared by the raw-LLM brains (Ollama / LiteLLM): drive act() over tools.py until the
    model emits {action:final}. ROBUSTNESS for weak / local models (the spin found a 7B writing CORRECT code then
    running out of turns mid self-debug, reported as 'hit max steps' with worker_ok=False):
      (a) GRACEFUL FINALIZATION -- on the LAST allowed turn, forbid more tools and FORCE a final, so a long
          write->test->fix->test loop never throws completed work away;
      (b) DE-FACTO RESULT -- track the last SUCCESSFUL tool output and return it (ok=True) if the model never says
          'final', instead of a bare failure. The mechanical verifier (judge) remains the real arbiter of CORRECTNESS;
          this only fixes the worker's self-report + stops wasting a built artifact."""
    from .tools import Tools
    tools = Tools(cwd=cwd)
    history: list = []
    task = (persona + "\n\nTASK: " + task) if persona else task
    last_obs = ""
    n = max(1, int(work_turns))
    for i in range(n):
        step_task = task if i < n - 1 else (
            task + "\n\nFINAL STEP: do NOT call any more tools. Reply with "
            '{"action":"final","result":"<one-line summary of what you produced>"} now.')
        d = brain.act(step_task, tools.schema(), history)
        history.append({"role": "assistant", "content": d})
        if d.get("_error"):
            return {"ok": False, "result": f"brain error: {d['_error']}"}
        if d.get("action") == "final":
            return {"ok": True, "result": str(d.get("result", "") or last_obs)}
        obs = tools.call(d.get("tool"), d.get("args", {}) or {})
        history.append({"role": "tool", "content": obs})
        if obs.get("ok"):
            last_obs = str(obs.get("output", ""))[:500]
    # exhausted even after the forced final -> if real work happened, report it (don't bury a built artifact).
    return {"ok": bool(last_obs), "result": last_obs or "hit max steps without final"}


class OllamaBrain(Brain):
    """LOCAL OPEN-SOURCE model via Ollama's HTTP API -- the proof the harness survives Claude leaving. Pure stdlib
    (urllib), no extra deps and no API key: a local model (e.g. qwen2.5-coder:7b) drives the SAME Brain interface.
    A raw local LLM has no hands -> work() runs the same ReAct loop as AnthropicBrain over tools.py. Local models are
    WEAKER than Claude (smaller context, looser JSON) -> we ask for json explicitly + lean on _extract_json, and we
    NEVER crash the loop: any HTTP/parse error returns an _error dict so the graph keeps routing."""
    name = "OllamaBrain"

    def __init__(self, model: str | None = None, host: str | None = None, timeout: int = 300,
                 work_turns: int = 12, cwd: str | None = None, domain: str = DEFAULT_DOMAIN):
        super().__init__(domain)
        from .config import build_cwd
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.work_turns = work_turns
        # the ReAct worker must write artifacts to the SAME dir the mechanical verifier runs in (the build cwd),
        # else the model's write_file lands in repo root and the verifier reports 'no such file'.
        self.cwd = cwd or str(build_cwd())

    def _call(self, system: str, user: str, max_tokens: int = 2000) -> str:
        """One /api/chat call (stream=false). Returns the assistant text, or a JSON-encoded _error (never raises)."""
        import urllib.request
        import urllib.error
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }).encode("utf-8")
        req = urllib.request.Request(self.host + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return (data.get("message") or {}).get("content", "") or ""
        except urllib.error.URLError as e:
            return json.dumps({"_error": f"ollama url: {getattr(e, 'reason', e)}"})
        except Exception as e:
            return json.dumps({"_error": f"ollama: {type(e).__name__}: {str(e)[:160]}"})

    def decide(self, role: str, payload: dict, persona: str = "") -> dict:
        sysp = (persona + "\n\n" + self._decide_sys()) if persona else self._decide_sys()
        out = self._call(sysp, f"ROLE: {role}\nPAYLOAD:\n{json.dumps(payload, default=str)[:8000]}\n"
                               "Respond with ONLY the JSON object for this role.")
        return _extract_json(out)

    def act(self, task: str, tools_schema: str, history: list) -> dict:
        sys = self._act_sys(tools_schema)
        hist = "\n".join(f"[{h['role']}] {json.dumps(h['content'], default=str)[:1200]}" for h in history[-8:])
        out = self._call(sys, f"TASK: {task}\nHISTORY:\n{hist}\nNext JSON step:")
        return _extract_json(out)

    def work(self, task: str, persona: str = "") -> dict:
        # ReAct loop over tools.py (a raw local LLM is just text). Shared with LiteLLMBrain; robust to a weak model's
        # long self-debug loop (forces a final on the last turn; reports a built artifact instead of "hit max steps").
        return _run_react_work(self, task, persona, self.work_turns, self.cwd)


# --------------------------------------------------------------------------- LiteLLM (unified gateway)
# Default fallback chain a LiteLLMBrain walks when its primary model errors: try the configured primary, then a
# LOCAL ollama model (works offline, no key), then -- if every litellm provider is down -- a deterministic MockBrain
# so the graph NEVER stalls. Override per-instance via model_fallbacks=, or globally via env (LITELLM_MODEL /
# LITELLM_FALLBACK_MODEL). Each entry is a dict litellm.completion(**entry, messages=...) understands; the sentinel
# {"_mock": True} means "give up on the LLMs and degrade to MockBrain".
LITELLM_DEFAULT_OLLAMA = {"model": "ollama/qwen2.5-coder:7b", "api_base": "http://localhost:11434"}
LITELLM_MOCK_SENTINEL = {"_mock": True}


def _litellm_available() -> bool:
    """True iff litellm is importable. Kept cheap (import only) so make_brain can route conditionally."""
    try:
        import litellm  # noqa: F401
        return True
    except Exception:
        return False


class LiteLLMBrain(Brain):
    """Brain routed through LiteLLM -- the battle-tested UNIFIED gateway (one API for Anthropic/OpenAI/Ollama/...).

    This is the swappable, installable seam: point `model` at ANY litellm-supported string and the same Brain
    interface (decide/act/work) works -- 'anthropic/claude-...', 'ollama/qwen2.5-coder:7b' (with
    api_base=http://localhost:11434), 'openai/gpt-...', etc. A raw LLM has no hands -> work() runs the SAME ReAct
    loop over tools.py that OllamaBrain/AnthropicBrain use. It NEVER raises into the graph: any error becomes an
    _error dict (decide/act) or {"ok": False, ...} (work).

    AUTO-FALLBACK: `model_fallbacks` is an ordered list of completion-kwarg dicts tried in turn when the one before
    errors; the final {"_mock": True} sentinel degrades to a MockBrain so a provider outage can never stall the loop.
    The chain is a REAL try-chain at the _call level (more robust than litellm's own `fallbacks=` because it can also
    fall THROUGH litellm entirely to MockBrain)."""
    name = "LiteLLMBrain"

    def __init__(self, model: str | None = None, model_fallbacks: list | None = None, api_base: str | None = None,
                 max_retries: int = 2, work_turns: int = 12, cwd: str | None = None, timeout: int = 300,
                 domain: str = DEFAULT_DOMAIN):
        super().__init__(domain)
        import litellm  # lazy -> a clear ImportError if litellm isn't installed (make_brain guards on availability)
        self._litellm = litellm
        # quiet litellm's chatty logging + tolerate provider-specific params it can't pass through
        try:
            litellm.drop_params = True
            litellm.suppress_debug_info = True
        except Exception:
            pass
        self.model = model or os.environ.get("LITELLM_MODEL") or LITELLM_DEFAULT_OLLAMA["model"]
        self.api_base = api_base or os.environ.get("LITELLM_API_BASE")
        # ollama models need an api_base; default to the local server unless one was supplied.
        if self.api_base is None and self.model.startswith("ollama/"):
            self.api_base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.max_retries = max_retries
        self.work_turns = work_turns
        self.timeout = timeout
        from .config import build_cwd
        self.cwd = cwd or str(build_cwd())
        self.model_fallbacks = model_fallbacks if model_fallbacks is not None else self._default_fallbacks()
        self._mock: MockBrain | None = None  # lazily built only if the whole LLM chain fails

    def _primary_kwargs(self) -> dict:
        d = {"model": self.model}
        if self.api_base:
            d["api_base"] = self.api_base
        return d

    def _default_fallbacks(self) -> list:
        """primary -> (a local ollama model, unless the primary already IS that ollama model) -> MockBrain sentinel."""
        chain = [self._primary_kwargs()]
        fb = os.environ.get("LITELLM_FALLBACK_MODEL")
        ollama = {"model": fb, "api_base": os.environ.get("OLLAMA_HOST", "http://localhost:11434")} if fb \
            else dict(LITELLM_DEFAULT_OLLAMA)
        if ollama["model"] != self.model:
            chain.append(ollama)
        chain.append(dict(LITELLM_MOCK_SENTINEL))
        return chain

    def _mock_brain(self) -> "MockBrain":
        if self._mock is None:
            self._mock = MockBrain(self.domain)
        return self._mock

    def _completion(self, kwargs: dict, system: str, user: str, max_tokens: int) -> str:
        """One litellm.completion call. Returns the assistant text, or a JSON-encoded _error (never raises)."""
        call = {k: v for k, v in kwargs.items() if not (k == "api_base" and v is None)}  # drop a null api_base
        try:
            resp = self._litellm.completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                stream=False, temperature=0.2, max_tokens=max_tokens, timeout=self.timeout, num_retries=self.max_retries,
                **call)
            return (resp.choices[0].message.content or "") if getattr(resp, "choices", None) else ""
        except Exception as e:
            return json.dumps({"_error": f"litellm: {type(e).__name__}: {str(e)[:160]}"})

    def _call(self, system: str, user: str, max_tokens: int = 2000) -> str:
        """Walk the fallback chain: return the FIRST non-empty, non-_error completion. The {"_mock": True} sentinel
        returns a marker the callers translate into MockBrain output. If even that is exhausted -> a final _error."""
        last = json.dumps({"_error": "litellm: empty fallback chain"})
        for kw in self.model_fallbacks:
            if kw.get("_mock"):
                return "__MOCK__"  # callers detect this and route to the MockBrain (graceful final degrade)
            out = self._completion(kw, system, user, max_tokens)
            if out and '"_error"' not in out[:40]:
                return out
            last = out or last
        return last

    def decide(self, role: str, payload: dict, persona: str = "") -> dict:
        sysp = (persona + "\n\n" + self._decide_sys()) if persona else self._decide_sys()
        out = self._call(sysp, f"ROLE: {role}\nPAYLOAD:\n{json.dumps(payload, default=str)[:8000]}\n"
                               "Respond with ONLY the JSON object for this role.")
        if out == "__MOCK__":
            return self._mock_brain().decide(role, payload, persona)
        return _extract_json(out)

    def act(self, task: str, tools_schema: str, history: list) -> dict:
        sys = self._act_sys(tools_schema)
        hist = "\n".join(f"[{h['role']}] {json.dumps(h['content'], default=str)[:1200]}" for h in history[-8:])
        out = self._call(sys, f"TASK: {task}\nHISTORY:\n{hist}\nNext JSON step:")
        if out == "__MOCK__":
            return self._mock_brain().act(task, tools_schema, history)
        return _extract_json(out)

    def work(self, task: str, persona: str = "") -> dict:
        # ReAct loop over tools.py (a raw LLM is just text). Shared helper -- robust to a weak model's long self-debug
        # loop (forces a final on the last turn; reports a built artifact instead of "hit max steps without final").
        return _run_react_work(self, task, persona, self.work_turns, self.cwd)


class MockBrain(Brain):
    """Deterministic + role/task-aware. Proves the FULL machinery (incl. real tool work) with no credentials."""
    name = "MockBrain"

    def decide(self, role: str, payload: dict, persona: str = "") -> dict:
        if role == "plan":
            obj = payload.get("objective", "objective")
            # n+-k decomposition: a primary BUILD node (n), a FALSIFIER (-k, kind=verify), and a GENERALIZATION
            # (+k, kind=diverge). The deterministic mock emits all three so the breadth contract is mechanically
            # testable (a real brain follows the _PLAN_INSTRUCTION above to do the same).
            return {"frontier": [
                {"id": "n1", "task": f"build: {obj} -- run a real check (inspect the working directory)",
                 "ev": 0.9, "kind": "build", "status": "open"},
                {"id": "n2", "task": f"falsifier (-k): is the approach/data/spec for '{obj}' sound? "
                                     "audit soundness / run a positive control",
                 "ev": 0.8, "kind": "verify", "status": "open"},
                {"id": "n3", "task": f"generalization (+k): does the solution to '{obj}' extend to the "
                                     "adjacent / more-general case?",
                 "ev": 0.6, "kind": "diverge", "status": "open"},
            ]}
        if role == "plan_critique":
            # Deterministic critique: add ONLY the named-missing breadth guard(s). The graph calls this only when a
            # gap exists, so honoring PAYLOAD.missing keeps it minimal (never pads).
            obj = payload.get("objective", "objective")
            missing = payload.get("missing", []) or []
            add = []
            if "falsifier" in missing:
                add.append({"id": "crit_falsifier", "task": f"falsifier (-k): audit whether the approach for "
                            f"'{obj}' is sound (positive control / leak check)", "ev": 0.7, "kind": "verify",
                            "status": "open"})
            if "generalization" in missing:
                add.append({"id": "crit_generalization", "task": f"generalization (+k): extend the solution for "
                            f"'{obj}' to the adjacent / general case", "ev": 0.55, "kind": "diverge",
                            "status": "open"})
            return {"add": add}
        if role == "judge":
            n = payload.get("node", {})
            ok = bool(n.get("result")) and "DENIED" not in str(n.get("result"))
            return {"verdict": "pass" if ok else "inconclusive",
                    "reason": "tool evidence present" if ok else "no verified evidence"}
        if role == "reflect":
            cyc = payload.get("cycle", 1)
            adj = [] if cyc >= payload.get("taper", 2) else [
                {"id": f"a{cyc}", "task": f"adjacent problem surfaced in cycle {cyc}",
                 "ev": 0.55, "kind": "build", "status": "open"}]
            return {"lesson": f"cycle {cyc}: node executed + judged via real tools.", "adjacent": adj}
        if role == "replan":
            # DETERMINISTIC revised frontier (offline tests): PRUNE refuted nodes, KEEP open/non-refuted ones, ADD a
            # single NEW-APPROACH node. The added id is unique (suffixed with the running replan_count) so repeated
            # replans don't collide -- and it is a genuinely new node id (not present in the current frontier).
            cur = payload.get("current_frontier", []) or []
            kept = [{"id": n.get("id"), "task": n.get("task", ""), "ev": n.get("ev", 0.5),
                     "kind": n.get("kind", "build"), "status": "open"}
                    for n in cur if isinstance(n, dict) and n.get("status") not in ("refuted", "done")]
            seen = {n.get("id") for n in cur if isinstance(n, dict)}
            i = 0
            new_id = "replan_alt1"
            while new_id in seen or any(k["id"] == new_id for k in kept):
                i += 1
                new_id = f"replan_alt{i + 1}"
            new_node = {"id": new_id,
                        "task": f"NEW-APPROACH (replan): attack '{payload.get('objective', '')[:60]}' a different way",
                        "ev": 0.7, "kind": "build", "status": "open"}
            return {"frontier": kept + [new_node]}
        return {}

    def act(self, task: str, tools_schema: str, history: list) -> dict:
        steps = [h for h in history if h.get("role") == "tool"]
        if not steps:
            return {"thought": "inspect state with a real tool", "action": "tool",
                    "tool": "run_shell", "args": {"command": "python --version"}}
        last = steps[-1]["content"]
        return {"thought": "tool ran; finalize with the evidence", "action": "final",
                "result": f"executed task '{task[:50]}' -- tool output: {str(last)[:160]}"}

    def work(self, task: str, persona: str = "") -> dict:
        from .tools import Tools
        r = Tools().run_shell("python --version")
        tag = "[mock+expert]" if persona else "[mock]"
        return {"ok": bool(r.get("ok")), "result": f"{tag} did '{task[:40]}' -> {r.get('output', '')}"}


class CompositeBrain(Brain):
    """The no-key FULL-LOOP brain: decide() via a strong STRUCTURED reasoner (AgentSdkBrain -> clean contract JSON,
    e.g. a real plan frontier) + work() via a file-writing agent (CliBrain -> real artifacts in the build cwd).
    Rationale (RWYB 2026-06-13): headless, each capable no-key backend does exactly HALF the brain contract well --
    AgentSdkBrain.decide returns a proper frontier but its work() executes no tools; CliBrain.work writes real files
    but its decide() won't emit the schema. Composing them gives a loop that both PLANS and BUILDS with no API key."""
    name = "CompositeBrain"

    def __init__(self, domain: str = DEFAULT_DOMAIN, cwd: str | None = None):
        super().__init__(domain)
        self.decider = AgentSdkBrain(domain=domain, cwd=cwd)
        self.worker = CliBrain(domain=domain, cwd=cwd)
        self.cwd = self.worker.cwd

    def set_plan_instruction(self, instruction):  # champion install -> the DECIDER owns planning
        super().set_plan_instruction(instruction)
        try:
            self.decider.set_plan_instruction(instruction)
        except Exception:
            pass

    def decide(self, role: str, payload: dict, persona: str = "") -> dict:
        return self.decider.decide(role, payload, persona)

    def act(self, task: str, tools_schema: str, history: list) -> dict:
        return self.worker.act(task, tools_schema, history)

    def work(self, task: str, persona: str = "") -> dict:
        return self.worker.work(task, persona)


def make_brain(kind: str = "auto", domain: str = DEFAULT_DOMAIN, cwd: str | None = None,
               model: str | None = None) -> Brain:
    """Select a backend. 'auto' prefers a real Claude brain when available, else the deterministic MockBrain.
    Pass an explicit kind (mock|sdk|api|cli|ollama|litellm) to force one. 'ollama' = a LOCAL open-source model (model
    portability: the harness runs with NO Claude). 'litellm' = the UNIFIED gateway (one API for any provider;
    auto-fallback to local ollama then MockBrain). Neither is part of 'auto' -- you opt in explicitly because they
    need a running local server / pulled model (ollama) or an installed litellm. `cwd` (= the build cwd) is threaded
    to the brains whose ReAct worker writes artifacts, so they land where the mechanical verifier runs. `model` (opt)
    overrides the model string for litellm/ollama backends.

    BACKWARD COMPAT: when litellm is installed, 'ollama' and 'api' route through LiteLLMBrain (the robust gateway);
    when litellm is ABSENT, they fall back to the hand-rolled OllamaBrain / AnthropicBrain -- i.e. litellm-absent ==
    exactly the pre-litellm behavior."""
    if kind == "composite":  # no-key full loop: sdk decides (clean schema) + cli works (writes artifacts)
        try:
            return CompositeBrain(domain=domain, cwd=cwd)
        except Exception:
            pass
    if kind == "cascade":  # cheap->strong escalation router (cheap defaults to local ollama; degrades gracefully)
        from .cascade_brain import make_cascade
        return make_cascade(cheap_kind=os.environ.get("CASCADE_CHEAP", "ollama"),
                            strong_kind=os.environ.get("CASCADE_STRONG") or None, domain=domain, cwd=cwd,
                            cheap_model=model)
    if kind == "litellm":  # the UNIFIED gateway: any provider, auto-fallback to local ollama -> MockBrain
        if _litellm_available():
            try:
                return LiteLLMBrain(model=model, domain=domain, cwd=cwd)
            except Exception:
                pass  # construction failed -> degrade to the local hand-rolled ollama brain, then MockBrain
        return OllamaBrain(model=model, domain=domain, cwd=cwd)
    if kind == "ollama":  # LOCAL open-source model -- proof the harness outlives Claude
        if _litellm_available():  # route through the robust gateway when present...
            try:
                m = model or "ollama/" + (os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b"))
                return LiteLLMBrain(model=m, domain=domain, cwd=cwd)
            except Exception:
                pass
        return OllamaBrain(model=model, domain=domain, cwd=cwd)  # ...else the hand-rolled brain (litellm-absent compat)
    if kind in ("sdk", "auto"):  # PREFERRED when present: fast in-process Claude (no API key)
        try:
            return AgentSdkBrain(domain=domain, cwd=cwd)  # thread cwd -> the worker writes artifacts to the build dir
        except Exception:
            pass
    if kind in ("api", "auto") and os.environ.get("ANTHROPIC_API_KEY"):
        if _litellm_available():  # robust gateway when present; AnthropicBrain remains the litellm-absent fallback
            try:
                m = model or os.environ.get("LITELLM_MODEL") or "anthropic/" + os.environ.get(
                    "HARNESS_MODEL", "claude-opus-4-8")
                return LiteLLMBrain(model=m, domain=domain, cwd=cwd)
            except Exception:
                pass
        try:
            return AnthropicBrain(domain=domain)
        except Exception:
            pass
    if kind == "persistent":  # opt-in only (never auto): ONE claude session carried across nodes (context persists)
        exe = find_claude()
        if exe:
            try:
                return PersistentCliBrain(exe, domain=domain, cwd=cwd)
            except Exception:
                pass  # construction failed -> fall through to plain CliBrain, then MockBrain
    if kind in ("cli", "auto", "persistent"):
        exe = find_claude()
        if exe:
            try:
                return CliBrain(exe, domain=domain, cwd=cwd)
            except Exception:
                pass
    return MockBrain(domain=domain)


if __name__ == "__main__":
    b = make_brain("mock")
    print("brain:", b.name)
    print("plan :", b.decide("plan", {"objective": "build a sorter"}))
    print("act  :", b.act("inspect state", "run_shell,...", []))
