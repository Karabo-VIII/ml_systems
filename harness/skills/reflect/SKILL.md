---
name: reflect
description: After a failure or a refuted attempt, write one verbal post-mortem line -- why it failed + what to try differently -- and carry it into the next attempt so the same mistake is not repeated. Use after any failed verify, refuted node, or dead end (Reflexion).
---
# Turn a failure into a directed retry

A failure is only wasted if you forget it. Before retrying, write down WHY it failed and WHAT you will change.
Re-read that note at the start of the next attempt. This is the difference between blind retry (repeat the
mistake) and a directed retry (a real next move). This is `debug-failure`'s learning half: debug fixes THIS
bug; reflect prevents the NEXT one.

## Steps
1. **State what failed** in one line -- the action + the observed error/verdict, concretely (the traceback, the
   refuting number, the dead end).
2. **Diagnose the cause** in one line -- not "it didn't work" but the actual reason ("assumed the API returned a
   list, it returns a dict"; "the null wasn't cost-matched so the edge was an artifact").
3. **State the change** in one line -- the specific different thing to do next ("read the dict keys first";
   "re-run with the cost-matched null").
4. **Record it** -- append the three lines to your working notes / the run's learnings lane / episodic memory,
   so it survives into the next attempt and the next session.
5. **Re-read before retrying** -- start the next attempt by reading the last reflection. A note you never re-read
   is not learning.

## Format
```
FAILED:   <what action, what error/verdict>
BECAUSE:  <the actual cause>
NEXT:     <the specific different move>
```

## When to use
- After any failed `verify_cmd` you are about to retry.
- After a refuted hypothesis / node (write why it was refuted so it is not re-mined).
- At the end of a run that fell short -- one reflection on the run itself.

## Why
- A verbal post-mortem converts a dead end into a pointer toward the live path (Reflexion).
- Recorded reflections compound across attempts and sessions -- the durable note IS the agent getting smarter;
  never re-pay for a lesson already learned.
