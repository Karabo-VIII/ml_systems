---
name: research-web
description: Gather external facts before building -- use the harness's web_search and fetch_url tools to verify an API signature, a library usage, a spec, or a current fact instead of guessing. Use whenever the task depends on knowledge you are not certain of.
---
# Look it up instead of guessing

The harness ships PUBLIC-web tools (`web_search`, `fetch_url`) so a model WITHOUT native web access can still
ground itself in real facts. A guessed API signature or a hallucinated flag wastes a whole build cycle. When
the task hinges on an external fact you are not sure of, search first.

## Steps
1. **Name the uncertainty** in one line -- "what is the exact signature of X?", "does library Y support Z?",
   "what is the current value/spec of W?".
2. **`web_search(query)`** -- get candidate sources (title + url + snippet). With `BRAVE_API_KEY` set you get
   general web results; without it you get entity abstracts -- enough to find a page to read.
3. **`fetch_url(url)`** -- read the most relevant page (HTML stripped). Pull the EXACT signature / flag / value.
4. **Use it in the build**, and cite where it came from in your reasoning so the fact is traceable.

## When to use it / when not
- USE when: an external API/library/spec/current-fact is load-bearing and you are not certain.
- SKIP when: the answer is in the repo (use `read_file` / `run_shell` grep first -- local truth beats the web),
  or you already know it cold. Do not search to look busy.

## Guard rails
- The tools are SSRF-guarded (no localhost / private / internal hosts) and hit the public web only.
- The web can be wrong or stale -- prefer official docs; corroborate a surprising claim with a second source.
- A fact you cannot fetch is still unknown -- say so; never upgrade a guess to a fact because a search failed.

## Why
- Grounding a single load-bearing fact up front prevents a full build-on-a-false-premise cycle.
- It gives a no-native-web model the same "verify the claim" reflex Claude uses by default.
