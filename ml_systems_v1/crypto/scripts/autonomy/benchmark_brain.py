"""benchmark_brain.py -- apples-to-apples benchmark of ollama coding brains.

For each model, on each verifiable Python task: query ollama (temp 0, deterministic), extract the function,
run it against hidden unit tests in a windowless subprocess, score pass@1 + wall latency + tokens/s.
Honest yardstick (not "beats random"): real held-out coding correctness, the metric a CODER brain is for.

Usage:
    python scripts/autonomy/benchmark_brain.py --models qwen2.5-coder:7b "hf.co/.../...:Q4_K_M"
    python scripts/autonomy/benchmark_brain.py            # defaults: qwen2.5-coder:7b + gemma4 + qwen-3b if present
Output: a markdown table to stdout + a JSON at runs/autonomy/brain_benchmark_<n>tasks.json
"""
import argparse, json, os, re, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLLAMA = "http://127.0.0.1:11434/api/generate"
NW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# -- verifiable tasks: (name, spec-prompt, function-name, hidden test asserting behavior incl. edge cases) --
TASKS = [
    ("is_palindrome",
     "a function is_palindrome(s: str) -> bool that returns True if s is a palindrome considering only "
     "alphanumeric characters and ignoring case.",
     "is_palindrome",
     "assert is_palindrome('A man, a plan, a canal: Panama') is True\n"
     "assert is_palindrome('race a car') is False\n"
     "assert is_palindrome('') is True\n"
     "assert is_palindrome('.,') is True\n"),
    ("two_sum",
     "a function two_sum(nums: list[int], target: int) -> list[int] that returns the indices of the two "
     "numbers that add up to target (exactly one solution; do not use the same element twice).",
     "two_sum",
     "assert sorted(two_sum([2,7,11,15], 9)) == [0,1]\n"
     "assert sorted(two_sum([3,2,4], 6)) == [1,2]\n"
     "assert sorted(two_sum([3,3], 6)) == [0,1]\n"),
    ("is_prime",
     "a function is_prime(n: int) -> bool that returns True iff n is a prime number.",
     "is_prime",
     "assert is_prime(2) is True\nassert is_prime(1) is False\nassert is_prime(0) is False\n"
     "assert is_prime(-7) is False\nassert is_prime(97) is True\nassert is_prime(100) is False\n"),
    ("roman_to_int",
     "a function roman_to_int(s: str) -> int that converts a Roman numeral string to its integer value.",
     "roman_to_int",
     "assert roman_to_int('III') == 3\nassert roman_to_int('IV') == 4\nassert roman_to_int('IX') == 9\n"
     "assert roman_to_int('LVIII') == 58\nassert roman_to_int('MCMXCIV') == 1994\n"),
    ("valid_parentheses",
     "a function valid_parentheses(s: str) -> bool that returns True iff the brackets in s (), [], {} are "
     "correctly matched and nested.",
     "valid_parentheses",
     "assert valid_parentheses('()') is True\nassert valid_parentheses('()[]{}') is True\n"
     "assert valid_parentheses('(]') is False\nassert valid_parentheses('([)]') is False\n"
     "assert valid_parentheses('{[]}') is True\nassert valid_parentheses('') is True\n"),
    ("max_subarray",
     "a function max_subarray(nums: list[int]) -> int that returns the largest sum of any contiguous "
     "non-empty subarray (Kadane's algorithm).",
     "max_subarray",
     "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6\nassert max_subarray([1]) == 1\n"
     "assert max_subarray([5,4,-1,7,8]) == 23\nassert max_subarray([-3,-1,-2]) == -1\n"),
    ("longest_common_prefix",
     "a function longest_common_prefix(strs: list[str]) -> str that returns the longest common prefix among "
     "all strings (empty string if none).",
     "longest_common_prefix",
     "assert longest_common_prefix(['flower','flow','flight']) == 'fl'\n"
     "assert longest_common_prefix(['dog','racecar','car']) == ''\n"
     "assert longest_common_prefix(['a']) == 'a'\nassert longest_common_prefix([]) == ''\n"),
    ("merge_intervals",
     "a function merge_intervals(intervals: list[list[int]]) -> list[list[int]] that merges all overlapping "
     "intervals and returns them sorted by start.",
     "merge_intervals",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
     "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
     "assert merge_intervals([[1,4]]) == [[1,4]]\n"),
    ("group_anagrams",
     "a function group_anagrams(strs: list[str]) -> list[list[str]] that groups words that are anagrams of "
     "each other. The order of groups and of words within a group does not matter.",
     "group_anagrams",
     "r = group_anagrams(['eat','tea','tan','ate','nat','bat'])\n"
     "norm = sorted(sorted(g) for g in r)\n"
     "assert norm == sorted(sorted(g) for g in [['ate','eat','tea'],['bat'],['nat','tan']])\n"),
    ("edit_distance",
     "a function edit_distance(a: str, b: str) -> int that returns the Levenshtein edit distance between a "
     "and b (min single-character insertions, deletions, or substitutions to turn a into b).",
     "edit_distance",
     "assert edit_distance('horse','ros') == 3\nassert edit_distance('intention','execution') == 5\n"
     "assert edit_distance('','abc') == 3\nassert edit_distance('same','same') == 0\n"),
    # --- harder tier (DP / graph / hard-string with edge cases) -- discriminates a strong 12B from a 7B ---
    ("coin_change",
     "a function coin_change(coins: list[int], amount: int) -> int that returns the minimum number of coins "
     "needed to make up amount, or -1 if it cannot be made. amount 0 needs 0 coins.",
     "coin_change",
     "assert coin_change([1,2,5],11) == 3\nassert coin_change([2],3) == -1\n"
     "assert coin_change([1],0) == 0\nassert coin_change([186,419,83,408],6249) == 20\n"),
    ("length_of_longest_substring",
     "a function length_of_longest_substring(s: str) -> int that returns the length of the longest substring "
     "of s without repeating characters.",
     "length_of_longest_substring",
     "assert length_of_longest_substring('abcabcbb') == 3\nassert length_of_longest_substring('bbbbb') == 1\n"
     "assert length_of_longest_substring('pwwkew') == 3\nassert length_of_longest_substring('') == 0\n"),
    ("longest_increasing_subsequence",
     "a function longest_increasing_subsequence(nums: list[int]) -> int that returns the length of the "
     "longest strictly increasing subsequence.",
     "longest_increasing_subsequence",
     "assert longest_increasing_subsequence([10,9,2,5,3,7,101,18]) == 4\n"
     "assert longest_increasing_subsequence([0,1,0,3,2,3]) == 4\n"
     "assert longest_increasing_subsequence([7,7,7,7]) == 1\nassert longest_increasing_subsequence([]) == 0\n"),
    ("word_break",
     "a function word_break(s: str, word_dict: list[str]) -> bool that returns True iff s can be segmented "
     "into a space-separated sequence of one or more words from word_dict (each word reusable).",
     "word_break",
     "assert word_break('leetcode',['leet','code']) is True\n"
     "assert word_break('applepenapple',['apple','pen']) is True\n"
     "assert word_break('catsandog',['cats','dog','sand','and','cat']) is False\n"),
    ("can_finish",
     "a function can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool where prerequisites[i] "
     "= [a, b] means you must take course b before a; return True iff all courses can be finished (the "
     "dependency graph has no cycle).",
     "can_finish",
     "assert can_finish(2,[[1,0]]) is True\nassert can_finish(2,[[1,0],[0,1]]) is False\n"
     "assert can_finish(3,[[1,0],[2,1]]) is True\nassert can_finish(1,[]) is True\n"),
    ("trap",
     "a function trap(height: list[int]) -> int that returns how much rain water can be trapped between the "
     "bars given the elevation map height.",
     "trap",
     "assert trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6\nassert trap([4,2,0,3,2,5]) == 9\n"
     "assert trap([]) == 0\nassert trap([1,2,3]) == 0\n"),
    ("num_islands",
     "a function num_islands(grid: list[list[str]]) -> int that counts islands in a 2D grid of '1' (land) "
     "and '0' (water); an island is land connected 4-directionally (up/down/left/right).",
     "num_islands",
     "assert num_islands([['1','1','0','0','0'],['1','1','0','0','0'],['0','0','1','0','0'],['0','0','0','1','1']]) == 3\n"
     "assert num_islands([['1','1','1'],['0','1','0'],['1','1','1']]) == 1\n"
     "assert num_islands([['0']]) == 0\n"),
    ("my_atoi",
     "a function my_atoi(s: str) -> int implementing string-to-integer: skip leading whitespace, optional "
     "single '+'/'-' sign, read consecutive digits until a non-digit, ignore the rest; if no digits, return 0; "
     "clamp the result to the 32-bit signed range [-2147483648, 2147483647].",
     "my_atoi",
     "assert my_atoi('42') == 42\nassert my_atoi('   -42') == -42\nassert my_atoi('4193 with words') == 4193\n"
     "assert my_atoi('words and 987') == 0\nassert my_atoi('-91283472332') == -2147483648\n"
     "assert my_atoi('91283472332') == 2147483647\n"),
    # --- hard tier (notoriously subtle -- the ones a 7B often gets wrong; a real discriminator) ---
    ("is_match",
     "a function is_match(s: str, p: str) -> bool implementing regular-expression matching where '.' matches "
     "any single character and '*' matches zero or more of the PRECEDING element; the match must cover the "
     "ENTIRE input string s (not partial).",
     "is_match",
     "assert is_match('aa','a') is False\nassert is_match('aa','a*') is True\nassert is_match('ab','.*') is True\n"
     "assert is_match('aab','c*a*b') is True\nassert is_match('mississippi','mis*is*p*.') is False\n"),
    ("min_window",
     "a function min_window(s: str, t: str) -> str that returns the minimum-length substring of s containing "
     "every character of t including multiplicity; return '' if there is no such window.",
     "min_window",
     "assert min_window('ADOBECODEBANC','ABC') == 'BANC'\nassert min_window('a','a') == 'a'\n"
     "assert min_window('a','aa') == ''\n"),
    ("longest_valid_parentheses",
     "a function longest_valid_parentheses(s: str) -> int that returns the length of the longest substring of "
     "well-formed (correctly matched) parentheses.",
     "longest_valid_parentheses",
     "assert longest_valid_parentheses('(()') == 2\nassert longest_valid_parentheses(')()())') == 4\n"
     "assert longest_valid_parentheses('') == 0\nassert longest_valid_parentheses('()(()') == 2\n"),
    ("decode_ways",
     "a function decode_ways(s: str) -> int that returns the number of ways to decode a digit string where "
     "'1'..'26' map to 'A'..'Z' (a leading zero or any invalid grouping contributes no decoding).",
     "decode_ways",
     "assert decode_ways('12') == 2\nassert decode_ways('226') == 3\nassert decode_ways('06') == 0\n"
     "assert decode_ways('10') == 1\nassert decode_ways('0') == 0\n"),
    # --- BRUTAL tier (where a strong 7B usually drops below 100% -- the real discriminator for 'a stronger brain') ---
    ("calculate",
     "a function calculate(s: str) -> int that evaluates an arithmetic expression string containing "
     "non-negative integers and the operators + - * / and parentheses, with normal precedence; integer "
     "division truncates toward zero; ignore spaces. Do NOT use eval().",
     "calculate",
     "assert calculate('3+2*2') == 7\nassert calculate(' 3/2 ') == 1\n"
     "assert calculate('(1+(4+5+2)-3)+(6+8)') == 23\nassert calculate('2*(5+5*2)/3+(6/2+8)') == 21\n"
     "assert calculate('14-3/2') == 13\n"),
    ("total_n_queens",
     "a function total_n_queens(n: int) -> int that returns the number of distinct solutions to the "
     "n-queens puzzle on an n x n board.",
     "total_n_queens",
     "assert total_n_queens(1) == 1\nassert total_n_queens(2) == 0\nassert total_n_queens(3) == 0\n"
     "assert total_n_queens(4) == 2\nassert total_n_queens(8) == 92\n"),
    ("ladder_length",
     "a function ladder_length(begin_word: str, end_word: str, word_list: list[str]) -> int that returns the "
     "number of words in the shortest transformation sequence from begin_word to end_word, changing one letter "
     "at a time, where every intermediate word must be in word_list; return 0 if no such sequence exists.",
     "ladder_length",
     "assert ladder_length('hit','cog',['hot','dot','dog','lot','log','cog']) == 5\n"
     "assert ladder_length('hit','cog',['hot','dot','dog','lot','log']) == 0\n"),
    ("LRUCache",
     "a class LRUCache with __init__(self, capacity: int), get(self, key: int) -> int (returns the value or "
     "-1 if absent), and put(self, key: int, value: int) -> None; when capacity is exceeded it evicts the "
     "least-recently-used key. get and put both count as a use.",
     "LRUCache",
     "c = LRUCache(2)\nc.put(1,1)\nc.put(2,2)\nassert c.get(1) == 1\nc.put(3,3)\nassert c.get(2) == -1\n"
     "c.put(4,4)\nassert c.get(1) == -1\nassert c.get(3) == 3\nassert c.get(4) == 4\n"),
    ("multiply",
     "a function multiply(num1: str, num2: str) -> str that returns the product of two non-negative integers "
     "given as strings, as a string, WITHOUT converting the whole inputs to int (no int(num1)) and without "
     "big-integer library tricks.",
     "multiply",
     "assert multiply('2','3') == '6'\nassert multiply('123','456') == '56088'\n"
     "assert multiply('0','52') == '0'\nassert multiply('999','999') == '998001'\n"),
    ("jump",
     "a function jump(nums: list[int]) -> int that returns the minimum number of jumps to reach the last "
     "index, where nums[i] is the max jump length from index i (assume the end is always reachable).",
     "jump",
     "assert jump([2,3,1,1,4]) == 2\nassert jump([2,3,0,1,4]) == 2\nassert jump([0]) == 0\n"
     "assert jump([1,2,3]) == 2\n"),
    ("largest_rectangle_area",
     "a function largest_rectangle_area(heights: list[int]) -> int that returns the area of the largest "
     "rectangle in a histogram whose bar heights are given by heights (each bar width 1).",
     "largest_rectangle_area",
     "assert largest_rectangle_area([2,1,5,6,2,3]) == 10\nassert largest_rectangle_area([2,4]) == 4\n"
     "assert largest_rectangle_area([]) == 0\nassert largest_rectangle_area([5]) == 5\n"
     "assert largest_rectangle_area([6,2,5,4,5,1,6]) == 12\n"),
    ("max_sliding_window",
     "a function max_sliding_window(nums: list[int], k: int) -> list[int] that returns a list of the maximum "
     "value in each contiguous window of size k as the window slides left to right.",
     "max_sliding_window",
     "assert max_sliding_window([1,3,-1,-3,5,3,6,7],3) == [3,3,5,5,6,7]\n"
     "assert max_sliding_window([1],1) == [1]\nassert max_sliding_window([9,8,7,6],2) == [9,8,7]\n"),
]

# tier of each task (for stratified reporting -- where does a model's ceiling actually sit?)
TIER_OF = {}
for _i, _t in enumerate(TASKS):
    _name = _t[0]
    if _i < 10:
        TIER_OF[_name] = "1-core"
    elif _i < 22:
        TIER_OF[_name] = "2-hard"
    else:
        TIER_OF[_name] = "3-brutal"

PROMPT_TMPL = ("Write {spec}\n\nReturn ONLY the complete function definition inside a single ```python code "
               "block, with no explanation and no example usage.")


def query(model, prompt, timeout=240):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 2048, "seed": 7}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    wall = time.time() - t0
    ev, evd = d.get("eval_count", 0), d.get("eval_duration", 0) or 0
    tps = (ev / (evd / 1e9)) if evd else 0.0
    return d.get("response", ""), wall, ev, tps


def extract_code(text):
    """Robust extraction so a FAIL reflects the model, not the parser/truncation. Prefer a fenced block that
    actually contains a def/class; tolerate an UNCLOSED final fence; else take from the first def/class onward."""
    text = text.replace("```py\n", "```python\n")
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    for b in blocks:                                   # a closed block with real code
        if re.search(r"^\s*(?:def|class)\s", b, re.M):
            return b.strip()
    if blocks:
        return blocks[0].strip()
    m = re.search(r"```(?:python)?\s*\n(.*)$", text, re.S)   # unclosed final fence (truncated reply)
    if m and re.search(r"^\s*(?:def|class)\s", m.group(1), re.M):
        return m.group(1).strip()
    m = re.search(r"((?:^|\n)[ \t]*(?:def|class)\s.*)$", text, re.S)  # no fence at all
    return (m.group(1) if m else text).strip()


def run_task(code, fn_name, test):
    src = code + "\n\n" + f"assert callable({fn_name}), '{fn_name} not defined'\n" + test + "\nprint('PASS')\n"
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "sol.py"
        fp.write_text(src, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(fp)], capture_output=True, text=True,
                               timeout=15, creationflags=NW)
            return (r.returncode == 0 and "PASS" in r.stdout), (r.stderr or r.stdout).strip()[-200:]
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT(15s)"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


def model_available(model):
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as r:
            tags = [m["name"] for m in json.loads(r.read()).get("models", [])]
        return any(model == t or model in t or t.startswith(model) for t in tags)
    except Exception:
        return True  # let the query fail loudly if the daemon is down


def bench_model(model, tasks=None):
    tasks = tasks if tasks is not None else TASKS
    rows, passes, lat, tps_all = [], 0, [], []
    for name, spec, fn, test in tasks:
        prompt = PROMPT_TMPL.format(spec=spec)
        try:
            resp, wall, ev, tps = query(model, prompt)
        except Exception as e:
            rows.append({"task": name, "tier": TIER_OF.get(name, "?"), "pass": False,
                         "err": f"query: {type(e).__name__}: {str(e)[:120]}"})
            print(f"  [{model}] {name:24} QUERY-FAIL  :: {type(e).__name__}: {str(e)[:120]}")
            continue
        ok, err = run_task(extract_code(resp), fn, test)
        passes += int(ok); lat.append(wall); tps_all.append(tps)
        tier = TIER_OF.get(name, "?")
        rows.append({"task": name, "tier": tier, "pass": ok, "latency_s": round(wall, 1),
                     "tok_s": round(tps, 1), "err": "" if ok else err})
        print(f"  [{model}] {tier:8} {name:24} {'PASS' if ok else 'FAIL'}  {wall:5.1f}s  {tps:5.1f} tok/s"
              + ("" if ok else f"  :: {err[:70]}"))
    n = len(tasks)
    tiers = {}
    for r in rows:
        d = tiers.setdefault(r["tier"], [0, 0]); d[0] += int(r["pass"]); d[1] += 1
    by_tier = {t: f"{p}/{tot}" for t, (p, tot) in sorted(tiers.items())}
    return {"model": model, "pass_at_1": passes, "n": n, "pct": round(100 * passes / n, 1),
            "by_tier": by_tier,
            "mean_latency_s": round(sum(lat) / len(lat), 1) if lat else None,
            "mean_tok_s": round(sum(tps_all) / len(tps_all), 1) if tps_all else None, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N tasks (quick speed probe on a slow/large model)")
    a = ap.parse_args()
    models = a.models or [m for m in [
        "qwen2.5-coder:7b",
        "hf.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M",
        "qwen2.5-coder:3b",
    ] if model_available(m)]
    tasks = TASKS[:a.limit] if a.limit else TASKS
    print(f"benchmarking {len(models)} model(s) x {len(tasks)} verifiable tasks (temp 0)\n")
    results = []
    for m in models:
        print(f"== {m} ==")
        results.append(bench_model(m, tasks))
        print()
    # markdown leaderboard
    print("\n## Brain benchmark -- pass@1 on " + str(len(tasks)) + " verifiable Python tasks (temp 0)\n")
    print("Tiers: 1-core (standard) / 2-hard (DP/graph/hard-string) / 3-brutal (calc/n-queens/LRU/histogram...).\n")
    print("| model | pass@1 | core | hard | brutal | mean latency | mean tok/s |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: (-x["pass_at_1"], x["mean_latency_s"] or 1e9)):
        bt = r["by_tier"]
        print(f"| {r['model']} | {r['pass_at_1']}/{r['n']} ({r['pct']}%) | {bt.get('1-core','-')} | "
              f"{bt.get('2-hard','-')} | {bt.get('3-brutal','-')} | {r['mean_latency_s']}s | {r['mean_tok_s']} |")
    out = ROOT / "runs" / "autonomy" / f"brain_benchmark_{len(tasks)}tasks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
