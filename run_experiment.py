"""Execution harness.

For every (model x task x MR x repetition) it calls the agent through Ollama
and appends one line per call to results.jsonl. The raw output is always
logged: if the model refuses or calls no tool, tool_called is None and the
text goes to 'raw'.

Usage:
    python run_experiment.py --pilot     # 1 model, 5 tasks
    python run_experiment.py             # full experiment
    python run_experiment.py --resume    # continue without re-running
"""
import argparse
import itertools
import json
import os
import time

import ollama

MODELS = ["qwen3.5:4b", "llama3.2:3b"]
MRS = ["mr0", "mr1", "mr2", "mr3"]
N = 5  # repetitions per case (non-determinism)
SYS = ("You are a helpful assistant with access to tools. "
       "Use exactly one tool to fulfill the user's request.")
OUT_PATH = "results.jsonl"

PILOT_MODELS = ["qwen3.5:4b"]
PILOT_TASK_IDS = ["t01", "t06", "t11", "t21", "t26"]  # one per tool family
PILOT_OUT_PATH = "results_pilot.jsonl"


def load_inputs():
    tools = json.load(open("tools.json", encoding="utf-8"))
    tasks = [json.loads(l) for l in open("tasks.jsonl", encoding="utf-8")]
    variations = {json.loads(l)["id"]: json.loads(l)
                  for l in open("variations.jsonl", encoding="utf-8")}
    return tools, tasks, variations


def done_keys(path):
    """Keys (model, task, mr, rep) already logged, used by --resume."""
    keys = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try:
                d = json.loads(l)
                keys.add((d["model"], d["task_id"], d["mr"], d["rep"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def run_one(model, tools, user_text, rep):
    """One agent call. Returns (tool_called, args_called, raw, latency_ms, error)."""
    t0 = time.time()
    try:
        r = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": SYS},
                      {"role": "user", "content": user_text}],
            tools=tools,
            options={"temperature": 0.7, "seed": rep},
        )
    except Exception as e:  # model/network failure: log and continue
        return None, None, None, int((time.time() - t0) * 1000), str(e)

    latency_ms = int((time.time() - t0) * 1000)
    msg = r.get("message", {}) or {}
    tcs = msg.get("tool_calls") or []
    if tcs:
        fn = (tcs[0] or {}).get("function", {}) or {}
        return fn.get("name"), fn.get("arguments"), msg.get("content"), latency_ms, None
    return None, None, msg.get("content"), latency_ms, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="1 model, 5 tasks")
    ap.add_argument("--resume", action="store_true", help="continue without re-running")
    args = ap.parse_args()

    tools, tasks, variations = load_inputs()

    if args.pilot:
        models = PILOT_MODELS
        tasks = [t for t in tasks if t["id"] in PILOT_TASK_IDS]
        out_path = PILOT_OUT_PATH
    else:
        models = MODELS
        out_path = OUT_PATH

    already = done_keys(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    combos = list(itertools.product(models, tasks, MRS, range(N)))
    total = len(combos)

    print(f"models={models} | tasks={len(tasks)} | MRs={MRS} | N={N}")
    print(f"total calls: {total} | already done (resume): {len(already)}")

    out = open(out_path, mode, encoding="utf-8")
    done = 0
    t_start = time.time()
    for model, task, mr, rep in combos:
        done += 1
        key = (model, task["id"], mr, rep)
        if key in already:
            continue

        user_text = variations[task["id"]][mr]
        tool_called, args_called, raw, latency_ms, error = run_one(
            model, tools, user_text, rep)

        out.write(json.dumps({
            "model": model, "task_id": task["id"], "mr": mr, "rep": rep,
            "input": user_text,
            "tool_called": tool_called,
            "args_called": args_called,
            "expected_tool": task["expected_tool"],
            "expected_args": task["expected_args"],
            "raw": raw,
            "latency_ms": latency_ms,
            "error": error,
        }, ensure_ascii=False) + "\n")
        out.flush()

        if done % 20 == 0 or done == total:
            elapsed = time.time() - t_start
            print(f"  {done}/{total}  ({elapsed:.0f}s)  last: "
                  f"{model} {task['id']} {mr} -> {tool_called}")
    out.close()
    print(f"Done. Written to {out_path}")


if __name__ == "__main__":
    main()
