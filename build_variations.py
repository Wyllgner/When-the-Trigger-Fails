"""Freezes the metamorphic variations into variations.jsonl.

For each task it generates the four input texts:
  mr0 = original prompt_en (baseline / flakiness)
  mr1 = manual paraphrase (paraphrases.jsonl)
  mr2 = prompt_en + deterministic noise (fixed seed)
  mr3 = prompt_pt (language swap)

Run it once: variations.jsonl is the frozen experimental input and should not
be regenerated after the experiment starts.
"""
import json
from noise import add_noise

NOISE_SEED = 42

tasks = [json.loads(l) for l in open("tasks.jsonl", encoding="utf-8")]
paraphrases = {json.loads(l)["id"]: json.loads(l)["mr1"]
               for l in open("paraphrases.jsonl", encoding="utf-8")}

with open("variations.jsonl", "w", encoding="utf-8") as out:
    for task in tasks:
        tid = task["id"]
        if tid not in paraphrases:
            raise SystemExit(f"Missing paraphrase (mr1) for {tid} in paraphrases.jsonl")
        row = {
            "id": tid,
            "mr0": task["prompt_en"],
            "mr1": paraphrases[tid],
            "mr2": add_noise(task["prompt_en"], seed=NOISE_SEED),
            "mr3": task["prompt_pt"],
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")

print("variations.jsonl generated with", len(tasks), "tasks x 4 MRs.")
