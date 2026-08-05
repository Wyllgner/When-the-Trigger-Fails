"""Qualitative inspection of the metamorphic violations.

Reads results.jsonl, identifies the real violations (modal signature under
mr1/mr2/mr3 different from the modal signature under mr0, on tasks that are
stable under mr0) and prints each one in a readable form for manual annotation
of the cause: model, task, MR, type, both inputs side by side, expected
behaviour, what the model did in each case and a sample of the raw output.

Usage:
    python inspect_violations.py            # all violations (results.jsonl)
    python inspect_violations.py --pilot    # results_pilot.jsonl
    python inspect_violations.py --max 30   # cap at 30
    python inspect_violations.py --mr mr3   # a single MR
    python inspect_violations.py --csv inspecao.csv   # table for annotation
"""
import argparse
import json
from collections import Counter

import pandas as pd

from analyze import essential_sig, norm_tool, args_match, classify


def modal_row(group):
    """Representative row: the one holding the modal behaviour of the group."""
    sig_counts = Counter(group["sig_str"])
    modal_sig = sig_counts.most_common(1)[0][0]
    sub = group[group["sig_str"] == modal_sig]
    return sub.iloc[0]


def fmt_call(tool, args):
    if tool is None or (isinstance(tool, float) and pd.isna(tool)):
        return "RECUSA (nenhuma tool)"
    a = args if isinstance(args, dict) else {}
    a = {k: v for k, v in a.items()}
    return f"{tool}({json.dumps(a, ensure_ascii=False)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--path", default=None)
    ap.add_argument("--max", type=int, default=None, help="cap on the number of violations")
    ap.add_argument("--mr", default=None, help="filter a single MR (mr1/mr2/mr3)")
    ap.add_argument("--csv", default=None, help="write a table for manual annotation")
    args = ap.parse_args()
    path = args.path or ("results_pilot.jsonl" if args.pilot else "results.jsonl")

    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["sig"] = df.apply(lambda r: essential_sig(
        r["tool_called"], r["args_called"], r["expected_args"]), axis=1)
    df["sig_str"] = df["sig"].map(str)

    # modal signature and representative row per (model, task, mr)
    modal = {}
    reps = {}
    for (model, task, mr), g in df.groupby(["model", "task_id", "mr"]):
        sig = Counter(g["sig_str"]).most_common(1)[0][0]
        modal[(model, task, mr)] = sig
        reps[(model, task, mr)] = modal_row(g)

    # a task is only evaluable if it is stable under mr0 (one distinct signature)
    flaky = (df[df.mr == "mr0"].groupby(["model", "task_id"])["sig_str"]
             .nunique().to_dict())

    mrs = [args.mr] if args.mr else ["mr1", "mr2", "mr3"]
    violations = []
    for (model, task, mr) in sorted(modal):
        if mr == "mr0":
            continue
        if mr not in mrs:
            continue
        if flaky.get((model, task), 99) != 1:
            continue  # flaky under mr0, excluded
        if modal[(model, task, mr)] != modal[(model, task, "mr0")]:
            violations.append((model, task, mr))

    print(f"Lido {path} | {len(violations)} violacoes reais "
          f"(MR estavel, modal != mr0){' [MR=' + args.mr + ']' if args.mr else ''}")
    print("=" * 78)

    csv_rows = []
    shown = 0
    for (model, task, mr) in violations:
        if args.max is not None and shown >= args.max:
            print(f"\n[... +{len(violations) - shown} violacoes nao exibidas "
                  f"(use --max maior)]")
            break
        shown += 1
        r0 = reps[(model, task, "mr0")]
        rx = reps[(model, task, mr)]
        tipo = classify(rx["tool_called"], rx["args_called"],
                        rx["expected_tool"], rx["expected_args"], rx.get("raw"))

        print(f"\n#{shown}  [{model}]  {task}  {mr.upper()}  ->  {tipo}")
        print(f"  esperado : {rx['expected_tool']}  args={json.dumps(rx['expected_args'], ensure_ascii=False)}")
        print(f"  input mr0: {r0['input']}")
        print(f"  input {mr}: {rx['input']}")
        print(f"  mr0  fez : {fmt_call(r0['tool_called'], r0['args_called'])}")
        print(f"  {mr}  fez : {fmt_call(rx['tool_called'], rx['args_called'])}")
        raw = rx.get("raw")
        if isinstance(raw, str) and raw.strip():
            print(f"  raw {mr} : {raw.strip()[:200]}")
        print("  causa (anote): _______________________________________________")

        csv_rows.append({
            "model": model, "task_id": task, "mr": mr, "tipo": tipo,
            "expected_tool": rx["expected_tool"],
            "input_mr0": r0["input"], "input_mrX": rx["input"],
            "mr0_call": fmt_call(r0["tool_called"], r0["args_called"]),
            "mrX_call": fmt_call(rx["tool_called"], rx["args_called"]),
            "raw_mrX": (rx.get("raw") or ""),
            "causa": "",
        })

    if args.csv and csv_rows:
        pd.DataFrame(csv_rows).to_csv(args.csv, index=False)
        print(f"\nTabela de anotacao gravada em {args.csv} "
              f"(coluna 'causa' vazia p/ voce preencher)")

    # summary per MR and per model
    if violations:
        by_mr = Counter(mr for _, _, mr in violations)
        by_model = Counter(m for m, _, _ in violations)
        print("\n" + "=" * 78)
        print("Resumo das violacoes:")
        print(f"  por MR    : {dict(by_mr)}")
        print(f"  por modelo: {dict(by_model)}")


if __name__ == "__main__":
    main()
