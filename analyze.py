"""Comparator and metrics.

Reads results.jsonl (or results_pilot.jsonl with --pilot) and produces
Table 1 (MVR per model x MR), Table 2 (MR0 flakiness), Table 3 (inconsistency
types), the Wilcoxon test between the two models, and mvr.png.

Definitions:
  * Behaviour signature of a call = (tool, essential args), where the essential
    args are the keys present in expected_args (optional fields such as
    subject/body/unit are ignored).
  * MR violation (mr1/mr2/mr3): the MODAL signature under the transformation
    differs from the modal signature under the original prompt (mr0), for the
    same (model, task).
  * Flakiness (MR0): number of distinct signatures across the N mr0 repetitions.
    Only tasks that are non-flaky under mr0 count as violations, so intrinsic
    noise is not confused with the effect of the transformation.
"""
import argparse
import json
from collections import Counter

import pandas as pd


def _null(v):
    """True for None or pandas NaN."""
    return v is None or (isinstance(v, float) and pd.isna(v))


def norm_tool(tool):
    if _null(tool) or not str(tool).strip():
        return "refused"
    return str(tool).strip().lower()


def norm_val(v):
    # numbers are compared by value, regardless of int/float/str ("100" == 100)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            return float(s.replace(",", "."))
        except ValueError:
            return s
    return v


def essential_sig(tool, args_called, expected_args):
    """Signature: (normalized tool, sorted tuple of the essential args)."""
    tool = norm_tool(tool)
    if _null(args_called):
        args_called = {}
    essential = {}
    for k in (expected_args or {}):
        essential[k] = norm_val(args_called.get(k))
    return (tool, tuple(sorted(essential.items(), key=lambda x: x[0])))


def args_match(args_called, expected_args):
    if _null(args_called):
        args_called = {}
    for k, v in (expected_args or {}).items():
        if norm_val(args_called.get(k)) != norm_val(v):
            return False
    return True


def classify(tool, args_called, expected_tool, expected_args, raw):
    """Inconsistency type for Table 3."""
    if _null(tool):
        return "recusa/sem_tool"
    if str(tool).strip().lower() != expected_tool.strip().lower():
        return "troca_de_ferramenta"
    if not args_match(args_called, expected_args):
        return "mudanca_de_args"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--path", default=None)
    args = ap.parse_args()
    path = args.path or ("results_pilot.jsonl" if args.pilot else "results.jsonl")

    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame(rows)
    print(f"Lido {path}: {len(df)} chamadas | modelos={sorted(df.model.unique())}")

    df["sig"] = df.apply(lambda r: essential_sig(
        r["tool_called"], r["args_called"], r["expected_args"]), axis=1)
    df["sig_str"] = df["sig"].map(str)
    df["correct"] = df.apply(lambda r: (
        norm_tool(r["tool_called"]) == r["expected_tool"].strip().lower()
        and args_match(r["args_called"], r["expected_args"])), axis=1)

    # modal signature per (model, task, mr)
    mode = (df.groupby(["model", "task_id", "mr"])["sig_str"]
              .agg(lambda s: s.value_counts().idxmax()).reset_index())
    pivot = mode.pivot_table(index=["model", "task_id"], columns="mr",
                             values="sig_str", aggfunc="first")

    # mr0 flakiness: number of distinct signatures across the N repetitions
    flak = (df[df.mr == "mr0"].groupby(["model", "task_id"])["sig_str"]
              .nunique().rename("flaky"))
    pivot = pivot.join(flak, on=["model", "task_id"])

    # violations per MR, restricted to tasks that are stable under mr0
    stable = pivot["flaky"] == 1
    for mr in ["mr1", "mr2", "mr3"]:
        pivot[f"viol_{mr}"] = ((pivot[mr] != pivot["mr0"]) & stable).astype(int)
    pivot["evaluable"] = stable.astype(int)

    # Table 1: MVR per model x MR
    print("\n=== Tabela 1 — MVR (taxa de violacao) por modelo x MR ===")
    t1 = pivot[stable].groupby(level="model")[["viol_mr1", "viol_mr2", "viol_mr3"]].mean()
    print((t1 * 100).round(1).astype(str) + "%")
    t1.to_csv("tabela1_mvr.csv")

    # Table 2: MR0 flakiness
    print("\n=== Tabela 2 — Flakiness MR0 (media de assinaturas distintas; % tarefas flaky) ===")
    t2 = pivot.groupby(level="model")["flaky"].agg(
        media_assinaturas="mean",
        pct_flaky=lambda s: (s > 1).mean())
    print(t2.round(3))
    t2.to_csv("tabela2_flakiness.csv")

    # Table 3: inconsistency types
    df["tipo"] = df.apply(lambda r: classify(
        r["tool_called"], r["args_called"], r["expected_tool"],
        r["expected_args"], r.get("raw")), axis=1)
    print("\n=== Tabela 3 — Tipos de inconsistencia (chamadas em mr1/mr2/mr3) ===")
    mrx = df[df.mr.isin(["mr1", "mr2", "mr3"])]
    t3 = mrx.groupby(["model", "tipo"]).size().unstack(fill_value=0)
    print(t3)
    t3.to_csv("tabela3_tipos.csv")

    # Wilcoxon test between the two models
    models = sorted(df.model.unique())
    if len(models) == 2:
        try:
            from scipy.stats import wilcoxon
            a, b = models
            pa = pivot.loc[a]; pb = pivot.loc[b]
            common = pa.index.intersection(pb.index)
            # mean MVR per task (over the 3 MRs) on tasks evaluable in both
            sa = pa.loc[common][["viol_mr1", "viol_mr2", "viol_mr3"]].mean(axis=1)
            sb = pb.loc[common][["viol_mr1", "viol_mr2", "viol_mr3"]].mean(axis=1)
            mask = (pa.loc[common, "evaluable"] == 1) & (pb.loc[common, "evaluable"] == 1)
            sa, sb = sa[mask], sb[mask]
            if (sa - sb).abs().sum() > 0:
                stat, p = wilcoxon(sa, sb)
                print(f"\n=== RQ4 — Wilcoxon {a} vs {b} ===")
                print(f"  n_pares={mask.sum()} | stat={stat:.3f} | p={p:.4f}")
                print(f"  MVR medio: {a}={sa.mean():.3f} | {b}={sb.mean():.3f}")
            else:
                print("\n[RQ4] Sem diferenca entre modelos (todas as diffs sao zero).")
        except ImportError:
            print("\n[RQ4] scipy nao instalado — pulei Wilcoxon.")

    # chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = (t1 * 100).plot.bar()
        ax.set_ylabel("Taxa de violacao MVR (%)")
        ax.set_xlabel("Modelo")
        ax.legend(title="MR")
        plt.tight_layout()
        plt.savefig("mvr.png", dpi=200)
        print("\nGrafico salvo em mvr.png")
    except ImportError:
        print("\n[grafico] matplotlib nao instalado — pulei mvr.png.")

    print("\nCSVs: tabela1_mvr.csv, tabela2_flakiness.csv, tabela3_tipos.csv")


if __name__ == "__main__":
    main()
