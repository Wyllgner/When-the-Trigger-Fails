"""Comparator and metrics.

Reads results.jsonl (or results_pilot.jsonl with --pilot) and produces
Table 1 (MVR per model x MR, with Wilson CIs), Table 2 (MR0 flakiness),
Table 3 (inconsistency types), Table 4 (severity of each violation under the
tool-aware equivalence), Table 5 (the funnel from raw calls to violations),
the Wilcoxon test plus a bootstrap CI between the two models, and mvr.png.

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
  * Severity of a violation (tool-aware equivalence, see SEVERITY below): S0
    acceptable, S1 tool-dependent, S2 functional defect. The strict MVR counts
    S0+S1+S2; the defect MVR counts only S1+S2.
"""
import argparse
import ast
import json
import math
from collections import Counter

import pandas as pd

# --- Tool-aware argument equivalence -------------------------------------
# free: the tool consumes the argument as natural-language payload, so a
#       faithful translation/paraphrase preserves the function of the call.
# resolved: the tool resolves the argument against an external vocabulary or
#       identifier (a geocoder, a language code, a ticker, a date), so any
#       change of surface form may change or break the resolution.
ARG_CLASS = {
    ("search_web", "query"): "free",
    ("create_calendar_event", "title"): "free",
    ("set_reminder", "task"): "free",
    ("translate_text", "text"): "free",
    ("send_email", "subject"): "free",
    ("send_email", "body"): "free",
    ("get_weather", "city"): "resolved",
    ("get_weather", "unit"): "resolved",
    ("translate_text", "target_language"): "resolved",
    ("create_calendar_event", "date"): "resolved",
    ("create_calendar_event", "time"): "resolved",
    ("set_reminder", "time"): "resolved",
    ("get_stock_price", "ticker"): "resolved",
    ("convert_currency", "amount"): "resolved",
    ("convert_currency", "from_currency"): "resolved",
    ("convert_currency", "to_currency"): "resolved",
    ("send_email", "to"): "resolved",
}

# Human labels (inspecao.csv) that denote a meaning-preserving rewrite of the
# argument. Every other label denotes corruption or a semantic error.
MEANING_PRESERVING = {"traducao_do_argumento", "parafrase_mudou_query"}

SEVERITY = {
    "S0": "acceptable (meaning preserved, free-text argument)",
    "S1": "tool-dependent (meaning preserved, resolved argument)",
    "S2": "functional defect (corruption, semantic error, tool switch, refusal)",
}


def boot_ci(xa, xb, n_boot=10000, seed=42):
    """Percentile bootstrap CI for the mean paired difference xa - xb."""
    import random
    d = [float(u - v) for u, v in zip(xa, xb)]
    n = len(d)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in range(n)) / n
                   for _ in range(n_boot))
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return lo, hi, sum(d) / n


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion; (lo, hi) in [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


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

    # --- violation-level detail + tool-aware severity ---------------------
    viol_rows = []
    for (model, task_id), r in pivot[stable].iterrows():
        sig0 = ast.literal_eval(r["mr0"])
        args0 = dict(sig0[1])
        for mr in ["mr1", "mr2", "mr3"]:
            if _null(r[mr]) or r[mr] == r["mr0"]:
                continue
            sigx = ast.literal_eval(r[mr])
            argsx = dict(sigx[1])
            diff = sorted(k for k in set(args0) | set(argsx)
                          if args0.get(k) != argsx.get(k))
            classes = {ARG_CLASS.get((sig0[0], k), "resolved") for k in diff}
            viol_rows.append(dict(
                model=model, task_id=task_id, mr=mr,
                tool_mr0=sig0[0], tool_mrX=sigx[0],
                tool_switch=int(sig0[0] != sigx[0]),
                diff_keys=",".join(diff),
                arg_classes=",".join(sorted(classes)) or "-"))
    viols = pd.DataFrame(viol_rows, columns=[
        "model", "task_id", "mr", "tool_mr0", "tool_mrX", "tool_switch",
        "diff_keys", "arg_classes"])
    print(f"\nViolacoes detectadas (tarefas estaveis em MR0): {len(viols)}")

    # human inspection labels; without them we cannot separate a faithful
    # rewrite from a corrupted one, so severity falls back to the strict view
    try:
        ins = pd.read_csv("inspecao.csv")[["model", "task_id", "mr", "causa"]]
        viols = viols.merge(ins, on=["model", "task_id", "mr"], how="left")
        missing = int(viols["causa"].isna().sum())
        if missing:
            print(f"[aviso] {missing} violacoes sem rotulo em inspecao.csv "
                  "-> tratadas como S2 (conservador).")
    except FileNotFoundError:
        print("[aviso] inspecao.csv ausente; severidade nao calculada.")
        viols["causa"] = None

    def severity(row):
        if row["tool_switch"] or row["tool_mrX"] == "refused":
            return "S2"
        if row["causa"] not in MEANING_PRESERVING:
            return "S2"  # corruption, semantic error or unlabelled
        return "S0" if row["arg_classes"] == "free" else "S1"

    viols["severidade"] = viols.apply(severity, axis=1)
    viols["defeito"] = (viols["severidade"] != "S0").astype(int)
    viols.sort_values(["model", "task_id", "mr"]).to_csv(
        "tabela4_severidade.csv", index=False)

    defect_keys = set(map(tuple, viols[viols.defeito == 1][
        ["model", "task_id", "mr"]].values))
    for mr in ["mr1", "mr2", "mr3"]:
        pivot[f"def_{mr}"] = [
            int((m, t, mr) in defect_keys) for m, t in pivot.index]

    # Table 1: MVR per model x MR (strict and defect-only, with Wilson CIs)
    print("\n=== Tabela 1: MVR (taxa de violacao) por modelo x MR ===")
    t1 = pivot[stable].groupby(level="model")[["viol_mr1", "viol_mr2", "viol_mr3"]].mean()
    print((t1 * 100).round(1).astype(str) + "%")
    t1.to_csv("tabela1_mvr.csv")

    print("\n=== Tabela 1b: MVR estrita vs. MVR de defeito, com IC95% (Wilson) ===")
    rows_ci = []
    for model, g in pivot[stable].groupby(level="model"):
        n = len(g)
        for mr in ["mr1", "mr2", "mr3"]:
            k_s = int(g[f"viol_{mr}"].sum())
            k_d = int(g[f"def_{mr}"].sum())
            lo_s, hi_s = wilson(k_s, n)
            lo_d, hi_d = wilson(k_d, n)
            rows_ci.append(dict(
                model=model, mr=mr, n=n,
                viol_estrita=k_s, mvr_estrita=k_s / n,
                ic95_estrita_lo=lo_s, ic95_estrita_hi=hi_s,
                viol_defeito=k_d, mvr_defeito=k_d / n,
                ic95_defeito_lo=lo_d, ic95_defeito_hi=hi_d))
    t1b = pd.DataFrame(rows_ci)
    fmt = t1b.assign(
        estrita=lambda d: (100 * d.mvr_estrita).round(1).astype(str) + "% ["
                          + (100 * d.ic95_estrita_lo).round(1).astype(str) + ", "
                          + (100 * d.ic95_estrita_hi).round(1).astype(str) + "]",
        defeito=lambda d: (100 * d.mvr_defeito).round(1).astype(str) + "% ["
                          + (100 * d.ic95_defeito_lo).round(1).astype(str) + ", "
                          + (100 * d.ic95_defeito_hi).round(1).astype(str) + "]")
    print(fmt[["model", "mr", "n", "viol_estrita", "estrita",
               "viol_defeito", "defeito"]].to_string(index=False))
    t1b.to_csv("tabela1b_mvr_ic.csv", index=False)

    print("\n=== Tabela 4: severidade das violacoes (equivalencia por ferramenta) ===")
    for k, v in SEVERITY.items():
        print(f"  {k} = {v}")
    print(viols.groupby(["severidade", "causa"]).size().rename("n").to_string())
    print(viols.groupby(["model", "mr", "severidade"]).size()
               .unstack(fill_value=0).to_string())

    # Table 2: MR0 flakiness
    print("\n=== Tabela 2: Flakiness MR0 (media de assinaturas distintas; % tarefas flaky) ===")
    t2 = pivot.groupby(level="model")["flaky"].agg(
        media_assinaturas="mean",
        pct_flaky=lambda s: (s > 1).mean())
    print(t2.round(3))
    t2.to_csv("tabela2_flakiness.csv")

    # Table 3: inconsistency types
    df["tipo"] = df.apply(lambda r: classify(
        r["tool_called"], r["args_called"], r["expected_tool"],
        r["expected_args"], r.get("raw")), axis=1)
    print("\n=== Tabela 3: Tipos de inconsistencia (chamadas em mr1/mr2/mr3) ===")
    mrx = df[df.mr.isin(["mr1", "mr2", "mr3"])]
    t3 = mrx.groupby(["model", "tipo"]).size().unstack(fill_value=0)
    print(t3)
    t3.to_csv("tabela3_tipos.csv")

    # Table 5: funnel from raw calls to violations. Explains why hundreds of
    # ground-truth deviations collapse into a few dozen MR violations.
    print("\n=== Tabela 5: funil (chamadas brutas -> violacoes) ===")
    mrx = mrx.copy()
    mrx["sig_mr0"] = [pivot["mr0"].get((m, t)) for m, t in
                      zip(mrx.model, mrx.task_id)]
    mrx["estavel"] = [pivot["evaluable"].get((m, t)) == 1 for m, t in
                      zip(mrx.model, mrx.task_id)]
    funnel = []
    for model, g in df.groupby("model"):
        gx = mrx[mrx.model == model]
        v = viols[viols.model == model]
        # sequential drops, call level
        dev = gx[gx.tipo != "ok"]
        dev_stable = dev[dev["estavel"]]
        dev_new = dev_stable[dev_stable["sig_str"] != dev_stable["sig_mr0"]]
        assert len(dev_new) == len(dev_stable), (
            "expected every MR0-stable task to match the ground truth at MR0")
        # aggregation to (task, MR) cases decided by the modal signature
        cases_touched = dev_new.groupby(["task_id", "mr"]).ngroups
        funnel.append(dict(
            model=model,
            invocacoes=len(g),
            chamadas_sob_transformacao=len(gx),
            desvios_do_ground_truth=len(dev),
            desvios_em_tarefas_mr0_estaveis=len(dev_stable),
            casos_tarefa_x_mr_afetados=cases_touched,
            casos_avaliaveis=int(pivot.loc[model, "evaluable"].sum()) * 3,
            casos_com_desvio_minoritario=cases_touched - len(v),
            violacoes_estritas=len(v),
            violacoes_defeito=int(v["defeito"].sum()),
            S0=int((v.severidade == "S0").sum()),
            S1=int((v.severidade == "S1").sum()),
            S2=int((v.severidade == "S2").sum())))
    t5 = pd.DataFrame(funnel).set_index("model")
    print(t5.T.to_string())
    t5.to_csv("tabela5_funil.csv")

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
            # defect-only counterpart of the same paired comparison
            da = pa.loc[common][["def_mr1", "def_mr2", "def_mr3"]].mean(axis=1)[mask]
            db = pb.loc[common][["def_mr1", "def_mr2", "def_mr3"]].mean(axis=1)[mask]
            if (sa - sb).abs().sum() > 0:
                stat, p = wilcoxon(sa, sb)
                print(f"\n=== RQ4: Wilcoxon {a} vs {b} ===")
                print(f"  n_pares={mask.sum()} | stat={stat:.3f} | p={p:.4f}")
                print(f"  MVR medio: {a}={sa.mean():.3f} | {b}={sb.mean():.3f}")
                for label, xa, xb in (("estrita", sa, sb), ("defeito", da, db)):
                    lo, hi, diff = boot_ci(xa.values, xb.values)
                    print(f"  [{label}] diff media ({a} - {b}) = {diff:+.3f} "
                          f"| IC95% bootstrap = [{lo:+.3f}, {hi:+.3f}] "
                          f"| MVR: {xa.mean():.3f} vs {xb.mean():.3f}")
                if (da - db).abs().sum() > 0:
                    _, p_def = wilcoxon(da, db)
                    print(f"  [defeito] Wilcoxon p={p_def:.4f}")
                else:
                    print("  [defeito] sem diferencas pareadas nao nulas; "
                          "Wilcoxon nao aplicavel.")
            else:
                print("\n[RQ4] Sem diferenca entre modelos (todas as diffs sao zero).")
        except ImportError:
            print("\n[RQ4] scipy nao instalado, pulei Wilcoxon.")

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
        plt.close()

        # strict MVR vs. defect-only MVR, with Wilson CIs
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, (model, g) in zip(axes, t1b.groupby("model")):
            g = g.set_index("mr").loc[["mr1", "mr2", "mr3"]]
            x = range(len(g))
            for off, col, lo, hi, lab in (
                    (-0.19, "mvr_estrita", "ic95_estrita_lo", "ic95_estrita_hi",
                     "MVR estrita"),
                    (0.19, "mvr_defeito", "ic95_defeito_lo", "ic95_defeito_hi",
                     "MVR de defeito (S1+S2)")):
                pos = [i + off for i in x]
                vals = (100 * g[col]).values
                err = [(vals - (100 * g[lo]).values).clip(0),
                       ((100 * g[hi]).values - vals).clip(0)]
                ax.bar(pos, vals, width=0.36, label=lab)
                ax.errorbar(pos, vals, yerr=err, fmt="none", ecolor="black",
                            capsize=3, lw=1)
            ax.set_xticks(list(x))
            ax.set_xticklabels([m.upper() for m in g.index])
            ax.set_title(model)
        axes[0].set_ylabel("Taxa de violacao (%)")
        axes[0].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("mvr_severidade.png", dpi=200)
        plt.close()
        print("\nGraficos salvos em mvr.png e mvr_severidade.png")
    except ImportError:
        print("\n[grafico] matplotlib nao instalado, pulei mvr.png.")

    print("\nCSVs: tabela1_mvr.csv, tabela1b_mvr_ic.csv, tabela2_flakiness.csv, "
          "tabela3_tipos.csv, tabela4_severidade.csv, tabela5_funil.csv")


if __name__ == "__main__":
    main()
