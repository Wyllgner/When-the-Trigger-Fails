# When the Trigger Fails: Metamorphic Testing of Tool-Calling AI Agents

Replication package for the paper *"Quando o Gatilho Falha: Teste Metamórfico de
Agentes de IA com Chamada de Ferramentas"* (When the Trigger Fails: Metamorphic
Testing of Tool-Calling AI Agents), accepted at **SAST 2026 / CBSoft 2026**.

**Paper (camera-ready PDF):** [`paper.pdf`](paper.pdf)

The study evaluates a catalogue of metamorphic relations (MRs) targeting the
*Trigger* component of an AI agent — the prompt. The premise is that
semantics-preserving transformations of the prompt should not change the agent's
behaviour, defined as the pair *(selected tool, tool arguments)*. When the
behaviour does change, a silent behavioural regression is revealed.

The experiment covers 2 open-source models running locally through
[Ollama](https://ollama.com) (`qwen3.5:4b` and `llama3.2:3b`), 40 tool-calling
tasks, 4 prompt variants and 5 repetitions per case, for a total of
**1,600 controlled agent invocations**.

## Metamorphic relations

| ID    | Transformation | Role |
|-------|----------------|------|
| `mr0` | none (original English prompt) | baseline, used to measure intrinsic flakiness |
| `mr1` | manual paraphrase | semantics-preserving rewording |
| `mr2` | deterministic surface noise (typos, doubled letters and spaces) | realistic typing noise |
| `mr3` | language swap (English → Portuguese) | same request, different language |

A violation is counted when, for a given (model, task), the **modal** behaviour
signature under `mr1`/`mr2`/`mr3` differs from the modal signature under `mr0`.
Tasks that are already unstable under `mr0` (flaky) are excluded, so intrinsic
non-determinism is not counted as sensitivity to the transformation.

## Repository contents

### Inputs (frozen experimental material)

| File | Description |
|------|-------------|
| `tools.json` | tool catalogue (JSON Schema) exposed to the agent |
| `tasks.jsonl` | 40 tasks: `prompt_en`, `prompt_pt`, `expected_tool`, `expected_args` |
| `paraphrases.jsonl` | manual paraphrases used by `mr1` |
| `variations.jsonl` | the four prompt variants per task, frozen (generated once) |

### Scripts

| File | Description |
|------|-------------|
| `noise.py` | deterministic noise generator used by `mr2` (fixed seed) |
| `build_variations.py` | builds `variations.jsonl` from the tasks and paraphrases |
| `run_experiment.py` | execution harness: calls the agent and logs every invocation |
| `analyze.py` | comparator and metrics: Tables 1–3, Wilcoxon test, `mvr.png` |
| `inspect_violations.py` | qualitative inspection of each real violation |
| `smoke_test.py` | minimal check that Ollama and tool-calling are working |

### Outputs (results reported in the paper)

| File | Description |
|------|-------------|
| `results.jsonl` | raw log of the 1,600 invocations (one JSON object per call) |
| `results_pilot.jsonl` | log of the pilot run (1 model, 5 tasks) |
| `run.log` | console log of the full experimental run |
| `tabela1_mvr.csv` | Table 1 — violation rate (MVR) per model × MR |
| `tabela2_flakiness.csv` | Table 2 — intrinsic flakiness under `mr0` |
| `tabela3_tipos.csv` | Table 3 — inconsistency types |
| `inspecao.csv` | violations exported for manual annotation of the cause |
| `mvr.png` | violation rate chart |

Each line of `results.jsonl` contains `model`, `task_id`, `mr`, `rep`, the exact
`input` sent to the agent, the `tool_called` and `args_called`, the expected
behaviour, the raw model output (`raw`), the latency and any error.

## Requirements

**Software**

- Linux, macOS or Windows (the reported run used Ubuntu with Linux 6.x)
- Python **3.12** (tested on 3.12.3)
- [Ollama](https://ollama.com) **0.6+** running locally, with the two models
  pulled: `ollama pull qwen3.5:4b` and `ollama pull llama3.2:3b`
- Python packages pinned in `requirements.txt`:
  `ollama`, `pandas`, `numpy`, `scipy`, `matplotlib`

**Hardware**

- No GPU is required, but one is strongly recommended: the models are 3B–4B
  parameters and run on CPU at a large latency cost.
- At least **8 GB of RAM** and roughly **6 GB of disk** for the two models.
- No unconventional peripherals. Total storage taken by this repository is
  under 1 MB; the full run produces a ~600 KB log.

**Runtime**

The full experiment (1,600 invocations) took about 2 hours on a consumer
machine with GPU. Reproducing only the analysis from the provided
`results.jsonl` takes a few seconds and requires no model and no GPU.

## Installation

```bash
git clone https://github.com/Wyllgner/When-the-Trigger-Fails.git
cd When-the-Trigger-Fails

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Check the installation (no model required)

Reproduce the paper's analysis from the logged results:

```bash
python analyze.py
```

Expected output — the tables reported in the paper, ending with the Wilcoxon
test and the regenerated `mvr.png`:

```
Lido results.jsonl: 1600 chamadas | modelos=['llama3.2:3b', 'qwen3.5:4b']

=== Tabela 1 — MVR (taxa de violacao) por modelo x MR ===
             viol_mr1 viol_mr2 viol_mr3
model
llama3.2:3b      7.4%     7.4%    40.7%
qwen3.5:4b      10.7%     3.6%    28.6%
...
=== RQ4 — Wilcoxon qwen3.5:4b vs llama3.2:3b ===
  n_pares=21 | stat=6.000 | p=0.0253
```

If those numbers appear, the artifact is correctly installed.

### 2. Check the agent setup (requires Ollama)

```bash
ollama serve            # in another terminal, if not already running
python smoke_test.py
```

Expected output: a non-empty list containing a `get_weather` tool call.

### 3. Re-run the experiment from scratch

```bash
python build_variations.py            # regenerates variations.jsonl (optional)
python run_experiment.py --pilot      # pilot: 1 model, 5 tasks, 100 calls
python run_experiment.py              # full experiment, overwrites results.jsonl
python run_experiment.py --resume     # resumes an interrupted run
python analyze.py                     # tables, Wilcoxon test and mvr.png
python inspect_violations.py --csv inspecao.csv   # qualitative inspection
```

> **Note.** `run_experiment.py` overwrites `results.jsonl`. Back up the provided
> file first if you want to keep the exact results reported in the paper.
> Because the models are non-deterministic, a new run will not reproduce the
> logged outputs verbatim; the reported trends (language swap dominating the
> violations, argument changes dominating the violation types) are what should
> be reproducible.

## Ethical and legal statement

The artifact contains no personal, sensitive or human-subject data. All 40
tasks are synthetic prompts written by the authors; the email addresses and
names appearing in them (e.g. `ana@x.com`) are fictitious. The tools are
declarations only — no tool is actually executed, so no external service is
contacted and no side effect is produced. Both models are publicly available
open-weight models distributed through Ollama and are subject to their own
licences.

## License

This artifact is released under the [MIT License](LICENSE), which covers both
the code and the accompanying data files.

## Citation

```bibtex
@inproceedings{franca2026trigger,
  title     = {Quando o Gatilho Falha: Teste Metam{\'o}rfico de Agentes de IA
               com Chamada de Ferramentas},
  author    = {Fran{\c c}a, Wyllgner},
  booktitle = {Simp{\'o}sio Brasileiro de Teste de Software Sistem{\'a}tico e
               Automatizado (SAST), CBSoft},
  year      = {2026}
}
```
