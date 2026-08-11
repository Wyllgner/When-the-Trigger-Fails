# Execution environment

Record of the environment in which the 1,600 invocations reported in the paper
were produced.

## Models

| Tag | Parameters | Served by |
|---|---|---|
| `qwen3.5:4b` | 4B | Ollama, locally |
| `llama3.2:3b` | 3B | Ollama, locally |

> **Limitation.** The manifest digests of the two models were **not captured at
> the time of the experiment**, and the weights are no longer installed on the
> host, so they cannot be recovered after the fact. The models are therefore
> identified here by their Ollama tag only. Because a tag is mutable, a
> replication performed later may pull weights that differ from the ones
> measured here. We record this openly rather than reporting a digest we did
> not observe.
>
> Anyone replicating the study should run `python env_report.py`, which writes
> this file with the digests of the models actually used, so that their run is
> pinned even though ours is not.

## When it was run

| Event | Date |
|---|---|
| Pilot run (1 model, 5 tasks) | 2026-06-29 |
| Full run, 1,600 invocations | 2026-06-30 |

The full run took 8,842 s (about 2 h 27 min) end to end, as recorded in
`run.log`.

## Decoding parameters

| Parameter | Value |
|---|---|
| temperature | 0.7 |
| seed | repetition index (0 to 4) |
| repetitions per case | 5 |
| tool exposure | function schema (`tools.json`), never inlined in the prompt |
| system prompt | `You are a helpful assistant with access to tools. Use exactly one tool to fulfill the user's request.` |

These values are set in `run_experiment.py` and were not varied across
conditions.

## Host

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 Mobile / Max-Q (4 GB VRAM) |
| OS | Linux x86_64, glibc 2.39 |
| Python | 3.12 |

The Ollama server version in use on 2026-06-30 was not recorded. The client
library is pinned in `requirements.txt`.

## Python packages

Pinned in `requirements.txt`:

```
ollama==0.6.2
pandas==3.0.3
numpy==2.5.0
scipy==1.18.0
matplotlib==3.11.0
```
