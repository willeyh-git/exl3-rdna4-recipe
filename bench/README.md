# Benchmarks and checks

Run most scripts from the repo root with the venv active (`source setup/env.sh`). The MoE context Bash launcher locates
the repo and activates the venv itself. Results go to `logs/` and `gpqa/`.

## Against the running server

These talk to TabbyAPI over its OpenAI-compatible API. They default to `http://127.0.0.1:8096/v1` and the first
`api_key` in `tabbyAPI/api_tokens.yml`; set `TABBY_URL` and `TABBY_API_KEY` (or `TABBY_TREE` for another TabbyAPI
checkout) to point elsewhere. Every script exits nonzero if a request fails or returns no text. They count tokens with the
model's tokenizer (`models/Qwen3.8-27B-EXL3-SC3.0bpw-H4/tokenizer.json`, or `MODEL_DIR`).

| Script | What it does |
|---|---|
| `bench_api.py NAME [URL] [KEY]` | Greedy code (600 tokens), sampled prose (600 tokens), prefill at 8k and 30k. Engine-neutral: works against llama-server too. |
| `run_moe_context_bench.sh LABEL [--contexts 2 8 16 32] [--repeats 3]` | Activates the project environment and measures prefill/decode speed and draft acceptance at prompt targets of 2k, 8k, 16k, and 32k by default. Appends one JSONL record per request to `logs/moe_context_bench.jsonl`; set `MODEL_DIR` to the served model. |
| `context_bench.py LABEL 8 32 60 ...` | Decode speed and recall by depth: a fresh filler document per depth with a code buried mid-way, then ~450 tokens of Python. This is the decode-by-depth table. |
| `ctx_probe.sh LABEL CTX_K [args]` | Restarts TabbyAPI with a context size and extra args, then runs `context_bench.py` near the top. Reports VRAM via `amd-smi`. Stops this repo's TabbyAPI first (`serve/stop_tabby.sh`), gives up after `START_TIMEOUT` seconds, and stops the server it started if anything fails. |
| `needle_test.py [N]` | Quick long-context recall check: a code buried in N filler sections (default 1400). |
| `gpqa_eval.py NAME [URL] [KEY]` | GPQA Diamond, first 50 questions (`GPQA_LIMIT`), thinking on, 30k token cap (`GPQA_MAX_TOKENS`), temperature 1.0. Resumable. `GPQA_EFFORT=medium GPQA_SYSTEM=antispiral` gives the recommended settings. `GPQA_IDS=1,12,...` asks only those questions, with the same shuffled choices as a full run; `GPQA_WORKERS=N` keeps N in flight. |
| `run_gpqa_all.sh` | Starts a server per model (SC 3.0 bpw H4, then the plain 3.0 bpw and 2.5 bpw if present), checks it serves that model, runs `gpqa_eval.py`, stops it. Refuses to start if something already answers on `TABBY_URL`. |
| `gpqa_resume.sh` | Restarts `run_gpqa_all.sh` after a crash or reboot. |
| `overthink_test.py VARIANT` | The 11 GPQA questions that spiralled, under one effort/system-prompt variant. This is how the anti-spiral prompt was picked. |

### Ornith MoE speed sweep

Start the Ornith server with enough context for the 32k prompt and leave it running. `CTX=32k` sets a 32k cache and a
32,512-token sequence limit; the sweep leaves room for the chat template and its default 128-token completion. This
larger cache may not fit on every GPU with this 24 GB-labeled quant and CPU expert split. If it does not load, try
`CTX=16k` with `--contexts 2 8 16`; if that also does not fit, use the default 8k profile with `--contexts 2 8`.

```bash
CTX=32k serve/run_tabby_ornith.sh
```

In another terminal, from the repo root, use Ornith's tokenizer and run three requests at each default context, including
32k:

```bash
MODEL_DIR=models/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB bench/run_moe_context_bench.sh ornith-baseline --repeats 3
```

Each JSON line records the target and actual prompt token counts, time to first token, estimated prefill and decode
tokens/second, generated length, draft tokens accepted/rejected/total, and `draft_acceptance_pct`. The acceptance
percentage is accepted draft tokens divided by total draft tokens; it is `null` when no draft tokens were produced or
the API did not return usage details. Results append to `logs/moe_context_bench.jsonl`; use a new label for each
configuration or date. The Bash launcher sources `setup/env.sh` and runs the benchmark with the project
virtual environment, so run `setup/make_venv.sh` first if it is not set up. The benchmark only sends requests to the
already-running API: it does not start or restart the server or compile the engine. The context targets are a useful
initial sweep, not a complete performance profile.

### GPQA data

GPQA is gated on Hugging Face and its authors ask that it not be reposted in plain text, so it isn't in this repo.
Accept the terms at https://huggingface.co/datasets/Idavidrein/gpqa, log in with `hf auth login`, then:

```bash
hf download Idavidrein/gpqa gpqa_diamond.csv --repo-type dataset --local-dir gpqa
```

The `xhigh` rerun in `notes/measurements.md` asked the questions that either 198-question `medium` pass got wrong:

```bash
IDS=$(python3 -c 'import json, sys
wrong = {r["idx"] for f in sys.argv[1:] for r in map(json.loads, open(f)) if not r["correct"]}
print(",".join(map(str, sorted(wrong))))' gpqa/results_PASS1.jsonl gpqa/results_PASS2.jsonl) &&
GPQA_LIMIT=198 GPQA_IDS=$IDS GPQA_EFFORT=xhigh GPQA_SYSTEM=antispiral GPQA_MAX_TOKENS=60000 GPQA_WORKERS=2 \
    python bench/gpqa_eval.py xhigh-retry
```

If the selection fails, the `&&` stops there; an empty `GPQA_IDS` is refused rather than running all 198.

## Direct model and kernel tests (GPU, no server)

These load the model or call kernels directly, so stop TabbyAPI first (`serve/stop_tabby.sh`); there isn't room for
two copies in 16 GB.

| Script | What it does |
|---|---|
| `smoke_test.py [--draft DIR \| --mtp] [--long]` | Load, generate, print TTFT and decode speed. The first thing to run after a build. |
| `test_wmma_model.py` | Full-model check of `EXL3_GEMV_WMMA`: logits of an 8/4/2-token forward with and without it (max logit diff, KL, top-1 agreement). |
| `test_wmma_gemv.py` | WMMA vs fdot2 multi-row GEMV per layer shape: relative error and µs per call. |
| `bench_gemv.py [--rows 1 8]` | Achieved weight bandwidth of each EXL3 linear (µs and GB/s, % of 640 GB/s). |
| `bench_decode_attn.py [depth_k] [q_len]` | Split-decode attention on a synthetic Q4 cache, over the `EXL3_DEC_*` launch configs; checks every output against the default. |
| `bench_batch_sanity.py [--bsz 2 4] [--out FILE]` | Diagnostic report: runs N different prompts at once and, per job, shows the common prefix with the same prompt run alone, whether it is identical, its non-ASCII share and longest repeated-token run. Fails only on generator errors, unfinished or empty jobs, or if the N jobs never actually decoded together. `--out` writes everything as JSON, also on failure. Speeds it prints are generator-only (no API). |
| `vram_breakdown.py` | Allocated VRAM per component (draft, main weights, KV) and the biggest tensors. |

The GPU tests that ship inside the exllamav3 patch (`test_gemv32.py`, `test_greedy_equal.py`) default to
`models/Qwen3.8-27B-EXL3-3.0bpw`, the plain quant they were written against; pass
`--model ../models/Qwen3.8-27B-EXL3-SC3.0bpw-H4` (from the `exllamav3/` checkout) to run them on the current default:

```bash
source setup/env.sh && cd exllamav3 && PYTHONPATH=$PWD python test_gemv32.py --model ../models/Qwen3.8-27B-EXL3-SC3.0bpw-H4 --draft
```

They need the extension built in place in that checkout (`setup/build_exllamav3.sh` does that). The previous base's
patch also had `test_gdn_replay.py`, the A/B test for our DeltaNet replay, which upstream's replay has replaced.
Upstream's own tests cover it: `tests/kernels/recurrent` and `tests/e2e/test_recurrent_spec_rewind.py` (the e2e test
needs a model registered with the "recurrent" tag; see `tests/models.yaml`).
