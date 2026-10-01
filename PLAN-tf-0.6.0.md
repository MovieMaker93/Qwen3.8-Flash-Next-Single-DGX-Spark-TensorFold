# This recipe on TensorFold 0.6.0: plan, results so far, next steps

Branch `tf-0.6.0` (fork of MiaAI-Lab's recipe). Goal: move the recipe from TensorFold v0.3.6.3 to v0.6.0, port
the patches that still pay off, find new speed-ups, and send them upstream (engine changes to
[ashhart/TensorFold](https://github.com/ashhart/TensorFold), recipe changes to MiaAI-Lab). Rules: outputs
byte-identical where a change claims "speed only", defaults never regress (trade-offs are opt-in), A/B measured
alternating (A B A B, fresh start each), on one DGX Spark, at the recipe's settings (5 x 262,144, int8 KV,
n-gram tables on SSD, MTP 6/0.60).

Status on 2026-10-01: text only (vision not ported yet). Production still runs the 0.3.6.3 recipe.

## What this branch changes

| File | Change |
| --- | --- |
| `scripts/config.sh` | `TF_VERSION` default `v0.6.0`; `patch_files` lists nothing (instead of failing) when `patches/` has no `.patch` |
| `scripts/prepare.sh` | the image check no longer imports `tensorfold.vision.videos` (patch 0008 adds it; put it back with the vision port) |
| `patches/v0.3.6.3/` | MiaAI's nine patches for 0.3.6.3, kept as references for porting (not applied) |
| `patches/0002-flash-next-ssd-read-ahead.patch` | read-ahead ported to 0.6.0 (applied; see results) |
| `patches/candidates/0007-flash-next-copy-drafts-adaptive.patch` | copy drafts ported to 0.6.0 + adaptive gating (not applied; not yet measured) |
| `dev/tf-0.6.0/` | the A/B, probe and quality scripts used below |

Local `.env` used for these builds (not committed; keeps 0.6.0's kernels and build marker apart from production):

```
KERNEL_CACHE=/home/<you>/.cache/tensorfold-qwen38-060
VISION=0
```

Build and run (production can keep serving while an image builds; a run needs production stopped):

```bash
PULL=0 scripts/prepare.sh                                            # tensorfold-qwen38:v0.6.0 with patches/*.patch
IMAGE=tensorfold-qwen38:v0.6.0-x PULL=0 scripts/prepare.sh           # a variant under its own tag
IMAGE=tensorfold-qwen38:v0.6.0-x PREPARE=0 ./start.sh restart --max-tokens 32768
```

## Patch status against 0.6.0

| 0.3.6.3 patch | On 0.6.0 |
| --- | --- |
| 0001 live token counters | upstream has Prometheus `/metrics` and lane stats in `/health`; check before porting |
| 0002 SSD read-ahead | **ported** (`patches/0002`): a lone prompt reads its next chunk's n-gram rows while the GPU runs this one |
| 0003 native SSD reader | not ported; with read-ahead the reads are hidden, so probably unnecessary (check decode staging) |
| 0004 QSA tiled select | upstream (#93) |
| 0005 draft stats | upstream counts concurrent drafted rows; check |
| 0006 prefill rows | not ported; 0.6.0 fixes `PREFILL_ROWS = 2048` (the recipe uses 2048 with vision anyway) |
| 0007 copy drafts | **ported + adaptive** (candidate, see below) |
| 0008/0009 vision | **not ported**: the largest piece; required before this can replace the 0.3.6.3 recipe |
| 0010 draft languages | not ported |
| local 0100 reasoning_effort | upstream since 0.5.0 |
| local 0101 Python-spelled tool args | upstream in 0.6.0 (4d9f241) |

`--prefill-fp8` does not apply: this checkpoint has no FP8 prompt kernels (0.6.0 refuses the flag).

## Results so far (Spark, 2026-10-01)

**A = production** (recipe on 0.3.6.3 + patches, vision on) vs **B = plain 0.6.0** (this branch, no patches),
sparkDash, two rounds each, tok/s:

| | A | B | B vs A |
| --- | ---: | ---: | ---: |
| 1 user prose / code | 69.2-69.4 / 104.1-104.3 | 69.4-69.9 / 100.0-100.6 | 0 / -3.6% |
| 2 users prose / code | 93.2-93.3 / 108.5 | 87.9-90.4 / 132.9-133.7 | -3 to -6% / +23% |
| 5 users prose / code | 130.3-130.9 / 135.0-136.8 | 145.7-146.5 / 192.3-194.1 | +12% / +41% |
| short-prompt TTFT, 1 user | 143-185 ms | 59-71 ms | 2.5x faster |
| sparkDash prefill 8k-65k / 131k / 256k | | | -1.5% / -1% / equal |
| memory allocated at start | 102.50 GiB | 84.26 GiB (caches grow) | |

Follow-up turns (agent resends) reuse the cached prompt on both: 0.14-0.27 s at 10k-120k tokens. No difference.

**Read-ahead (R = B + patch 0002)**, debug build with per-pass timings (prompt passes of 2,048 rows):

| | read-ahead off | read-ahead on |
| --- | ---: | ---: |
| hits / misses | 0 / 165 | 158 / 7 |
| median pass | 1,049 ms | 888 ms |
| share of pass time staging n-gram rows | 14.1% | 0.6% |
| chat first turn, ~39k tokens | 22.5 s | 17.0 s (A: 16.1 s) |
| chat first turn, ~120k tokens | 74.6 s | 57.1 s (A: 56.0 s) |

sparkDash's prefill prompts need ~3-5 ms of n-gram reads a pass (so its ladder shows no change); varied text needs
~300 ms a pass, all of it on the critical path without read-ahead. Outputs: byte-identical (6/6 probe prompts up
to 60k tokens, two rounds). Decode unchanged.

**Copy drafts (CP = B + MiaAI's 0007 ported, always on)**, two rounds:

| | B | CP | CP vs B |
| --- | ---: | ---: | ---: |
| edit test (reply repeats a 170-line file) | 91.2-91.4 | 127.1-127.8 | +40% (A: 131) |
| code 1 / 2 / 5 users | 100 / 133 / 194 | 97-98 / 111 / 139-141 | -2% / -16% / -28% |
| prose 5 users | 145.7 | 139.5-142.9 | -2 to -4% |

Same replies (edit reply hash equal on A, B, CP; probe 6/6). In code, the last 8 tokens often match earlier text but
the continuation differs: 6 copied drafts a round replace the MTP chain and get rejected, and with 5 users every
round carries those rows. A's weak multi-user code numbers (108.5, 135-137) are this. The candidate patch gates
copies per stream: a moving average of the share of copied drafts kept; below `TENSORFOLD_MTP_COPY_GATE` (default
0.5) the stream drafts with MTP and the score recovers 0.05 per skipped copy. `TENSORFOLD_MTP_COPY_GATE=0` is the
always-copy behaviour.

Bits: B and A give different replies on 2 of 6 probe prompts (the short ones); B, R and CP are identical to each
other. So quality must be compared (running tonight, below).

## Running tonight

`dev/tf-0.6.0/quality_night.sh`: R (0.6.0 + read-ahead) then A, MMLU-Pro 350 (25/category), GSM8K 400, HumanEval 164,
thinking on (effort medium), per-item seeds, 5 at a time; production left serving. Results in
`~/tf-eval/results/q060/`, log `~/tf-eval/logs/q060.log` (MMLU-Pro and GSM8K are scored at the end). HumanEval:

```bash
cd ~/tf-eval/bench
../venv/bin/python he_run.py ../results/q060/he_recipe      # sandboxed run of the generated programs
../venv/bin/python he_run.py ../results/q060/he_tf060ra
../venv/bin/python score.py he-report ../results/q060/he_recipe/results.json ../results/q060/he_tf060ra/results.json
../venv/bin/python score.py paired <A per-item json> <B per-item json>                 # McNemar on each task
```

Pass: no task differs beyond noise (paired test p > 0.05; earlier runs showed +-0.2 NLL swings on single documents,
so judge on the whole sets).

## Next steps

1. **Quality** (morning): score the night run as above. If 0.6.0 is equal, the version switch is safe quality-wise.
2. **Read-ahead, before its TensorFold PR:**
   - re-measure first turns on real text (source files, docs), B vs R, alternating;
   - cover prompts that arrive while other streams decode (`multi.py` `round()`, the mixed pass): there the next
     piece's size adapts to timing, so exact-key hits are rare. Design: `ReadAhead` also serves a lookup whose ids
     are a **prefix** of a pending read (n-gram ids of row i depend only on earlier tokens, checked in
     `cuda/ngram.py` `ids`), and `round()` reads ahead `prompt[a+n : a+n+prefill_rows]` for its first pieces;
   - then the PR to TensorFold, crediting MiaAI-Lab's recipe patch 0002 (their idea).
3. **Adaptive copy drafts:** A/B B vs CPA (gate 0.5) vs CPA with `TENSORFOLD_MTP_COPY_GATE=0` (= always copy),
   sparkDash 1/2/5 users + `edit_bench.py` + probe; try gates 0.3 / 0.7. Target: >= +30% on edits, <= 1% loss on
   multi-user code. If it holds: TensorFold PR (opt-in `TENSORFOLD_MTP_COPY=1`), credit MiaAI-Lab's 0007.
4. **Remaining gaps vs A:** code 1 user -3.6% and prose 2 users -3 to -6% (not copy drafts: CP is lower too).
   Profile a decode round (kernel time by name, GPU-busy share) on A vs B.
5. **Native reader (0003):** with read-ahead the prompt reads are hidden; check whether decode rounds still spend time
   staging n-gram rows (debug counters below), and port only if they do.
6. **Vision (0008/0009)** on 0.6.0: needed before the recipe can move to 0.6.0. Then put
   `tensorfold.vision.videos` back in the `prepare.sh` check.
7. **PR to MiaAI-Lab:** the recipe on 0.6.0 (ported patches, the `patch_files` fix, README tables re-measured).
8. **Production switch:** only when quality passes, vision works, and no default regresses.

## Scripts (`dev/tf-0.6.0/`, copies of `~/tf-eval/bench` on the Spark)

| Script | What |
| --- | --- |
| `ab060.sh`, `ab_ra.sh`, `ab3.sh` | the three GPU windows above: start each build with this repo's `start.sh`, run, restore production |
| `dash_bench.py`, `dash_summary.py` | sparkDash decode (1/2/5 streams, prose/code) and prefill ladder; medians |
| `exact_probe.py` | 6 fixed greedy prompts (short to 60k tokens); `cmp` gives byte-identical counts across builds |
| `resend_bench.py` | agent-style turns: first turn cold, then two resends of the growing conversation |
| `edit_bench.py` | edit-style decode; needs `edit_src.py` next to it: `git -C <TensorFold clone> show v0.6.0:src/tensorfold/families/qwen4_exp/cuda/multi.py \| sed -n 1,170p > edit_src.py` |
| `quality_night.sh` | the quality run above |

Debug counters (not in the patches): `TENSORFOLD_RA_DEBUG=1` printed per prompt pass `stage_ms`, `pass_ms` and
read-ahead hits; `TENSORFOLD_RA_OFF=1` turned read-ahead off in the same build. The instrumented TensorFold branch
is `ra-debug` in `~/tf-eval/TensorFold-radbg` on the Spark.

## Where things are on the Spark

| What | Where |
| --- | --- |
| this branch | `~/tf-eval/miatf-060` (worktree of the recipe clone) |
| production recipe | `~/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` (main + local patches 0100, 0101) |
| TensorFold branches (on v0.6.0) | `~/tf-eval/TensorFold-ra` (`ngram-read-ahead`), `~/tf-eval/TensorFold-cp` (`copy-drafts`, `copy-adaptive`) |
| images | `tensorfold-qwen38:v0.6.0` (plain), `-ra`, `-cp`, `-cpa`, `-radbg`; production `v0.3.6.3` (rollback `v0.3.6.3-pre0101`) |
| results | `~/tf-eval/results/ab060`, `ab_ra`, `ab3`, `q060` |
