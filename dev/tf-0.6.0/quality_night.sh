#!/bin/bash
# Quality, same items and seeds on both: TensorFold 0.6.0 + read-ahead (R, branch tf-0.6.0, text only) then the
# production recipe (A, TensorFold 0.3.6.3 + patches), each at the recipe's settings (5 x 262,144, int8 KV).
# MMLU-Pro 25/category (350), GSM8K 400, HumanEval 164; thinking on (effort medium), sampled with per-item seeds;
# 5 requests at a time. MMLU-Pro and GSM8K scored here (paired test too); HumanEval generated, run in the morning.
# Production (A, --max-tokens 32768) is left serving.
set -u
PROD=~/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
NEW=~/tf-eval/miatf-060
BENCH=~/tf-eval/bench; RES=~/tf-eval/results/q060; LOGS=~/tf-eval/logs
M=Qwen3.8-Flash-Next; BASE=http://127.0.0.1:8888; PY=~/tf-eval/venv/bin/python
mkdir -p $RES
say() { echo "== $(date '+%m-%d %H:%M:%S') $*"; }
up() { for i in $(seq 1 $(( $1 / 5 ))); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 localhost:8888/v1/models)" = "200" ] && return 0; sleep 5; done; return 1; }
start() {   # label dir image
  say "start $1 ($3)"
  ( cd $2 && IMAGE=$3 PREPARE=0 timeout 3000 ./start.sh restart --max-tokens 32768 ) > $LOGS/q060_start_$1.log 2>&1 < /dev/null
  say "start.sh exit $?"; up 60
}
quality() {
  cd $BENCH
  for t in "mmlu_pro --n-per-cat 25" "gsm8k --n 400" "humaneval --n 164"; do
    set -- $t
    say "$LABEL $1"
    $PY quality.py --base $BASE --model $M --task $t --concurrency 5 --out $RES/${LABEL}_$1.jsonl > $LOGS/q060_${LABEL}_$1.log 2>&1
    say "$LABEL $1 exit $? ($(wc -l < $RES/${LABEL}_$1.jsonl) items)"
  done
}

say "Q060 START"
for i in $(seq 1 120); do
  r=$(curl -s -m 5 localhost:8888/health | python3 -c "import json,sys; print(json.load(sys.stdin).get('requests_running', 1))" 2>/dev/null || echo 1)
  [ "$r" = "0" ] && break; sleep 5
done
if start R $NEW tensorfold-qwen38:v0.6.0-ra; then LABEL=tf060ra quality; else say "R NOT UP"; tail -20 $LOGS/q060_start_R.log; fi
if start A $PROD tensorfold-qwen38:v0.3.6.3; then LABEL=recipe quality; else say "A NOT UP"; tail -20 $LOGS/q060_start_A.log; fi
cd $BENCH
say "scores"
$PY score.py mmlu_pro $RES/recipe_mmlu_pro.jsonl $RES/tf060ra_mmlu_pro.jsonl 2>&1 | tail -6
$PY score.py gsm8k $RES/recipe_gsm8k.jsonl $RES/tf060ra_gsm8k.jsonl 2>&1 | tail -6
$PY score.py he-build $RES/recipe_humaneval.jsonl $RES/he_recipe 2>&1 | tail -2
$PY score.py he-build $RES/tf060ra_humaneval.jsonl $RES/he_tf060ra 2>&1 | tail -2
say "production: $(curl -s -m 5 localhost:8888/v1/models | head -c 90)"
docker inspect qwen38-flash-next-tf --format "{{.Config.Image}} {{join .Args \" \"}}" | grep -oE "^[^ ]+|--max-tokens [0-9]+|--vision"
say "Q060 DONE"
