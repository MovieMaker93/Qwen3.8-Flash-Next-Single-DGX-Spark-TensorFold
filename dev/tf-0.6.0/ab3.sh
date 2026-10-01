#!/bin/bash
# Window 3. (1) Diagnostics on plain 0.6.0 + read-ahead, debug build: per prompt pass, staging ms vs pass ms and
# read-ahead hits (TENSORFOLD_RA_DEBUG=1), read-ahead off (D0) then on (D1): ladder + resend first turns.
# (2) Copy drafts: B (plain 0.6.0) vs CP (0.6.0 + copy drafts, TENSORFOLD_MTP_COPY=1 as the recipe sets), B CP B CP:
# sparkDash decode 1/2/5, edit test, byte probe. (3) Production (A) back, with the edit test for reference.
set -u
PROD=~/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
NEW=~/tf-eval/miatf-060
BENCH=~/tf-eval/bench; RES=~/tf-eval/results/ab3; LOGS=~/tf-eval/logs
M=Qwen3.8-Flash-Next; BASE=http://127.0.0.1:8888
mkdir -p $RES
say() { echo "== $(date +%H:%M:%S) $*"; }
up() { for i in $(seq 1 $(( $1 / 5 ))); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 localhost:8888/v1/models)" = "200" ] && return 0; sleep 5; done; return 1; }
start() {   # label dir image [args]
  local label=$1 dir=$2 image=$3; shift 3
  say "start $label ($image $*)"
  ( cd $dir && IMAGE=$image PREPARE=0 timeout 3000 ./start.sh restart --max-tokens 32768 "$@" ) > $LOGS/ab3_start_$label.log 2>&1 < /dev/null
  say "start.sh exit $?"
  up 60
}
ladder() { cd $BENCH; MODEL=$M python3 dash_bench.py --tag $1 --out $RES/dash.jsonl prefill --sizes 8192 32768 131072 2>&1 | tail -1; }
decode() { cd $BENCH; MODEL=$M python3 dash_bench.py --tag $1 --out $RES/dash.jsonl decode --streams 1 2 5 --prompt prose code --repeats 3 2>&1 | tail -1; }
edit() { cd $BENCH; python3 edit_bench.py --base $BASE --model $M --tag $1 --out $RES/edit.jsonl --reps 2 2>&1 | cut -c1-160; }
probe() { cd $BENCH; python3 exact_probe.py run --base $BASE --model $M --out $RES/probe_$1.json > /dev/null 2>&1; }
resend() { cd $BENCH; python3 resend_bench.py --base $BASE --model $M --tag $1 --out $RES/resend.jsonl --sizes 32768 100000 2>&1 | cut -c1-140; }
dbglog() { docker logs qwen38-flash-next-tf 2>&1 | grep "ra-debug" > $RES/radebug_$1.log; say "$1: $(wc -l < $RES/radebug_$1.log) pass lines; last: $(tail -1 $RES/radebug_$1.log | cut -c1-200)"; }

say "AB3 START"
for i in $(seq 1 120); do
  r=$(curl -s -m 5 localhost:8888/health | python3 -c "import json,sys; print(json.load(sys.stdin).get('requests_running', 1))" 2>/dev/null || echo 1)
  [ "$r" = "0" ] && break; sleep 5
done
export TENSORFOLD_RA_DEBUG=1
if TENSORFOLD_RA_OFF=1 start D0 $NEW tensorfold-qwen38:v0.6.0-radbg; then ladder D0; resend D0; dbglog D0; say "D0 done"; else say "D0 NOT UP"; tail -20 $LOGS/ab3_start_D0.log; fi
if start D1 $NEW tensorfold-qwen38:v0.6.0-radbg; then ladder D1; resend D1; dbglog D1; say "D1 done"; else say "D1 NOT UP"; fi
unset TENSORFOLD_RA_DEBUG
for round in 1 2; do
  if start B$round $NEW tensorfold-qwen38:v0.6.0; then decode B$round; edit B$round; [ $round = 1 ] && probe B1; say "B$round done"; else say "B$round NOT UP"; fi
  if start CP$round $NEW tensorfold-qwen38:v0.6.0-cp; then decode CP$round; edit CP$round; [ $round = 1 ] && probe CP1; say "CP$round done"; else say "CP$round NOT UP"; tail -20 $LOGS/ab3_start_CP$round.log; fi
done
if start A $PROD tensorfold-qwen38:v0.3.6.3; then edit A; say "A done"; else say "A NOT UP"; tail -20 $LOGS/ab3_start_A.log; fi
cd $BENCH
say "probe B1 vs CP1: $(python3 exact_probe.py cmp $RES/probe_B1.json $RES/probe_CP1.json)"
docker inspect qwen38-flash-next-tf --format "{{.Config.Image}} {{join .Args \" \"}}" | grep -oE "^[^ ]+|--max-tokens [0-9]+|--vision"
say "AB3 DONE"
