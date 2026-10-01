#!/bin/bash
# A/B on the Spark, same port and settings (5 x 262,144, int8 KV, n-gram tables on SSD, MTP 6/0.60, --max-tokens 32768):
#   A = production: MiaAI recipe, TensorFold 0.3.6.3 + patches (vision on, as served)
#   B = plain TensorFold 0.6.0 through the recipe's scripts (branch tf-0.6.0, text only)
#   C = B with --prefill-fp8 (0.6.0 made bf16 prompts the default)
# Order B C A B C A, fresh start each; sparkDash decode (1/2/5 streams, prose+code, 3 reps) and prefill ladder.
# Production (A) is left serving at the end.
set -u
PROD=~/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
NEW=~/tf-eval/miatf-060
BENCH=~/tf-eval/bench; RES=~/tf-eval/results/ab060; LOGS=~/tf-eval/logs
M=Qwen3.8-Flash-Next
mkdir -p $RES
say() { echo "== $(date +%H:%M:%S) $*"; }
up() { for i in $(seq 1 $(( $1 / 5 ))); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 localhost:8888/v1/models)" = "200" ] && return 0; sleep 5; done; return 1; }
start() {   # label dir [serve args]
  local label=$1 dir=$2; shift 2
  say "start $label ($dir $*)"
  ( cd $dir && PREPARE=0 timeout 3000 ./start.sh restart --max-tokens 32768 "$@" ) > $LOGS/ab060_start_$label.log 2>&1 < /dev/null
  say "start.sh exit $?"
  docker logs qwen38-flash-next-tf 2>&1 | grep -E "estimate|streams of|serving|Error|error" | cut -c1-260 > $RES/server_$label.txt
  cat $RES/server_$label.txt | sed 's/^/   /'
  up 60
}
suite() {   # tag
  cd $BENCH
  free -g | sed -n 2p | sed "s/^/   $1 mem: /"
  MODEL=$M python3 dash_bench.py --tag $1 --out $RES/dash.jsonl decode --streams 1 2 5 --prompt prose code --repeats 3 2>&1 | tail -2
  MODEL=$M python3 dash_bench.py --tag $1 --out $RES/dash.jsonl prefill --sizes 8192 16384 32768 65536 131072 256000 2>&1 | tail -2
  say "suite $1 done"
}

say "AB060 START"
for i in $(seq 1 120); do
  r=$(curl -s -m 5 localhost:8888/health | python3 -c "import json,sys; print(json.load(sys.stdin).get('requests_running', 1))" 2>/dev/null || echo 1)
  [ "$r" = "0" ] && break; sleep 5
done
for round in 1 2; do
  if start B$round $NEW; then suite B$round; else say "B$round NOT UP"; tail -30 $LOGS/ab060_start_B$round.log; fi
  if start C$round $NEW --prefill-fp8; then suite C$round; else say "C$round NOT UP"; tail -30 $LOGS/ab060_start_C$round.log; fi
  if start A$round $PROD; then suite A$round; else say "A$round NOT UP"; tail -30 $LOGS/ab060_start_A$round.log; fi
done
say "production: $(curl -s -m 5 localhost:8888/v1/models | head -c 100)"
docker inspect qwen38-flash-next-tf --format "{{.Config.Image}} {{join .Args \" \"}}" | grep -oE "^[^ ]+|--max-tokens [0-9]+|--vision"
say "AB060 DONE"
