#!/bin/bash
# Window 2: plain TensorFold 0.6.0 (B) vs 0.6.0 + n-gram read-ahead (R), same recipe settings; then production (A).
# B R B R: prefill ladder twice per start; quick 1-user decode and the byte probe on round 1 (R also on round 2);
# agent resend test on B1 and A. Production (A, --max-tokens 32768) is left serving.
set -u
PROD=~/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
NEW=~/tf-eval/miatf-060
BENCH=~/tf-eval/bench; RES=~/tf-eval/results/ab_ra; LOGS=~/tf-eval/logs
M=Qwen3.8-Flash-Next; BASE=http://127.0.0.1:8888
mkdir -p $RES
say() { echo "== $(date +%H:%M:%S) $*"; }
up() { for i in $(seq 1 $(( $1 / 5 ))); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 localhost:8888/v1/models)" = "200" ] && return 0; sleep 5; done; return 1; }
start() {   # label dir image [args]
  local label=$1 dir=$2 image=$3; shift 3
  say "start $label ($image $*)"
  ( cd $dir && IMAGE=$image PREPARE=0 timeout 3000 ./start.sh restart --max-tokens 32768 "$@" ) > $LOGS/ab_ra_start_$label.log 2>&1 < /dev/null
  say "start.sh exit $?"
  docker logs qwen38-flash-next-tf 2>&1 | grep -E "estimate|serving|Error|error" | cut -c1-200 | sed 's/^/   /'
  up 60
}
ladder() { cd $BENCH; for rep in 1 2; do MODEL=$M python3 dash_bench.py --tag $1 --out $RES/dash.jsonl prefill --sizes 8192 16384 32768 65536 131072 2>&1 | tail -1; done; }
decode1() { cd $BENCH; MODEL=$M python3 dash_bench.py --tag $1 --out $RES/dash.jsonl decode --streams 1 --prompt prose code --repeats 3 2>&1 | tail -1; }
probe() { cd $BENCH; python3 exact_probe.py run --base $BASE --model $M --out $RES/probe_$1.json 2>&1 | tr '\n' ' '; echo; }
resend() { cd $BENCH; python3 resend_bench.py --base $BASE --model $M --tag $1 --out $RES/resend.jsonl --sizes 8192 32768 100000 2>&1 | cut -c1-160; }

say "AB_RA START"
for i in $(seq 1 120); do
  r=$(curl -s -m 5 localhost:8888/health | python3 -c "import json,sys; print(json.load(sys.stdin).get('requests_running', 1))" 2>/dev/null || echo 1)
  [ "$r" = "0" ] && break; sleep 5
done
if start B1 $NEW tensorfold-qwen38:v0.6.0; then ladder B1; decode1 B1; probe B1; resend B1; say "B1 done"; else say "B1 NOT UP"; fi
if start R1 $NEW tensorfold-qwen38:v0.6.0-ra; then ladder R1; decode1 R1; probe R1; say "R1 done"; else say "R1 NOT UP"; tail -20 $LOGS/ab_ra_start_R1.log; fi
if start B2 $NEW tensorfold-qwen38:v0.6.0; then ladder B2; say "B2 done"; else say "B2 NOT UP"; fi
if start R2 $NEW tensorfold-qwen38:v0.6.0-ra; then ladder R2; probe R2; say "R2 done"; else say "R2 NOT UP"; fi
if start A $PROD tensorfold-qwen38:v0.3.6.3; then resend A; probe A; say "A done"; else say "A NOT UP"; tail -20 $LOGS/ab_ra_start_A.log; fi
cd $BENCH
say "probe B1 vs R1: $(python3 exact_probe.py cmp $RES/probe_B1.json $RES/probe_R1.json)"
say "probe R1 vs R2: $(python3 exact_probe.py cmp $RES/probe_R1.json $RES/probe_R2.json)"
say "probe B1 vs A:  $(python3 exact_probe.py cmp $RES/probe_B1.json $RES/probe_A.json)"
docker inspect qwen38-flash-next-tf --format "{{.Config.Image}} {{join .Args \" \"}}" | grep -oE "^[^ ]+|--max-tokens [0-9]+|--vision"
say "AB_RA DONE"
