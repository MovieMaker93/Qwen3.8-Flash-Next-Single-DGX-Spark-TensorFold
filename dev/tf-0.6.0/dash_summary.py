#!/usr/bin/env python3
"""Medians per server from the sparkDash runs: decode (aggregate, per stream, TTFT) and the prefill ladder."""
import json, statistics, sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for path in sys.argv[2:]:                       # sweep.py rows (vLLM, current config) -> same shape
    for l in open(path):
        r = json.loads(l)
        rows.append({"tag": "vllm_cur", "kind": "decode", "prompt": r["prompt"], "S": r["S"], "rep": r["rep"],
                     "aggregate_tps": r["dash_aggregate_tps"], "mean_tps": r["dash_mean_tps"], "ttft_ms": r["dash_ttft_ms"],
                     "decode_tokens": None, "tok_per_step": r.get("tok_per_step"), "ms_per_step": r.get("ms_per_step")})

dec = defaultdict(list)
pre = {}
for r in rows:
    if r["kind"] == "decode":
        dec[(r["tag"], r["prompt"], r["S"])].append(r)
    elif r.get("status") == "completed":
        pre[r["tag"]] = {x["promptTokens"]: x for x in r["job"].get("results", [])}

out = {"decode": {}, "prefill": {}}
med = lambda xs: round(statistics.median(xs), 2)
print(f"{'tag':<20}{'prompt':<7}{'S':>2}{'agg med':>9}{'agg range':>14}{'per-stream med':>16}{'ttft med ms':>12}{'tokens':>18}{'tok/step':>9}")
for (tag, prompt, s), rs in sorted(dec.items()):
    agg = [r["aggregate_tps"] for r in rs]; per = [r["mean_tps"] for r in rs]; tt = [r["ttft_ms"] for r in rs]
    toks = sorted({t for r in rs for t in (r.get("decode_tokens") or [])})
    tps = [r["tok_per_step"] for r in rs if r.get("tok_per_step")]
    out["decode"][f"{tag}|{prompt}|{s}"] = {"agg": med(agg), "agg_min": min(agg), "agg_max": max(agg),
                                             "per": med(per), "ttft_ms": med(tt), "n": len(rs), "tokens": toks}
    print(f"{tag:<20}{prompt:<7}{s:>2}{med(agg):>9}{f'{min(agg):.1f}-{max(agg):.1f}':>14}{med(per):>16}{med(tt):>12.0f}"
          f"{str(toks):>18}{(med(tps) if tps else ''):>9}")
print()
for tag, res in sorted(pre.items()):
    out["prefill"][tag] = {str(k): {"ttft_s": round(v["ttftMs"] / 1000, 2), "tps": round(v["prefillTps"])} for k, v in sorted(res.items())}
    print(f"{tag:<20}" + "  ".join(f"{k // 1000}k {v['ttftMs'] / 1000:.1f}s ({v['prefillTps']:.0f}/s)" for k, v in sorted(res.items())))
json.dump(out, open("dash_summary.json", "w"), indent=1)
