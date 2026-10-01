#!/usr/bin/env python3
"""sparkDash benches (decode sweep + prefill ladder) against whatever serves :8888, any engine.

The decode part is MiaAI-Lab's bench/sweep.py protocol (one sparkDash decode job per concurrency level, levels
alternated per repeat, 600 tokens, temperature 0, thinking off); vLLM's /metrics counters are recorded when the
server has them and skipped when it does not (TensorFold). Every reply's token count is kept, because sparkDash
asks for min_tokens and not every server honours it. One JSON line per job is appended to --out.

    python3 dash_bench.py --tag vllm_cur decode --streams 1 2 --prompt prose code --repeats 3
    python3 dash_bench.py --tag vllm_cur prefill --sizes 8192 16384 32768 65536 131072 256000
"""
import argparse, json, os, re, sys, threading, time, urllib.error, urllib.request

BASE = "http://localhost:5555/api/sparks/spark1/llm"
METRICS = "http://localhost:8888/metrics"
LINE = re.compile(r'^(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.eE+-]+)$')
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")          # the served model id
COUNTERS = ["vllm:inter_token_latency_seconds_sum", "vllm:inter_token_latency_seconds_count",
            "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total"]


def http(url, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def snapshot():
    out = {}
    try:
        with urllib.request.urlopen(METRICS, timeout=10) as r:
            for raw in r.read().decode().splitlines():
                m = LINE.match(raw)
                if m and m.group(1) in COUNTERS:
                    out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    except Exception:
        pass
    return out


class MemMin(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop, self.avail, self.free = threading.Event(), 1e9, 1e9

    def run(self):
        while not self.stop.is_set():
            d = {}
            for raw in open("/proc/meminfo"):
                k, v = raw.split(":")
                d[k] = int(v.split()[0]) / 2**20
            self.avail, self.free = min(self.avail, d["MemAvailable"]), min(self.free, d["MemFree"])
            self.stop.wait(1)


def run_job(kind, payload):
    url = f"{BASE}/{'bench' if kind == 'decode' else 'prefill-bench'}"
    while True:
        try:
            job = http(url, payload)
            break
        except urllib.error.HTTPError as e:
            body = e.read()
            if e.code in (409, 429):
                print(f"  sparkDash busy ({e.code} {body[:120]!r}), waiting 15 s", file=sys.stderr)
                time.sleep(15)
                continue
            raise RuntimeError(f"{e.code} {body[:300]!r}")
    bid = job["benchId"]
    while True:
        time.sleep(3)
        try:
            job = http(f"{url}/{bid}")
        except Exception as e:
            print(f"  poll failed ({e}), retrying", file=sys.stderr)
            continue
        if job.get("status") in ("completed", "failed", "cancelled"):
            return job


def measured(kind, payload):
    before, mm, t0 = snapshot(), MemMin(), time.time()
    mm.start()
    job = run_job(kind, payload)
    mm.stop.set()
    after = snapshot()
    d = {k: after[k] - before.get(k, 0) for k in after}
    extra = {"wall_s": round(time.time() - t0, 1), "mem_avail_min_gib": round(mm.avail, 2),
             "mem_free_min_gib": round(mm.free, 2)}
    if d.get("vllm:inter_token_latency_seconds_count"):
        extra["ms_per_step"] = round(d["vllm:inter_token_latency_seconds_sum"] / d["vllm:inter_token_latency_seconds_count"] * 1000, 1)
    if d.get("vllm:spec_decode_num_drafts_total"):
        extra["tok_per_step"] = round(1 + d["vllm:spec_decode_num_accepted_tokens_total"] / d["vllm:spec_decode_num_drafts_total"], 2)
    return job, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    sub = ap.add_subparsers(dest="kind", required=True)
    dp = sub.add_parser("decode")
    dp.add_argument("--streams", type=int, nargs="+", default=[1, 2])
    dp.add_argument("--prompt", nargs="+", default=["prose", "code"])
    dp.add_argument("--repeats", type=int, default=3)
    dp.add_argument("--max-tokens", type=int, default=600)
    pp = sub.add_parser("prefill")
    pp.add_argument("--sizes", type=int, nargs="+", default=[8192, 16384, 32768, 65536, 131072, 256000])
    a = ap.parse_args()

    if a.kind == "decode":
        print(f"{'tag':<16}{'prompt':<7}{'S':>2}{'rep':>4}{'agg tok/s':>10}{'mean/str':>9}{'ttft ms':>8}{'tokens/stream':>16}{'ms/step':>8}{'tok/step':>9}{'avail':>7}")
        for rep in range(a.repeats):
            for s in (a.streams if rep % 2 == 0 else a.streams[::-1]):
                for prompt in a.prompt:
                    job, extra = measured("decode", {"port": 8888, "modelId": MODEL, "concurrencies": [s], "maxTokens": a.max_tokens, "promptType": prompt})
                    res = (job.get("results") or [{}])[0]
                    toks = [st.get("decodeTokens") for st in res.get("streams", [])]
                    row = {"tag": a.tag, "kind": "decode", "prompt": prompt, "S": s, "rep": rep,
                           "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": job.get("status"),
                           "aggregate_tps": res.get("aggregateDecodeTps"), "mean_tps": res.get("meanDecodeTps"),
                           "ttft_ms": res.get("meanTtftMs"), "failed": res.get("streamsFailed"),
                           "decode_tokens": toks, **extra, "benchId": job.get("benchId"), "result": res}
                    with open(a.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"{a.tag:<16}{prompt:<7}{s:>2}{rep:>4}{(row['aggregate_tps'] or 0):>10.1f}{(row['mean_tps'] or 0):>9.1f}"
                          f"{(row['ttft_ms'] or 0):>8.0f}{str(toks):>16}{extra.get('ms_per_step', 0):>8}{extra.get('tok_per_step', 0):>9}"
                          f"{extra['mem_avail_min_gib']:>7}", flush=True)
    else:
        job, extra = measured("prefill", {"port": 8888, "modelId": MODEL, "contextSizes": a.sizes})
        row = {"tag": a.tag, "kind": "prefill", "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": job.get("status"),
               **extra, "job": job}
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        res = job.get("results") or []
        print(f"{a.tag} prefill status={job.get('status')} wall={extra['wall_s']}s avail_min={extra['mem_avail_min_gib']}")
        for r in res:
            print(f"  {json.dumps({k: r.get(k) for k in ('contextSize', 'promptTokens', 'ttftMs', 'prefillTps', 'error') if k in r})}")
        if not res:
            print("  " + json.dumps(job)[:1500])


if __name__ == "__main__":
    main()
