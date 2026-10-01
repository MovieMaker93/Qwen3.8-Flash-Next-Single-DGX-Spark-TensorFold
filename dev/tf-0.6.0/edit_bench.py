#!/usr/bin/env python3
"""Edit-style decode (the reply repeats the prompt's text, as an agent rewriting a file does): greedy, thinking off.
  edit_bench.py --base URL --model M --tag T --out F.jsonl [--reps 2]"""
import argparse, hashlib, json, time, urllib.request
from pathlib import Path

SRC = Path(__file__).with_name("edit_src.py").read_text()
ASK = ("Return the following Python file unchanged except that the function `_slot` is renamed to `_make_slot` "
       "everywhere (its definition and every call). Output only the complete file, no explanation.\n\n```python\n"
       + SRC + "```")

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True); ap.add_argument("--reps", type=int, default=2)
a = ap.parse_args()
for rep in range(a.reps):
    body = {"model": a.model, "messages": [{"role": "user", "content": ASK}], "max_tokens": 4000, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, text, usage = time.time(), None, [], {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            s = raw.decode().strip()
            if not s.startswith("data:") or s == "data: [DONE]":
                continue
            ev = json.loads(s[5:])
            usage = ev.get("usage") or usage
            for c in ev.get("choices", []):
                d = (c.get("delta") or {}).get("content")
                if d:
                    first = first or time.time()
                    text.append(d)
    end, n = time.time(), usage.get("completion_tokens", 0)
    rec = {"tag": a.tag, "rep": rep, "ttft": round(first - t0, 3), "completion": n,
           "decode_tps": round((n - 1) / (end - first), 2) if n > 1 else None,
           "sha": hashlib.sha256("".join(text).encode()).hexdigest()[:12]}
    open(a.out, "a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
