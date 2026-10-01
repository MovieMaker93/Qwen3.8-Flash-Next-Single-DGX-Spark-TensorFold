#!/usr/bin/env python3
"""Agent-style resend TTFT: turn 1 sends a fresh N-token conversation, turns 2-3 resend it with the reply and a short
new user message (what an agent harness does every step). Thinking off, greedy, short replies.
  resend_bench.py --base http://127.0.0.1:8888 --model M --tag T --out F.jsonl --sizes 8192 32768 100000"""
import argparse, json, random, time, urllib.request

WORDS = ("amber basin cedar delta ember fjord granite harbor island juniper kestrel lagoon meadow nectar orchard "
         "pebble quarry ridge summit thicket upland valley willow yarrow zephyr anchor beacon canyon dune estuary "
         "falcon glacier heron inlet jetty knoll lichen marsh nomad osprey prairie quartz reef sierra tundra").split()

def stream(base, model, messages, max_tokens=48):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
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
    return {"ttft": round((first or time.time()) - t0, 3), "total": round(time.time() - t0, 3), "text": "".join(text),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--sizes", type=int, nargs="+", default=[8192, 32768, 100000])
a = ap.parse_args()
out = open(a.out, "a")
for n in a.sizes:
    rng = random.Random(f"{a.tag}-{n}-{time.time()}")
    words = int(n / 1.33)                                   # ~1.33 tokens a word for this filler
    doc = f"[{rng.random():.12f}] Notes:\n" + " ".join(rng.choice(WORDS) for _ in range(words))
    msgs = [{"role": "user", "content": doc + "\n\nList the first three words of the notes."}]
    for turn in (1, 2, 3):
        r = stream(a.base, a.model, msgs)
        rec = {"tag": a.tag, "size": n, "turn": turn, **{k: v for k, v in r.items() if k != "text"}}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(json.dumps(rec), flush=True)
        msgs = msgs + [{"role": "assistant", "content": r["text"]},
                       {"role": "user", "content": f"Now name word number {turn * 7 + 3} of the notes."}]
