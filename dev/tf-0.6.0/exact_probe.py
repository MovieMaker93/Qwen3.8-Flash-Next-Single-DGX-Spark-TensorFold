#!/usr/bin/env python3
"""Fixed greedy prompts (same text every run): replies saved for byte comparison across builds.
  exact_probe.py run --base URL --model M --out F.json      exact_probe.py cmp A.json B.json"""
import json, random, sys, urllib.request

WORDS = ("amber basin cedar delta ember fjord granite harbor island juniper kestrel lagoon meadow nectar orchard "
         "pebble quarry ridge summit thicket upland valley willow yarrow zephyr anchor beacon canyon dune estuary "
         "falcon glacier heron inlet jetty knoll lichen marsh nomad osprey prairie quartz reef sierra tundra").split()

def doc(seed, words):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(words))

PROMPTS = [
    ("short", "Explain in a few sentences how a hash map handles collisions.", False, 200),
    ("code", "Write a Python function that merges two sorted lists into one sorted list, with a docstring.", False, 200),
    ("think", "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Answer briefly.", True, 400),
    ("doc8k", doc(8, 6000) + "\n\nWhich three words appear most often above? Answer briefly.", False, 120),
    ("doc30k", doc(30, 22500) + "\n\nWhat are the first five words above? Answer briefly.", False, 120),
    ("doc60k", doc(60, 45000) + "\n\nWhat are the last five words above? Answer briefly.", False, 120),
]

def run(base, model, out):
    res = {}
    for name, text, think, mt in PROMPTS:
        body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": mt, "temperature": 0,
                "seed": 7, "chat_template_kwargs": {"enable_thinking": think}}
        r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
        m = r["choices"][0]["message"]
        res[name] = {"content": m.get("content"), "reasoning": m.get("reasoning_content"),
                     "finish": r["choices"][0]["finish_reason"], "usage": r.get("usage")}
        print(name, res[name]["finish"], (r.get("usage") or {}).get("prompt_tokens"), flush=True)
    json.dump(res, open(out, "w"), indent=1)

def cmp(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    same = [k for k in A if A[k]["content"] == B.get(k, {}).get("content") and A[k]["reasoning"] == B.get(k, {}).get("reasoning")]
    print(f"identical {len(same)}/{len(A)}: {', '.join(same)}; different: {', '.join(k for k in A if k not in same) or '-'}")

if sys.argv[1] == "run":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--base"); ap.add_argument("--model"); ap.add_argument("--out")
    a = ap.parse_args(); run(a.base, a.model, a.out)
else:
    cmp(sys.argv[2], sys.argv[3])
