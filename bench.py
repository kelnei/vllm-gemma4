#!/usr/bin/env python3
"""Benchmark a vLLM OpenAI server: greedy streaming chat completions,
1024 tokens out, 3 single-stream runs + one 8-way concurrent batch.

Reports single-stream decode tok/s (mean of runs) and aggregate tok/s at c8.
Same pattern as the vllm-qwen36 bench for comparability.
"""
import json
import sys
import time
import threading
import urllib.request

BASE = "http://localhost:8000"
MAX_TOKENS = 1024
SINGLE_RUNS = 3
CONCURRENCY = 8

PROMPTS = [
    "Write a detailed essay about the history of container shipping.",
    "Explain how a modern CPU branch predictor works, in depth.",
    "Describe the water cycle and its role in climate, thoroughly.",
    "Write a long story about a lighthouse keeper on a remote island.",
    "Explain the theory behind public-key cryptography in detail.",
    "Describe the geology of the Grand Canyon at length.",
    "Write an in-depth guide to sourdough bread baking.",
    "Explain how transformers (the ML architecture) work, in detail.",
]


def get_model():
    with urllib.request.urlopen(BASE + "/v1/models") as r:
        return json.load(r)["data"][0]["id"]


def stream_run(model, prompt):
    """Returns (n_completion_tokens, first_token_time, end_time, start_time)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    usage_tokens = None
    n_chunks = 0
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage_tokens = obj["usage"].get("completion_tokens")
            choices = obj.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                if first is None:
                    first = time.perf_counter()
                n_chunks += 1
    end = time.perf_counter()
    tokens = usage_tokens if usage_tokens is not None else n_chunks
    return tokens, first or start, end, start


def main():
    model = get_model()
    print(f"model: {model}")

    # warmup
    stream_run(model, "Say hello.")

    # single-stream
    singles = []
    for i in range(SINGLE_RUNS):
        tokens, first, end, start = stream_run(model, PROMPTS[i])
        decode_tps = (tokens - 1) / (end - first) if end > first else 0.0
        ttft = first - start
        singles.append(decode_tps)
        print(f"single run {i+1}: {tokens} tok, ttft {ttft*1000:.0f} ms, "
              f"decode {decode_tps:.1f} tok/s")
    mean_single = sum(singles) / len(singles)

    # concurrent batch
    results = [None] * CONCURRENCY
    def worker(idx):
        results[idx] = stream_run(model, PROMPTS[idx % len(PROMPTS)])
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(CONCURRENCY)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.perf_counter()
    total_tokens = sum(r[0] for r in results)
    agg_tps = total_tokens / (t1 - t0)
    print(f"c{CONCURRENCY}: {total_tokens} tok total in {t1-t0:.1f}s, "
          f"aggregate {agg_tps:.1f} tok/s")

    print(f"\nRESULT {model}: single {mean_single:.1f} tok/s "
          f"(mean of {SINGLE_RUNS}), aggregate c{CONCURRENCY} {agg_tps:.1f} tok/s")


if __name__ == "__main__":
    main()
