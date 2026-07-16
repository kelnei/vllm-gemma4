# Gemma 4 NVFP4 on Blackwell with vLLM

Docker Compose setup for serving [unsloth Gemma 4 NVFP4](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-NVFP4) checkpoints with [vLLM](https://github.com/vllm-project/vllm), targeting NVIDIA Blackwell hardware. Exposes an OpenAI-compatible API on port 8000 with the model's full 262,144-token context, fp8 KV cache, vision input enabled, and Gemma 4 thinking + tool-call parsing.

Default model: [unsloth/gemma-4-26B-A4B-it-NVFP4](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-NVFP4) — a multimodal MoE (25.2B total / 3.8B active parameters) that decodes at near-4B-model speed. Any of these swaps in the same way (see [Swapping models](#swapping-models)):

| Model | Type | Max context | Notes |
| --- | --- | --- | --- |
| [gemma-4-31B-it-NVFP4](https://huggingface.co/unsloth/gemma-4-31B-it-NVFP4) | Dense 31B | 262,144 | Highest quality |
| [gemma-4-26B-A4B-it-NVFP4](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-NVFP4) | MoE, 3.8B active | 262,144 | **Default** — fastest quality/speed trade-off |
| [gemma-4-12b-it-NVFP4](https://huggingface.co/unsloth/gemma-4-12b-it-NVFP4) | Dense 12B | 262,144 | |
| [gemma-4-E4B-it-NVFP4](https://huggingface.co/unsloth/gemma-4-E4B-it-NVFP4) | 4B effective | 131,072 | Audio input support |
| [gemma-4-E2B-it-NVFP4](https://huggingface.co/unsloth/gemma-4-E2B-it-NVFP4) | 2B effective | 131,072 | Audio input support |

All five have been verified with this compose file on an RTX PRO 6000 Blackwell (96 GB) — see [Benchmarks](#benchmarks).

## Requirements

- An NVIDIA Blackwell GPU — NVFP4 relies on Blackwell's native FP4 tensor cores. The defaults here assume a ~96 GB card; see [Tuning](#tuning) for smaller GPUs.
- A recent NVIDIA driver, Docker, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- A Hugging Face token for the model download.

## Quick start

```bash
cp .env.example .env   # then fill in your HF token (or export HF_TOKEN instead)
./start.sh
```

First boot downloads the model weights and warms up the engine, which can take several minutes; the healthcheck allows up to 10 minutes. Watch progress with `docker compose logs -f`.

Once healthy, test it:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26b-a4b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Vision works through the standard OpenAI `image_url` content parts (up to 4 images per prompt by default):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26b-a4b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}},
        {"type": "text", "text": "Describe this image in one sentence."}
      ]
    }]
  }'
```

Gemma 4 wants images *before* text in the content array. `data:` URIs work too for local files.

Model weights are cached in `${HOME}/.cache/huggingface` on the host (bind-mounted into the container), so they survive container rebuilds and are shared with any local Hugging Face tooling.

## What's configured

| Setting | Value | Why |
| --- | --- | --- |
| Context length | 262,144 tokens | The model's full native context (hybrid sliding-window + global attention keeps the KV footprint small) |
| KV cache | fp8 | The checkpoint ships calibrated fp8 KV scales; roughly doubles KV capacity vs fp16 |
| Vision | `--limit-mm-per-prompt '{"image": 4}'` | Up to 4 images per prompt; 280 soft tokens per image by default (tunable 70–1120 via `mm_processor_kwargs`) |
| Reasoning parser | `gemma4` | Exposes thinking via the API's `reasoning` field |
| Tool-call parser | `gemma4` | Matches Gemma 4's native tool-call protocol |
| Chat template | model's bundled template | The unsloth checkpoint's template already renders tools and supports `enable_thinking`; no override needed |
| Quantization | auto-detected | No `--quantization` flag: vLLM detects compressed-tensors and picks the fast NVFP4 kernel; forcing a backend (e.g. marlin) costs ~2x decode throughput |
| Sampling defaults | temperature 1.0, top_p 0.95, top_k 64 | Google's recommended params, baked into the checkpoint's `generation_config.json`; vLLM applies them when the request doesn't override |

Thinking is controlled per request via the chat template:

```json
{"chat_template_kwargs": {"enable_thinking": true}}
```

## Swapping models

The compose file defaults to the 26B-A4B MoE, but any Gemma 4 NVFP4 checkpoint from the table above works the same way. Change `--model` (and `--served-model-name`) in `docker-compose.yml`, then `docker compose up -d`.

Two things to adjust for the E-series variants:

- E2B/E4B max out at 131,072 context (vs 262,144 for the others), so also lower `--max-model-len` to `131072` — vLLM refuses to start if it exceeds the model's maximum.
- They additionally support audio input; enable it by adding `"audio": 1` to `--limit-mm-per-prompt`.

## Benchmarks

Measured 2026-07-15 on an RTX PRO 6000 Blackwell (96 GB) with vLLM v0.25.1 and this repo's compose settings (fp8 KV cache, no speculative decoding). Method: greedy decoding, 1024 tokens generated per request; single-stream is the mean of 3 runs, aggregate is one batch of 8 concurrent streams. Reproduce with [bench.py](bench.py) (stdlib only) against a running server:

```bash
python3 bench.py
```

| Model | Single-stream decode | Aggregate, 8 streams | KV cache capacity |
| --- | --- | --- | --- |
| gemma-4-31B | 55.5 tok/s | 412 tok/s | 660k tokens |
| **gemma-4-26B-A4B** (default) | **219 tok/s** | **1,120 tok/s** | 3.0M tokens |
| gemma-4-12b | 115 tok/s | 859 tok/s | 2.7M tokens |
| gemma-4-E4B | 211 tok/s | 1,440 tok/s | 6.0M tokens |
| gemma-4-E2B | 329 tok/s | 2,084 tok/s | 18.0M tokens |

KV cache capacity is what vLLM reports at startup at `--gpu-memory-utilization 0.92` with each model's maximum context configured (262,144, except 131,072 for E2B/E4B). The headline result: the 26B-A4B MoE decodes ~4x faster than the dense 31B while being close to it in quality, and even edges out the much smaller E4B.

### DGX Spark (GB10)

Same method on a DGX Spark using [docker-compose.spark.yml](docker-compose.spark.yml) (`--gpu-memory-utilization 0.78`, unified memory):

| Model | Single-stream decode | Aggregate, 8 streams | KV cache capacity |
| --- | --- | --- | --- |
| gemma-4-31B | 8.9 tok/s | 68 tok/s | 731k tokens |
| **gemma-4-26B-A4B** (default) | **48.6 tok/s** | **217 tok/s** | 3.3M tokens |
| gemma-4-12b | 21.5 tok/s | 170 tok/s | 3.0M tokens |
| gemma-4-E4B | 42.0 tok/s | 343 tok/s | 6.6M tokens |
| gemma-4-E2B | 78.3 tok/s | 606 tok/s | 19.5M tokens |

The GB10's unified LPDDR5X gives roughly a fifth of the discrete card's memory bandwidth, and decode is bandwidth-bound, so everything scales down accordingly. The same conclusion holds even more strongly here: the dense 31B is not interactive on this hardware (~9 tok/s), while the 26B-A4B MoE stays comfortably usable.

## Tuning

- `--gpu-memory-utilization 0.92` leaves headroom for CUDA graph capture; pushing it higher can OOM after the KV cache is allocated.
- On unified-memory machines (DGX Spark / GB10) use [docker-compose.spark.yml](docker-compose.spark.yml) instead — select it with `COMPOSE_FILE=docker-compose.spark.yml` in `.env`. The GPU shares its ~120 GB with the OS: utilization is capped at 0.78 because higher fractions starve the host during KV-cache allocation, hard enough to need a power cycle at 0.92 (disable swap so an overrun OOM-kills the engine instead of thrashing).
- On GPUs with less memory, lower `--max-model-len` first — the full 262k context is the main memory consumer after the weights.
- `--max-num-seqs 64` and `--max-num-batched-tokens 32768` are sized for a workstation serving a handful of concurrent clients; raise them for heavier batch serving.
- Vision detail per image is tunable per request: `"mm_processor_kwargs": {"max_soft_tokens": 1120}` (default 280; 70 for cheap thumbnails).
- Speculative decoding is not configured: unlike the Qwen3.6 NVFP4 checkpoints, this checkpoint bundles no MTP head. Google publishes separate lightweight assistant/MTP draft models for Gemma 4 — untested here, candidates for a future speedup.

## License

[MIT](LICENSE)
