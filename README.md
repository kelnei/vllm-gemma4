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

All five have been verified with this compose file on an RTX PRO 6000 Blackwell (96 GB) running vLLM v0.26.0 — see [Benchmarks](#benchmarks). Optional [speculative decoding](#enabling-speculative-decoding) adds up to +114% decode on the dense models.

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
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png"}},
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

The compose file defaults to the 26B-A4B MoE, but any Gemma 4 NVFP4 checkpoint from the table above works the same way. Change the model (the first entry under `command:`, passed positionally) and `--served-model-name` in `docker-compose.yml`, then `docker compose up -d`.

Two things to adjust for the E-series variants:

- E2B/E4B max out at 131,072 context (vs 262,144 for the others), so also lower `--max-model-len` to `131072` — vLLM refuses to start if it exceeds the model's maximum.
- They additionally support audio input; enable it by adding `"audio": 1` to `--limit-mm-per-prompt`.

## Enabling speculative decoding

Off by default, but worth turning on — it is worth up to +114% decode on the dense models. See [Speculative decoding](#speculative-decoding) for the measurements behind these recommendations.

**DFlash** (recommended for the dense 31B and 12b) needs one extra argument in `docker-compose.yml`:

```yaml
      - "--speculative-config"
      - '{"method": "dflash", "model": "z-lab/gemma-4-31B-it-DFlash", "num_speculative_tokens": 8, "attention_backend": "triton_attn"}'
```

`"attention_backend": "triton_attn"` is required, and is the one place this deviates from the drafter's model card. The card suggests `flash_attn`, which cannot start here: FlashAttention supports neither the fp8 KV cache nor Gemma 4's multimodal PrefixLM attention, and the engine exits during startup. vLLM propagates the target's forced Triton backend to MTP drafters automatically but not to DFlash ones, so it has to be stated. Drafters are `z-lab/gemma-4-31B-it-DFlash`, `z-lab/gemma-4-26B-A4B-it-DFlash` and `z-lab/gemma4-12B-it-DFlash` (note the inconsistent `gemma4-` prefix on the 12b).

**MTP** (recommended for the 26B-A4B MoE) needs the argument *and* an environment variable:

```yaml
    environment:
      - VLLM_USE_V2_MODEL_RUNNER=1
    command:
      - "--speculative-config"
      - '{"model": "google/gemma-4-26B-A4B-it-assistant", "num_speculative_tokens": 2}'
```

The environment variable is not optional. Under vLLM's default V1 model runner, Gemma 4 MTP fails at startup with `a and b must have same reduction dim, but got [s47, 3840] X [5632, 1024]`. The assistant checkpoint's `pre_projection` expects two backbone-width tensors concatenated (2 x 2816 = 5632), which requires the target's embeddings to be shared into the draft model; the V1 proposer only does that sharing for EAGLE-family drafters, so the draft falls back to its own 1024-wide embeddings and the shapes do not line up. The V2 speculator shares them unconditionally. The method is inferred from the checkpoint, so `"method"` can be omitted.

Two caveats on the V2 runner:

- It does not support the `thinking_token_budget` request parameter. If you need that, you cannot use MTP on this release.
- **MTP does not work on the 12b.** The 12b is the `gemma4_unified` architecture, and its assistant trips a separate bug during CUDA graph capture: `compute_logits` suppresses tokens with `logits[:, self._suppress_token_ids] = -inf` using a CPU index tensor, which capture rejects. Adding `--enforce-eager` does get it to start — confirming that is the cause — but it is not worth doing: without CUDA graphs the 12b drops to 76 tok/s, well below its 114 tok/s baseline. Use DFlash on the 12b, which is the better choice there regardless.

## Benchmarks

Measured 2026-07-26 on an RTX PRO 6000 Blackwell (96 GB) with vLLM v0.26.0 and this repo's compose settings (fp8 KV cache, no speculative decoding). Method: greedy decoding, 1024 tokens generated per request; single-stream is the mean of one run per prompt (8 prompts), aggregate is one batch of 8 concurrent streams. Reproduce with [bench.py](bench.py) (stdlib only) against a running server:

```bash
python3 bench.py
```

| Model | Single-stream decode | Aggregate, 8 streams | KV cache capacity |
| --- | --- | --- | --- |
| gemma-4-31B | 56.3 tok/s | 417 tok/s | 427k tokens |
| **gemma-4-26B-A4B** (default) | **221 tok/s** | **1,122 tok/s** | 1.95M tokens |
| gemma-4-12b | 114 tok/s | 843 tok/s | 1.60M tokens |
| gemma-4-E4B | 214 tok/s | 1,450 tok/s | 4.37M tokens |
| gemma-4-E2B | 324 tok/s | 2,072 tok/s | 13.5M tokens |

KV cache capacity is what vLLM reports at startup at `--gpu-memory-utilization 0.92` with each model's maximum context configured (262,144, except 131,072 for E2B/E4B). The headline result: the 26B-A4B MoE decodes ~4x faster than the dense 31B while being close to it in quality, and even edges out the much smaller E4B.

Decode throughput is unchanged from v0.25.1 (within ~1% on every model), but reported KV cache capacity dropped by roughly a third across the board — the 26B-A4B went from 3.0M tokens to 1.95M at the same utilization. Nothing in this repo's settings changed; it is how v0.26.0 accounts for Gemma 4's two different head dimensions (256 on sliding-window layers, 512 on global ones). Even the reduced figure is 7.4x the model's own 262k context, so it only matters if you were relying on the old number for capacity planning.

### Speculative decoding

Gemma 4's NVFP4 checkpoints bundle no draft head, but two separate drafters exist, and vLLM v0.26.0 supports both. Neither is enabled in `docker-compose.yml` — see [Enabling speculative decoding](#enabling-speculative-decoding) for the flags.

- **MTP** — Google's `*-it-assistant` checkpoints, a 4-layer decoder that shares the target's KV cache. Published for all five models.
- **DFlash** — [z-lab](https://z-lab.ai/projects/dflash/)'s block-diffusion drafter, which proposes a whole block in one pass. Published for the 31B, 26B-A4B and 12b.

Same method and hardware as above, `num_speculative_tokens` tuned per method (2 for MTP, 8 for DFlash):

| Model | Baseline | MTP | DFlash |
| --- | --- | --- | --- |
| gemma-4-31B | 56.3 / 417 | 101.4 / 697 | **120.3 / 702** |
| **gemma-4-26B-A4B** (default) | 221 / 1,122 | **283.1 / 1,462** | 278.9 / 1,233 |
| gemma-4-12b | 114 / 843 | *fails, see below* | **203.1 / 1,203** |

*single-stream tok/s / aggregate tok/s at 8 streams.*

Which drafter wins depends on the model, and the gains are much larger for the dense models than for the MoE:

- **Dense 31B and 12b: DFlash, and it is transformative.** The 31B goes from 56 to 120 tok/s (+114%) and the 12b from 114 to 203 (+78%). Decode on a dense model is bandwidth-bound, so amortizing weight reads across several tokens per forward pass pays off enormously. A 31B at 120 tok/s is a genuinely different serving proposition from one at 56.
- **26B-A4B MoE: MTP, mostly for concurrency.** Single-stream is a near tie (283 vs 279), but MTP delivers 1,462 tok/s aggregate against DFlash's 1,233. The MoE only activates 3.8B parameters per token, so it is far less bandwidth-starved to begin with and has less headroom to reclaim.
- **Draft length matters, and shorter is usually better.** DFlash's model card suggests `num_speculative_tokens: 15`; on the 26B-A4B, 8 measured faster (278.9 vs 261.6) and 4 gave the best aggregate (1,268). Acceptance decays steeply with position — the first four positions account for ~1.14 of the 1.29 tokens accepted per step — so a long draft mostly buys verification cost. For MTP, 2 beat 3 (283.1 vs 277.4): vLLM re-runs the same MTP layer for each extra token, which lowers acceptance.
- **Speculative throughput is strongly prompt-dependent.** With DFlash on the 26B-A4B, the technical prompts in `bench.py` decode ~50% faster than the prose ones (309 vs 199 tok/s); MTP is steadier (260–303). Predictable text drafts well and narrative does not, so a 3-run mean swings with the prompts it happens to hit — this is why `bench.py` now averages one run per prompt across all 8.

Both drafters cost KV cache capacity: on the 26B-A4B, 1.95M tokens baseline drops to 1.72M with DFlash.

### DGX Spark (GB10)

Same method on a DGX Spark using [docker-compose.spark.yml](docker-compose.spark.yml) (`--gpu-memory-utilization 0.78`, unified memory), on vLLM v0.25.1\*:

| Model | Single-stream decode | Aggregate, 8 streams | KV cache capacity |
| --- | --- | --- | --- |
| gemma-4-31B | 8.9 tok/s | 68 tok/s | 731k tokens |
| **gemma-4-26B-A4B** (default) | **48.6 tok/s** | **217 tok/s** | 3.3M tokens |
| gemma-4-12b | 21.5 tok/s | 170 tok/s | 3.0M tokens |
| gemma-4-E4B | 42.0 tok/s | 343 tok/s | 6.6M tokens |
| gemma-4-E2B | 78.3 tok/s | 606 tok/s | 19.5M tokens |

\* One release behind the RTX PRO 6000 figures above — the Spark was in use when those were re-run on v0.26.0, so these are pending a refresh. Speculative decoding is untested on this hardware for the same reason.

The GB10's unified LPDDR5X gives roughly a fifth of the discrete card's memory bandwidth, and decode is bandwidth-bound, so everything scales down accordingly. The same conclusion holds even more strongly here: the dense 31B is not interactive on this hardware (~9 tok/s), while the 26B-A4B MoE stays comfortably usable.

## Tuning

- `--gpu-memory-utilization 0.92` leaves headroom for CUDA graph capture; pushing it higher can OOM after the KV cache is allocated.
- On unified-memory machines (DGX Spark / GB10) use [docker-compose.spark.yml](docker-compose.spark.yml) instead — select it with `COMPOSE_FILE=docker-compose.spark.yml` in `.env`. The GPU shares its ~120 GB with the OS: utilization is capped at 0.78 because higher fractions starve the host during KV-cache allocation, hard enough to need a power cycle at 0.92 (disable swap so an overrun OOM-kills the engine instead of thrashing).
- On GPUs with less memory, lower `--max-model-len` first — the full 262k context is the main memory consumer after the weights.
- `--max-num-seqs 64` and `--max-num-batched-tokens 32768` are sized for a workstation serving a handful of concurrent clients; raise them for heavier batch serving.
- Vision detail per image is tunable per request: `"mm_processor_kwargs": {"max_soft_tokens": 1120}` (default 280; 70 for cheap thumbnails).
- Speculative decoding is not enabled by default, but both a DFlash and an MTP drafter exist for these models and are worth adding — see [Enabling speculative decoding](#enabling-speculative-decoding).
- On this GPU vLLM serves Gemma 4 with the Triton attention backend, not FlashAttention. Gemma 4's head dimensions differ between sliding-window (256) and global (512) layers, which needs FA4; FA3/FA4 are not built for sm_120, so vLLM logs `FA4 not available, forcing TRITON_ATTN backend` and falls back. There is nothing to tune here — it is the only working backend for this combination — but it explains why `--attention-backend flash_attn` fails.

## License

[MIT](LICENSE)
