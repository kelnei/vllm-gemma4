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

All five have been verified with this compose file on an RTX PRO 6000 Blackwell (96 GB) and a DGX Spark (GB10), both running vLLM v0.26.0 — see [Benchmarks](#benchmarks). Optional [speculative decoding](#enabling-speculative-decoding) adds up to +118% decode on the dense models. All five also run across two DGX Sparks as one tensor-parallel cluster — see [Two-Spark cluster](#two-spark-cluster).

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

Off by default, but worth turning on — it is worth up to +118% decode on the dense models, on both the RTX PRO 6000 and the DGX Spark. See [Speculative decoding](#speculative-decoding) for the measurements behind these recommendations.

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

## Two-Spark cluster

[run_cluster.sh](run_cluster.sh) joins two DGX Sparks into a Ray cluster and serves one Gemma 4 model across both GPUs with tensor parallelism (TP=2), NCCL riding RDMA (RoCE) over the dedicated 200 GbE link between them. A single GB10 already runs every model in this repo; what the second Spark buys is headroom — twice the aggregate memory bandwidth and twice the KV-cache memory — at the price of putting every per-layer all-reduce on the wire. What that trade returns varies enormously by model, from +80% single-stream decode on the dense 31B to a marginal +4% aggregate on the E2B — see [the cluster benchmarks](#2x-dgx-spark-tp2-cluster).

```bash
./run_cluster.sh head                # on the head Spark
./run_cluster.sh worker              # on the other Spark (head IP from .env, or pass it)
./run_cluster.sh serve 31b           # back on the head; 31b | 26b-a4b | 12b | e4b | e2b
```

`status` reports tmux/container/Ray/API state on any node; `stop` tears down that node's half. Everything long-running lives in detached tmux sessions (`ray-node` holds the Ray container on each node, `vllm-serve` holds the engine on the head), so an SSH drop doesn't take the cluster down; engine output is mirrored to `~/vllm-cluster-serve.log`. Set `CLUSTER_HEAD_IP` in `.env` (see `.env.example`) to the head's IP *on the 200G link*; `CLUSTER_IF` and `CLUSTER_HCA` default to the Spark's 200G netdev and its RoCE device. The image ships without Ray, so each node pip-installs it at container start (~1 min, needs internet). Once healthy, the API is on port 8000 of the head node, same as the single-node compose.

The serve profile reuses the single-Spark tuning unchanged — fp8 KV cache, utilization 0.78 (a per-node fraction; the host-starvation ceiling it protects doesn't move by adding a machine), `--max-num-batched-tokens 32768` — with vision and both parsers enabled. Speculative decoding is not configured on the cluster. Two things are specific to this setup:

- **The 26B-A4B runs its expert layers with expert parallelism** (`--enable-expert-parallel`, already wired into the script), not tensor parallelism. TP=2 would halve each expert's intermediate size (704 → 352), making the fused gate|up weight 704 rows — not a multiple of the NVFP4 128-row scale tile — and the fast `FLASHINFER_CUTLASS` MoE backend refuses to pad gated weights (`NotImplementedError` at load). With EP each rank instead holds 64 whole experts at the single-GPU shape, the fast backend loads as-is, and attention is still TP=2.
- **The cluster pins a nightly vLLM image** by commit SHA instead of v0.26.0. v0.26.0's shared-memory message queue — which the engine uses to drive cross-node workers — can lose a reader wakeup notification, parking the engine and both workers forever on queues that have data; the engine then dies minutes later with "RPC call to sample_tokens timed out". Upstream has since bounded the park time so a lost wakeup recovers within ~5 s, and the pinned nightly is the first known-good image. Single-node deployments don't exercise this path at risk, so the compose files stay on v0.26.0. The pin moves to the next tagged release when it lands.

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

Same method on a DGX Spark using [docker-compose.spark.yml](docker-compose.spark.yml) (`--gpu-memory-utilization 0.78`, unified memory), re-measured 2026-07-28 on vLLM v0.26.0:

| Model | Single-stream decode | Aggregate, 8 streams | KV cache capacity |
| --- | --- | --- | --- |
| gemma-4-31B | 8.8 tok/s | 68 tok/s | 476k tokens |
| **gemma-4-26B-A4B** (default) | **46.3 tok/s** | **211 tok/s** | 2.12M tokens |
| gemma-4-12b | 21.1 tok/s | 166 tok/s | 1.74M tokens |
| gemma-4-E4B | 41.8 tok/s | 341 tok/s | 4.76M tokens |
| gemma-4-E2B | 77.4 tok/s | 591 tok/s | 14.7M tokens |

The GB10's unified LPDDR5X gives roughly a fifth of the discrete card's memory bandwidth, and decode is bandwidth-bound, so everything scales down accordingly. The same conclusion holds even more strongly here: the dense 31B is not interactive on this hardware at 8.8 tok/s, while the 26B-A4B MoE stays comfortably usable. Speculative decoding changes that picture substantially — see [below](#speculative-decoding-on-the-spark).

**Decode throughput is within a few percent of v0.25.1 on every model** — between −4.7% and −0.5% single-stream — which is inside the run-to-run spread described below rather than a version regression. KV cache capacity fell by roughly a third for the same reason documented above — v0.26.0's accounting for Gemma 4's two head dimensions — and the reduced figures are still 1.8x to 112x each model's own maximum context.

**Expect up to 5% run-to-run spread on these figures, and do not read a small delta as a regression.** Repeating the same configuration on the same hardware moved these numbers by as much as 5%, so treat any sub-5% difference — against a previous release, another machine, or another run — as noise until it reproduces. These figures were taken on a machine that had been under continuous load for hours, which is what a running server actually delivers.

#### Speculative decoding on the Spark

DFlash works on GB10 and is the single highest-leverage flag available on this hardware. Same method as above, `num_speculative_tokens: 8`:

| Model | Baseline | DFlash | Single | Aggregate | Acceptance | KV cache |
| --- | --- | --- | --- | --- | --- | --- |
| gemma-4-31B | 8.8 / 68 | **19.2 / 111** | +118% | +64% | 21.3% | 476k → 426k |
| **gemma-4-26B-A4B** (default) | 46.3 / 211 | **56.0 / 217** | +21% | +3.1% | 16.0% | 2.12M → 1.89M |
| gemma-4-12b | 21.1 / 166 | **40.8 / 242** | +93% | +45% | 18.5% | 1.74M → 1.56M |

*single-stream tok/s / aggregate tok/s at 8 streams.*

The dense-model gains match what the RTX PRO 6000 sees (+118% here against +114% there), which is the expected result: dense decode is bandwidth-bound, the Spark is bandwidth-poor, and speculation amortizes each weight read across several tokens. Three things are specific to this hardware:

- **The 31B becomes usable.** 8.8 tok/s is below reading speed; 19.2 tok/s is not. That does not make it the right default — the MoE is still faster untuned than the 31B is with DFlash — but it moves the dense 31B from "not worth serving here" to "viable if you need its quality".
- **The 12b with DFlash is the Spark's aggregate throughput winner** among the three, at 242 tok/s against the MoE's 217. If you are serving several concurrent users and can accept a 12B-class model, that is the configuration to run.
- **The MoE gains single-stream but almost nothing in aggregate** (+21% vs +3.1%). At 8 concurrent streams it already amortizes weight reads across the batch, so there is little left for speculation to reclaim — the same effect the RTX shows, but starker here.

Two caveats on these numbers. `num_speculative_tokens: 8` was carried over from the RTX tuning rather than re-swept on GB10; per-position acceptance decays steeply (0.81, 0.54, 0.36, 0.18, 0.12, 0.06, 0.05, 0.03), so positions 6–8 contribute only ~6% of accepted tokens and a shorter draft would likely trade a little throughput for meaningfully less verification work. And **MTP is untested on the Spark** — on the RTX it beat DFlash on the MoE for aggregate throughput, so the MoE row here may not be that model's best available configuration.

### 2x DGX Spark (TP=2 cluster)

Same method on both Sparks joined by [run_cluster.sh](run_cluster.sh) — TP=2 over the 200 GbE RoCE link, `--gpu-memory-utilization 0.78` per node, no speculative decoding. Measured 2026-07-28 on the cluster's pinned nightly image (see [Two-Spark cluster](#two-spark-cluster)); the comparison columns are against the v0.26.0 single-Spark table above:

| Model | Single-stream decode | Aggregate, 8 streams | KV cache capacity | vs one Spark (single / agg / KV) |
| --- | --- | --- | --- | --- |
| gemma-4-31B | 15.8 tok/s | 117 tok/s | 1.12M tokens | +80% / +72% / 2.35x |
| **gemma-4-26B-A4B** (default) | **62.1 tok/s** | **297 tok/s** | 4.54M tokens | +34% / +41% / 2.14x |
| gemma-4-12b | 34.1 tok/s | 248 tok/s | 3.15M tokens | +62% / +49% / 1.81x |
| gemma-4-E4B | 58.7 tok/s | 422 tok/s | 9.81M tokens | +40% / +24% / 2.06x |
| gemma-4-E2B | 96.0 tok/s | 616 tok/s | 15.0M tokens | +24% / +4% / 1.02x |

How much the second Spark buys tracks how starved the model was in the first place:

- **The dense models gain most, and the biggest gains the most of all.** Decode on a dense model is memory-bandwidth-bound, and TP=2 splits every weight read across two memory systems: +80% single-stream on the 31B (8.8 → 15.8 tok/s), +62% on the 12b. The 31B also frees the most weight memory per node, which is why its KV capacity scales furthest (2.35x).
- **The MoE gains a solid +34% / +41% through expert parallelism.** With only 3.8B active parameters it is far less bandwidth-starved than the dense models, so there is less for the cluster to reclaim — but at 62 tok/s single-stream and 4.54M tokens of KV cache it is still the model to serve, now with 2.1x the capacity.
- **The E2B is the floor of the approach.** +24% single-stream, +4% aggregate — and its KV cache capacity does not grow at all (1.02x). That last one is architectural, not noise: the E2B has a single KV head (`num_key_value_heads: 1`), which TP=2 must replicate on both ranks, so each node still pays the full per-token KV cost and total capacity stays at one node's worth. The E4B's two KV heads split exactly, hence its 2.06x. Below ~4B effective parameters, the wire costs about what the second memory system pays back.

DFlash and MTP were not re-benchmarked on the cluster; the [single-Spark speculative numbers](#speculative-decoding-on-the-spark) are the reference if you enable a drafter there.

### Serving under concurrency

The tables above measure one client, or eight. This one sweeps concurrency and prompt length with `vllm bench serve` across three configurations and the two models worth serving, to answer what those tables cannot: how many people can this serve at once, and what does it feel like while it does.

Method: `--dataset-name random`, 1024 output tokens per request, `--ignore-eos` so every request generates exactly that many, `--seed 0`, all requests submitted at once (`--request-rate inf`). The prefix cache is flushed between every shape — the seeded dataset generates identical prompts for consecutive shapes, so without a flush each shape prefills out of the previous one's KV blocks, which inflated throughput by 37% and understated TTFT by 12x in testing. (The flush route needs the server started with `VLLM_SERVER_DEV_MODE=1`.) The 2x Spark rows ran on the cluster's pinned nightly, whose `vllm bench serve` is a Rust reimplementation; the classic "Serving Benchmark Result" table is what is reported, same as the other rows.

**1,024-token prompts** — output tok/s / p99 ITL

| Config | c1 | c8 | c32 | c64 |
| --- | --- | --- | --- | --- |
| RTX PRO 6000 · 26B-A4B | 207 / 5 ms | 1,159 / 8 ms | 2,932 / 12 ms | 4,247 / 16 ms |
| RTX PRO 6000 · 31B | 55 / 19 ms | 377 / 22 ms | 1,031 / 30 ms | 1,460 / 40 ms |
| DGX Spark · 26B-A4B | 46 / 23 ms | 241 / 35 ms | 508 / 65 ms | 682 / 93 ms |
| DGX Spark · 31B | 9 / 118 ms | 60 / 137 ms | 154 / 189 ms | 205 / 273 ms |
| 2x DGX Spark · 26B-A4B | 60 / 18 ms | 314 / 26 ms | 719 / 44 ms | 947 / 82 ms |
| 2x DGX Spark · 31B | 16 / 66 ms | 104 / 74 ms | 259 / 111 ms | 341 / 160 ms |

**8,192-token prompts** — output tok/s / p99 ITL

| Config | c1 | c8 | c32 | c64 |
| --- | --- | --- | --- | --- |
| RTX PRO 6000 · 26B-A4B | 182 / 7 ms | 937 / 9 ms | 1,844 / 14 ms | 2,299 / 20 ms |
| RTX PRO 6000 · 31B | 50 / 20 ms | 271 / 23 ms | 505 / 37 ms | 597 / 52 ms |
| DGX Spark · 26B-A4B | 42 / 24 ms | 178 / 38 ms | 279 / 82 ms | 344 / 115 ms |
| DGX Spark · 31B | 8 / 121 ms | 40 / 145 ms | 67 / 242 ms | 76 / 391 ms |
| 2x DGX Spark · 26B-A4B | 54 / 19 ms | 231 / 27 ms | 399 / 63 ms | 442 / 120 ms |
| 2x DGX Spark · 31B | 14 / 68 ms | 67 / 81 ms | 108 / 146 ms | 124 / 218 ms |

Reading it:

- **The MoE is the one to serve.** On the RTX PRO 6000 it sustains 4,247 tok/s across 64 concurrent 1k-prompt requests at a 16 ms p99 inter-token latency — about 66 tok/s per user, every one of them faster than reading speed. The dense 31B manages 1,460 tok/s on the same shape.
- **Tail latency stays interactive on the discrete card everywhere.** p99 ITL never exceeds 52 ms in any of its 16 cells, including 64 concurrent 8k-token prompts. Concurrency costs throughput per user, not smoothness.
- **On the Spark, concurrency is where the MoE earns its place.** It scales 46 → 682 tok/s from c1 to c64 while holding p99 ITL under 100 ms. The dense 31B on the same shape gives 205 tok/s at a 273 ms p99 — visibly stuttery, and the clearest statement yet that the 31B is the wrong model for this hardware.
- **Long prompts cost more than long outputs.** Moving from 1k to 8k prompts costs 46% of c64 throughput on the RTX MoE and 50% on the Spark MoE, because prefill competes with decode for the same token budget rather than adding to it.
- **The cluster lifts every cell, and the dense model twice as hard as the MoE.** TP=2 over the 200G link buys the 26B-A4B +28–43% throughput and the 31B +62–76%, on all eight shapes, with better p99 ITL in 15 of the 16 cells (the MoE's 8k/c64 gives back 5 ms). The 31B at 1k/c64 moves from 205 tok/s at a visibly stuttery 273 ms p99 to 341 tok/s at 160 ms: usable now, though the MoE still nearly triples its throughput on the same shape. Prefill also rides the split (~1.5x faster on the 31B, ~1.15x on the MoE), so TTFT drops nearly everywhere — the exception is the MoE's 1k/c64 cell (3.88 → 4.72 s), where prefill is short enough that the per-layer all-reduce cost outweighs the faster compute.

**TTFT at high concurrency is queueing, not latency.** Every request is submitted at t=0, so at 8k/c64 the server has 512k tokens of prompt to chew through before the last request emits anything — which is how the Spark's dense 31B reports a 271-second median TTFT. That is a saturation measure, not interactive latency. Median TTFT at c1 is the number a user would actually see:

| Config | 1k c1 | 1k c64 | 8k c1 | 8k c64 |
| --- | --- | --- | --- | --- |
| RTX PRO 6000 · 26B-A4B | 0.03 s | 0.72 s | 0.18 s | 5.87 s |
| RTX PRO 6000 · 31B | 0.09 s | 3.59 s | 0.88 s | 32.47 s |
| DGX Spark · 26B-A4B | 0.14 s | 3.88 s | 1.27 s | 42.49 s |
| DGX Spark · 31B | 0.43 s | 37.45 s | 7.46 s | 271.09 s |
| 2x DGX Spark · 26B-A4B | 0.13 s | 4.72 s | 1.10 s | 39.69 s |
| 2x DGX Spark · 31B | 0.34 s | 19.12 s | 4.96 s | 171.20 s |

## Tuning

- `--gpu-memory-utilization 0.92` leaves headroom for CUDA graph capture; pushing it higher can OOM after the KV cache is allocated. Verified safe for Gemma 4 on a 96 GB RTX PRO 6000 — it survives 8k prompts at 32 and 64 concurrent with no OOM and no engine restart. Other model families are less forgiving at this value, so it is worth re-checking if you point this compose file at something else.
- On unified-memory machines (DGX Spark / GB10) use [docker-compose.spark.yml](docker-compose.spark.yml) instead — select it with `COMPOSE_FILE=docker-compose.spark.yml` in `.env`. The GPU shares its ~120 GB with the OS: utilization is capped at 0.78 because higher fractions starve the host during KV-cache allocation, hard enough to need a power cycle at 0.92 (disable swap so an overrun OOM-kills the engine instead of thrashing).
- On GPUs with less memory, lower `--max-model-len` first — the full 262k context is the main memory consumer after the weights.
- `--max-num-seqs 64` is sized for a workstation serving a handful of concurrent clients; raise it for heavier batch serving. `--max-num-batched-tokens 32768` is a different matter — it has been swept and should be left alone, for the reasons below.
- **`--max-num-batched-tokens 32768` is the right default on both machines, but for different reasons.** Swept across 4096/8192/32768 on all five models. On the RTX PRO 6000 lowering it is simply pointless: the best any lower value bought was +2.6% throughput. On the DGX Spark it is a real trade — 4096 is worth **+7% to +14%** on the dense models (31B +14%, 12b +12%, E2B +9.4%, E4B +7%), because a smaller GEMM is more efficient against unified LPDDR5X. The 26B-A4B MoE is the exception on both machines, gaining at most ~3% (two runs measured +0.3% and +3.1%, which brackets the run-to-run noise) — so the default model is the one with the least to gain from tuning this.
- **What lowering it costs is tail latency, everywhere: p99 ITL gets 5–15x worse.** The mechanism is that a chunked-prefill step blocks decoding requests only when it consumes the whole token budget. Once the budget exceeds the prompt, prefill and decode co-schedule in the same step and the stall stops existing rather than merely getting shorter — which is why 32768 is not on the same curve as 8192 and 4096 at all. Between those two the usual model does hold: halving the chunk halves the stall, measured at 1.84–2.39x across every model and both machines. Lowering MNBT is also a *capacity* lever, buying 1.9–3.7x the KV cache since peak activation memory falls with chunk size.
- **The one case where that trade is worth taking is the dense 31B**, which is the only model without comfortable KV headroom (1.6x its own 262k context on the RTX PRO 6000, 1.8x on the Spark; the other four have 6x–112x). On the Spark at 8k prompts, dropping it to 4096 gives 2.8x the KV cache, 19% lower TTFT and 14% more throughput at c32 — but takes p99 ITL from 234 ms to 3,233 ms. Reasonable for long-context batch work, wrong for interactive serving, which is what the default targets.
- **4096 is the floor for any Gemma 4 model.** These are multimodal, and vLLM refuses to start when `--max-num-batched-tokens` is below the encoder's per-item budget: `max_tokens_per_mm_item (2496) is larger than max_num_batched_tokens`.
- Vision detail per image is tunable per request: `"mm_processor_kwargs": {"max_soft_tokens": 1120}` (default 280; 70 for cheap thumbnails).
- Speculative decoding is not enabled by default, but both a DFlash and an MTP drafter exist for these models and are worth adding — see [Enabling speculative decoding](#enabling-speculative-decoding).
- On both these GPUs vLLM serves Gemma 4 with the Triton attention backend, not FlashAttention. Gemma 4's head dimensions differ between sliding-window (256) and global (512) layers, which needs FA4; FA3/FA4 are built for neither sm_120 (RTX PRO 6000) nor sm_121 (GB10), so vLLM logs `FA4 not available, forcing TRITON_ATTN backend` and falls back. There is nothing to tune here — it is the only working backend for this combination — but it explains why `--attention-backend flash_attn` fails, and why a DFlash drafter needs its own `"attention_backend"` entry (see [Enabling speculative decoding](#enabling-speculative-decoding)).

## License

[MIT](LICENSE)
