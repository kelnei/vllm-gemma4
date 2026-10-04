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

All five have been verified with this compose file on an RTX PRO 6000 Blackwell (96 GB) and a DGX Spark (GB10), both running vLLM v0.26.0; the default 26B-A4B config was re-verified on v0.29.0 on 2026-09-10 on both machines (all smoke tests pass, decode throughput within +3–4% of the v0.26.0 figures). The repo now pins v0.30.0, re-verified on the RTX PRO 6000 on 2026-10-01 for all five models and the recommended speculative configs: everything works as it did on v0.29.0, but single-stream decode is up to 7% slower on every model except the 12b — see [vLLM v0.30.0](#vllm-v0300). The DGX Spark and the two-Spark cluster were re-verified on 2026-10-03 for all five models, with and without each drafter: the RTX slowdown does not show up on the Spark, where every config is within 2% of v0.29.0, but on the cluster the 26B-A4B cannot start with a drafter on stock v0.30.0 — see [Speculative decoding on the cluster](#speculative-decoding-on-the-cluster). Optional [speculative decoding](#enabling-speculative-decoding) adds up to +118% decode on the dense models. All five also run across two or four DGX Sparks as one tensor-parallel cluster, measured on v0.30.0 at both sizes — see [Two-Spark cluster](#two-spark-cluster). And all five run on a single RTX 5090 (32 GB), the 31B without vision; see [RTX 5090 (32 GB)](#rtx-5090-32-gb).

## Requirements

- An NVIDIA Blackwell GPU — NVFP4 relies on Blackwell's native FP4 tensor cores. The defaults here assume a ~96 GB card; a 32 GB RTX 5090 has [its own compose file](#rtx-5090-32-gb), and [Tuning](#tuning) covers other smaller GPUs.
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

Off by default, but worth turning on — it is worth up to +118% decode on the dense models, on both the RTX PRO 6000 and the DGX Spark. See [Speculative decoding](#speculative-decoding) for the measurements behind these recommendations. (On the two-Spark cluster the same drafters are built into `run_cluster.sh` as a serve argument instead of a compose edit.)

**DFlash** (recommended for the dense 31B and 12b) needs one extra argument in `docker-compose.yml`:

```yaml
      - "--speculative-config"
      - '{"method": "dflash", "model": "z-lab/gemma-4-31B-it-DFlash", "num_speculative_tokens": 8, "attention_backend": "triton_attn"}'
```

`"attention_backend": "triton_attn"` is required, and is the one place this deviates from the drafter's model card. The card suggests `flash_attn`, which cannot start here: FlashAttention supports neither the fp8 KV cache nor Gemma 4's multimodal PrefixLM attention, and the engine exits during startup. vLLM propagates the target's forced Triton backend to MTP drafters automatically but not to DFlash ones, so it has to be stated. Drafters are `z-lab/gemma-4-31B-it-DFlash`, `z-lab/gemma-4-26B-A4B-it-DFlash` and `z-lab/gemma4-12B-it-DFlash` (note the inconsistent `gemma4-` prefix on the 12b).

**MTP** (recommended for the 26B-A4B MoE) needs only the argument:

```yaml
      - "--speculative-config"
      - '{"model": "google/gemma-4-26B-A4B-it-assistant", "num_speculative_tokens": 2}'
```

The method is inferred from the checkpoint, so `"method"` can be omitted. Drafters are `google/gemma-4-31B-it-assistant`, `google/gemma-4-26B-A4B-it-assistant` and `google/gemma-4-12B-it-assistant` (one per Gemma 4 size; the E-series ones are measured on the [DGX Spark](#speculative-decoding-on-the-spark) and the [cluster](#speculative-decoding-on-the-cluster)).

Two things that were true on vLLM v0.26.0–v0.27.x and are no longer:

- **`VLLM_USE_V2_MODEL_RUNNER=1` is no longer needed.** Gemma 4 MTP requires the target's embeddings to be shared into the draft model (the assistant's `pre_projection` expects two backbone-width tensors concatenated, 2 x 2816 = 5632), and only the V2 model runner's speculator does that sharing; under the old default V1 runner MTP died at startup with `a and b must have same reduction dim, but got [s47, 3840] X [5632, 1024]`. v0.29.0 made the V2 runner the default for every model, so the variable is redundant — re-verified 2026-09-10 without it: 285 / 1,476 tok/s on the 26B-A4B, matching the table below. The V2 runner's earlier gap, no support for the `thinking_token_budget` request parameter, is also closed: the budget is honored on v0.29.0.
- **MTP now works on the 12b.** Its `gemma4_unified` assistant used to trip CUDA graph capture (`compute_logits` suppressed tokens with a CPU index tensor, and `--enforce-eager` cost more than the drafter returned); vLLM v0.29.0 fixed Gemma 4 MTP under CUDA graphs (vllm-project/vllm#53884). Measured 2026-09-10 on the RTX PRO 6000: 193.8 / 1,339 tok/s against a 114 / 843 baseline — single-stream is a shade behind DFlash (203), aggregate a shade ahead (1,203). DFlash remains the recommendation for the 12b; MTP is now a working alternative, not a broken one.

## RTX 5090 (32 GB)

[docker-compose.rtx5090.yml](docker-compose.rtx5090.yml) serves the default 26B-A4B on a single RTX 5090, at the full 262,144-token context, with [MTP](#enabling-speculative-decoding) speculative decoding on. Select it with `COMPOSE_FILE=docker-compose.rtx5090.yml` in `.env`. It assumes a headless machine with the card dedicated to the model. The other four models run on the card too, the 31B only without vision; see [Other models on the 5090](#other-models-on-the-5090).

The stock `docker-compose.yml` does not start on a 32 GB card: after the 16.13 GiB of weights and the activation peak at `--max-num-batched-tokens 32768`, it is left with 7.47 GiB of KV cache against the 8.85 GiB one 262k request needs. The 5090 file lowers that flag to 8192. The activation peak scales with it, and every GiB saved goes to the KV pool (measured without MTP):

| `--max-num-batched-tokens` | KV cache | KV tokens | Full-context requests |
| --- | --- | --- | --- |
| 32768 (stock) | 7.47 GiB | — (refuses to start) | — |
| 16384 | 9.80 GiB | 448,664 | 1.71x |
| **8192** (shipped) | **10.92 GiB** | **687,683** | **2.62x** |
| 4096 | 11.45 GiB | 888,194 | 3.39x |

Prefill is unaffected: single 8k, 32k and 100k-token prompts take 183 ms, 1.11 s and 6.9 s at both 8192 and 16384. The cost is the one [Tuning](#tuning) describes for any value below 32768: when long prompts arrive alongside decoding streams, a prefill chunk that fills the whole budget stalls them, so p99 ITL is worse than with the stock file. 32768 does not fit on this card anyway, and 8192 stalls half as long as the 4096 floor. `--gpu-memory-utilization` stays at 0.92. Under load, usage runs past that budget (29,998 MiB) but stays clear of the card: it peaked at 31,092 of 32,607 MiB with MTP (30,334 without), with no failures, across three bursts: 64 concurrent 8k-token prompts, 16 concurrent prompts of four images each at `max_soft_tokens` 1120, and two concurrent 250k-token prompts. A desktop session or any other CUDA process on the card eats into that ~1.5 GiB of headroom.

Measured 2026-10-02 on vLLM v0.30.0 with [bench.py](bench.py), same method as [Benchmarks](#benchmarks). The kernels are the same as on the RTX PRO 6000, and smoke tests pass (chat, thinking, auto tool calls, vision, 20/20 on the accuracy set):

| Config | Single-stream decode | Aggregate, 8 streams | KV cache capacity |
| --- | --- | --- | --- |
| **gemma-4-26B-A4B + MTP** (shipped) | **279.0 tok/s** | **1,484 tok/s** | 607,331 tokens |
| gemma-4-26B-A4B, no MTP | 220.5 tok/s | 1,122 tok/s | 687,683 tokens |

Without MTP that is on par with the RTX PRO 6000 on the same vLLM release (216.3 / 1,150, see [vLLM v0.30.0](#vllm-v0300)). MTP is on here, unlike in the other compose files, because on this card it costs little: the drafter (weights, CUDA graphs and activations) takes 1.3 GiB out of the KV cache, still leaving 2.32x the full context, for +27% single-stream and +32% at 8 streams. Acceptance was 53% at k=2. The KV figure is for a start that loads the compile cache; the first start on a machine compiles from scratch, profiles a higher peak, and gets 570,250 tokens. To turn it off, delete the `--speculative-config` lines.

### Other models on the 5090

Start from the 5090 file, swap the model as in [Swapping models](#swapping-models), and replace the drafter in `--speculative-config` with the model's own (or delete it). Same day, release and method as above, at `--max-num-batched-tokens 8192` except for the 31B:

| Config | Context | Single-stream decode | Aggregate, 8 streams | KV cache capacity |
| --- | --- | --- | --- | --- |
| gemma-4-31B, text-only + MTP | 49,152 | 103.8 tok/s | 704 tok/s | 50,372 tokens (1.02x) |
| gemma-4-31B, text-only | 65,536 | 57.3 tok/s | 427 tok/s | 78,847 tokens (1.20x) |
| **gemma-4-26B-A4B + MTP** (shipped) | 262,144 | **279.0 tok/s** | **1,484 tok/s** | 607,331 tokens (2.32x) |
| gemma-4-26B-A4B | 262,144 | 220.5 tok/s | 1,122 tok/s | 687,683 tokens (2.62x) |
| gemma-4-12b + DFlash | 262,144 | 218.1 tok/s | 1,305 tok/s | 758,979 tokens (2.90x) |
| gemma-4-12b + MTP | 262,144 | 202.7 tok/s | 1,423 tok/s | 943,756 tokens (3.60x) |
| gemma-4-12b | 262,144 | 127.4 tok/s | 953 tok/s | 1.02M tokens (3.88x) |
| gemma-4-E4B + MTP | 131,072 | 227.3 tok/s | 1,700 tok/s | 1.84M tokens (14.1x) |
| gemma-4-E4B | 131,072 | 205.4 tok/s | 1,369 tok/s | 1.93M tokens (14.7x) |
| gemma-4-E2B + MTP | 131,072 | 340.8 tok/s | 2,569 tok/s | 5.86M tokens (44.7x) |
| gemma-4-E2B | 131,072 | 307.4 tok/s | 1,891 tok/s | 6.08M tokens (46.4x) |

*In parentheses: how many requests at the full configured context fit at once. DFlash is `z-lab/gemma4-12B-it-DFlash` at k=8 with the `triton_attn` backend, as in [Enabling speculative decoding](#enabling-speculative-decoding); MTP is the size's `google/gemma-4-*-it-assistant` at k=2.*

The card decodes at the RTX PRO 6000's speed: every config measured on both is within 4% of its [v0.30.0](#vllm-v0300) figures, single-stream and at 8 streams, most a little ahead. Both cards are GB202 chips with 1.79 TB/s of memory bandwidth. Each config passed the load test above, with the long prompts cut to fit the context (2 x 125k on the E-series; 2 x 60k, or 2 x 45k with MTP, and no image phase on the 31B), and peaked between 30,268 and 31,272 MiB with no failures. Smoke-test results do not depend on the card: the 26B and 31B pass everything (the 31B has no vision to test), while the 12b, E4B and E2B miss one of the 20 accuracy questions (spelling "necessary" backwards) and read the vision test image's "vLLM" as "LLM", and the E4B returns no reasoning with thinking on. These are model-level misses, not 5090 ones; the 12b gets the image right with DFlash on.

- **The 31B fits only without vision.** With the vision tower, its 23.55 GiB of weights leave 3.0 GiB of KV cache at `--max-num-batched-tokens 8192`, enough for one 7,120-token request, and 9,056 at 4096, the lowest the vision encoder allows. `--language-model-only` drops the vision tower (22.47 GiB of weights) and with it that floor. At `--max-num-batched-tokens 2048` that leaves 5.37 GiB. vLLM puts the longest request that fits at 89k tokens; at `--max-model-len 65536` the pool holds 1.2 full-length requests. Image requests then fail with a 400.
- **Speculative decoding on the 31B: MTP fits, DFlash does not.** The DFlash drafter (2.86 GiB) runs out of memory while loading. The MTP drafter (`google/gemma-4-31B-it-assistant`, 0.88 GiB) fits, at 3.93 GiB of KV, enough for `--max-model-len 49152`. It is worth +81% single-stream and +65% at 8 streams, at the cost of a quarter of the context.
- **On the 12b, DFlash wins single-stream and MTP wins at 8 streams**, as on the RTX PRO 6000. MTP also costs less KV.
- **MTP on the E-series** is worth +11% single-stream on both, and +24% (E4B) and +36% (E2B) at 8 streams. The drafters cost under 5% of the KV cache.
- **The 31B configs fail their first start.** The first start of any config compiles from scratch and profiles a higher activation peak than later starts, which load the compile cache from the `vllm_cache` volume. On the other models that costs a few percent of the KV cache, but on the 31B it is over 2 GiB: the first start gets 3.11 GiB of KV (1.5 with MTP), less than one full-length request needs, and vLLM exits. The compile cache survives, so the next start comes up with the full figure; with `restart: unless-stopped` that happens on its own. Changing `--max-model-len` invalidates the cache, so expect it again after editing it.

## Two-Spark cluster

[run_cluster.sh](run_cluster.sh) joins two DGX Sparks into a Ray cluster and serves one Gemma 4 model across both GPUs with tensor parallelism (TP=2), NCCL riding RDMA (RoCE) over the dedicated 200 GbE link between them. A single GB10 already runs every model in this repo; what the second Spark buys is headroom — twice the aggregate memory bandwidth and twice the KV-cache memory — at the price of putting every per-layer all-reduce on the wire. What that trade returns varies enormously by model, from +80% single-stream decode on the dense 31B to a marginal +4% aggregate on the E2B — see [the cluster benchmarks](#2x-dgx-spark-tp2-cluster).

```bash
./run_cluster.sh head                # on the head Spark
./run_cluster.sh worker              # on the other Spark (head IP from .env, or pass it)
./run_cluster.sh serve 31b           # back on the head; 31b | 26b-a4b | 12b | e4b | e2b
```

More Sparks join the same way: start `worker` on each, then give `serve` the tensor-parallel size as its last argument (`./run_cluster.sh serve 31b dflash 4`). Four Sparks pay off on the dense 31B and 12b; the 26B-A4B gains KV capacity rather than speed, and the E-series runs better on two. See [4x DGX Spark](#4x-dgx-spark-tp4-cluster).

`status` reports tmux/container/Ray/API state on any node; `stop` tears down that node's half. Everything long-running lives in detached tmux sessions (`ray-node` holds the Ray container on each node, `vllm-serve` holds the engine on the head), so an SSH drop doesn't take the cluster down; engine output is mirrored to `~/vllm-cluster-serve.log`. Set `CLUSTER_HEAD_IP` in `.env` (see `.env.example`) to the head's IP *on the 200G link*; `CLUSTER_IF` and `CLUSTER_HCA` default to the Spark's 200G netdev and its RoCE device. The image ships without Ray, so each node pip-installs it at container start (~1 min, needs internet). Once healthy, the API is on port 8000 of the head node, same as the single-node compose.

The serve profile reuses the single-Spark tuning unchanged — fp8 KV cache, utilization 0.78 (a per-node fraction; the host-starvation ceiling it protects doesn't move by adding a machine), `--max-num-batched-tokens 32768` — with vision and both parsers enabled. Speculative decoding is an optional second argument to `serve` (e.g. `./run_cluster.sh serve 31b dflash`; `dflash` or `mtp`) — see [Speculative decoding on the cluster](#speculative-decoding-on-the-cluster) for what each buys, and for the 26B-A4B, which cannot start with a drafter on stock v0.30.0. Three things are specific to this setup:

- **The 26B-A4B runs its expert layers with expert parallelism** (`--enable-expert-parallel`, already wired into the script), not tensor parallelism. TP=2 would halve each expert's intermediate size (704 → 352), making the fused gate|up weight 704 rows — not a multiple of the NVFP4 128-row scale tile — and the fast `FLASHINFER_CUTLASS` MoE backend refuses to pad gated weights (`NotImplementedError` at load). With EP each rank instead holds 64 whole experts at the single-GPU shape, the fast backend loads as-is, and attention is still TP=2.
- **On v0.30.0 the 26B-A4B runs on the cluster without a drafter only.** `serve 26b-a4b dflash` needs the v0.29.0 image (`VLLM_IMAGE=vllm/vllm-openai:v0.29.0` on the head and every worker), and `serve 26b-a4b mtp` does not start on any release yet. Both are upstream bugs in how the drafter inherits expert parallelism; see [Speculative decoding on the cluster](#speculative-decoding-on-the-cluster).
- **Multi-node needs vLLM v0.27.0 or later.** v0.26.0's shared-memory message queue — which the engine uses to drive cross-node workers — can lose a reader wakeup notification, parking the engine and both workers forever on queues that have data; the engine then dies minutes later with "RPC call to sample_tokens timed out". v0.27.0 bounds the park time so a lost wakeup recovers within ~5 s. Single-node deployments don't exercise this path at risk.

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

- **MTP** — Google's `*-it-assistant` checkpoints, a 4-layer decoder that shares the target's KV cache. Published for all five models; the E-series assistants (the only speculative option for E4B/E2B, which have no DFlash drafter) are measured on the [DGX Spark](#speculative-decoding-on-the-spark) and the [two-Spark cluster](#speculative-decoding-on-the-cluster).
- **DFlash** — [z-lab](https://z-lab.ai/projects/dflash/)'s block-diffusion drafter, which proposes a whole block in one pass. Published for the 31B, 26B-A4B and 12b.

Same method and hardware as above, `num_speculative_tokens` tuned per method (2 for MTP, 8 for DFlash):

| Model | Baseline | MTP | DFlash |
| --- | --- | --- | --- |
| gemma-4-31B | 56.3 / 417 | 101.4 / 697 | **120.3 / 702** |
| **gemma-4-26B-A4B** (default) | 221 / 1,122 | **283.1 / 1,462** | 278.9 / 1,233 |
| gemma-4-12b | 114 / 843 | 193.8 / 1,339 (v0.29.0) | **203.1 / 1,203** |

*single-stream tok/s / aggregate tok/s at 8 streams.*

Which drafter wins depends on the model, and the gains are much larger for the dense models than for the MoE:

- **Dense 31B and 12b: DFlash, and it is transformative.** The 31B goes from 56 to 120 tok/s (+114%) and the 12b from 114 to 203 (+78%). Decode on a dense model is bandwidth-bound, so amortizing weight reads across several tokens per forward pass pays off enormously. A 31B at 120 tok/s is a genuinely different serving proposition from one at 56.
- **26B-A4B MoE: MTP, mostly for concurrency.** Single-stream is a near tie (283 vs 279), but MTP delivers 1,462 tok/s aggregate against DFlash's 1,233. The MoE only activates 3.8B parameters per token, so it is far less bandwidth-starved to begin with and has less headroom to reclaim.
- **Draft length matters, and shorter is usually better.** DFlash's model card suggests `num_speculative_tokens: 15`; on the 26B-A4B, 8 measured faster (278.9 vs 261.6) and 4 gave the best aggregate (1,268). Acceptance decays steeply with position — the first four positions account for ~1.14 of the 1.29 tokens accepted per step — so a long draft mostly buys verification cost. For MTP, 2 beat 3 (283.1 vs 277.4): vLLM re-runs the same MTP layer for each extra token, which lowers acceptance.
- **Speculative throughput is strongly prompt-dependent.** With DFlash on the 26B-A4B, the technical prompts in `bench.py` decode ~50% faster than the prose ones (309 vs 199 tok/s); MTP is steadier (260–303). Predictable text drafts well and narrative does not, so a 3-run mean swings with the prompts it happens to hit — this is why `bench.py` now averages one run per prompt across all 8.

Both drafters cost KV cache capacity: on the 26B-A4B, 1.95M tokens baseline drops to 1.72M with DFlash.

### vLLM v0.30.0

Re-verified 2026-10-01 on the RTX PRO 6000 with the method above. Each v0.30.0 run is paired with a v0.29.0 run of the same config on the same day, so the comparison is not against the months-old tables above. Every config boots with the same kernels and gives the same smoke-test results as on v0.29.0 (chat, thinking, auto tool calls, vision, and a 20-question accuracy set). That includes the E4B and E2B, whose KV-shared layers now load a separate `q_proj` weight instead of a fused `qkv_proj`.

| Config | v0.29.0 | v0.30.0 | Single-stream |
| --- | --- | --- | --- |
| gemma-4-31B | 56.7 / 421 | 55.9 / 416 | −1.4% |
| **gemma-4-26B-A4B** (default) | **228.6 / 1,152** | **216.3 / 1,150** | **−5.4%** |
| gemma-4-12b | 123.0 / 923 | 123.8 / 930 | +0.7% |
| gemma-4-E4B | 215.1 / 1,458 | 204.6 / 1,399 | −4.9% |
| gemma-4-E2B | 320.3 / 2,040 | 297.0 / 1,874 | −7.3% |
| 31B + DFlash | 121.1 / 665 | 120.1 / 711 | −0.8% |
| 26B-A4B + MTP | 284.1 / 1,463 | 271.7 / 1,428 | −4.4% |
| 12b + DFlash | 220.2 / 1,294 | 220.2 / 1,291 | 0.0% |
| 12b + MTP | 193.3 / 1,334 | 197.0 / 1,391 | +1.9% |

*single-stream tok/s / aggregate tok/s at 8 streams.*

**v0.30.0 decodes a single stream more slowly on every Gemma 4 model except the 12b.** The cost is a roughly fixed 0.25 ms per decode step, not a percentage. That is why it is largest on the fastest model (E2B, −7%), barely visible on the 31B (−1%), and smaller under speculative decoding, which pays it once per several tokens. It is not run-to-run noise: four more 26B-A4B runs, alternating versions and starting with v0.30.0, gave 217.0 and 216.8 tok/s on v0.30.0 against 228.8 and 229.0 on v0.29.0. The 12b uses a different model class (`Gemma4UnifiedForConditionalGeneration`) and shows no slowdown. Aggregate throughput mostly holds, except on the E-series (E2B −8%, E4B −4%).

A torch profile of the 26B-A4B locates the time but not its cause. On v0.30.0 each decode step replays a CUDA graph with the same GEMM, MoE and attention kernels and launch configurations, about 60 fewer small elementwise kernels, and slightly *less* total kernel time. What grows is the idle time between consecutive kernels inside the graph, from ~0.1 µs to ~0.4 µs; across the step's ~800 kernels, that is the whole 0.25 ms. The 12b runs at ~0.4 µs between kernels on both versions: it never had the tighter spacing, so it has nothing to lose, and the same drop in small kernels gives it its slight gain. The GPU driver, CUDA runtime, torch and Triton are the same in both images. Clocks during the benchmark are identical, and nothing else runs on the GPU.

The v0.29.0 controls match the v0.26.0 tables above to within about 1%, with two exceptions: the 26B-A4B runs 3% faster, and the 12b 8% faster (123 against 114, and 220 against 203 with DFlash). Those tables have not been re-measured.

### DGX Spark (GB10)

Same method on a DGX Spark using [docker-compose.spark.yml](docker-compose.spark.yml) (`--gpu-memory-utilization 0.78`, unified memory). Re-measured 2026-10-03 on vLLM v0.30.0, with each config paired with a v0.29.0 run on the same Spark that night. Where a config ran more than once on a version, the figure is the mean:

| Model | Single-stream decode | Aggregate, 8 streams | KV cache capacity | v0.29.0 |
| --- | --- | --- | --- | --- |
| gemma-4-31B | 8.9 tok/s | 68 tok/s | 453k tokens | 8.8 / 68 |
| **gemma-4-26B-A4B** (default) | **47.7 tok/s** | **216 tok/s** | 2.06M tokens | 47.5 / 216 |
| gemma-4-12b | 21.5 tok/s | 171 tok/s | 1.69M tokens | 21.5 / 170 |
| gemma-4-E4B | 43.8 tok/s | 361 tok/s | 4.59M tokens | 43.4 / 357 |
| gemma-4-E2B | 78.9 tok/s | 625 tok/s | 14.2M tokens | 78.6 / 613 |

*v0.29.0 column: single-stream / aggregate tok/s.*

The GB10's unified LPDDR5X gives roughly a fifth of the discrete card's memory bandwidth, and decode is bandwidth-bound, so everything scales down accordingly. The same conclusion holds even more strongly here: the dense 31B is not interactive on this hardware at 8.9 tok/s, while the 26B-A4B MoE stays comfortably usable. Speculative decoding changes that picture substantially — see [below](#speculative-decoding-on-the-spark).

**v0.30.0's single-stream slowdown does not show up on the Spark.** On the RTX PRO 6000 the release costs ~0.25 ms per decode step (see [vLLM v0.30.0](#vllm-v0300)). At the E2B's 12.7 ms per token here that would be −2%; the E2B measures +0.4%. Every config in this section, with or without a drafter, is within 2% of v0.29.0 single-stream and within 4% at 8 streams. The exception is 31B + DFlash, which varies between starts for an unrelated reason (below). The smoke tests give the same results on both versions: chat, auto tool calls, vision and the accuracy check pass on every model, and with thinking on, the E4B and E2B return no reasoning. Against the v0.26.0 figures this table replaces (2026-07-28), decode is 1–5% faster and KV capacity 2–5% lower.

**Expect up to 5% run-to-run spread on these figures, and do not read a small delta as a regression.** Repeating the same configuration on the same hardware moved these numbers by as much as 5%, and two Sparks running the same config never agreed better than 1–2%. Treat any sub-5% difference — against a previous release, another machine, or another run — as noise until it reproduces. These figures were taken on machines that had been under continuous load for hours, which is what a running server actually delivers.

#### Speculative decoding on the Spark

Both drafters work on GB10, and speculative decoding is the single highest-leverage flag available on this hardware. Same v0.30.0 runs as above, with DFlash at `num_speculative_tokens: 8` and MTP at 2:

| Config | Single-stream | Aggregate, 8 streams | vs no drafter (single / agg) | Acceptance | KV cache |
| --- | --- | --- | --- | --- | --- |
| gemma-4-31B + DFlash | **17.9–19.6 tok/s** | 106–113 tok/s | +101–120% / +56–67% | 20.9–21.6% | 393k tokens |
| gemma-4-31B + MTP | 17.4 tok/s | 112 tok/s | +96% / +65% | 53.7% | 435k tokens |
| **gemma-4-26B-A4B + MTP** | **65.8 tok/s** | **273 tok/s** | +38% / +26% | 53.7% | 2.01M tokens |
| gemma-4-26B-A4B + DFlash | 55.1 tok/s | 219 tok/s | +16% / +1% | 15.8% | 1.77M tokens |
| gemma-4-12b + DFlash | **41.3 tok/s** | 239 tok/s | +92% / +40% | 18.3% | 1.48M tokens |
| gemma-4-12b + MTP | 38.6 tok/s | **264 tok/s** | +80% / +55% | 50.9% | 1.64M tokens |
| gemma-4-E4B + MTP | 64.0 tok/s | 441 tok/s | +46% / +22% | 23.1% | 4.49M tokens |
| gemma-4-E2B + MTP | 112.4 tok/s | 701 tok/s | +43% / +12% | 25.4% | 14.0M tokens |

*The 31B + DFlash row spans six starts on three Sparks; see the last bullet.*

- **MTP is the MoE's drafter here, by a wider margin than on the RTX.** On the 26B-A4B it beats DFlash by 19% single-stream (65.8 against 55.1 tok/s) and 25% at 8 streams, and it costs less KV (2.01M tokens against 1.77M). DFlash's +16% single-stream and +1% at 8 streams on this model are the smallest gains of any config here. The MoE already amortizes weight reads across the batch, so there is little left for DFlash's long draft to reclaim.
- **On the dense models the split is the same as on the RTX: DFlash for one stream, MTP for eight.** The 12b runs at 41.3 tok/s with DFlash against 38.6 with MTP, and at 264 tok/s at 8 streams with MTP against 239. On the 31B the two are closer: DFlash's 17.9–19.6 tok/s against MTP's 17.4 single-stream, and a tie at 8 streams. MTP's drafters are smaller and leave more KV cache on both.
- **The 31B becomes usable.** 8.9 tok/s is below reading speed; 18–20 tok/s is not. That does not make it the right default — the MoE is faster untuned than the 31B is with a drafter — but it moves the dense 31B from "not worth serving here" to "viable if you need its quality".
- **The E-series assistants are worth far more here than on faster hardware:** +46% single-stream on the E4B and +43% on the E2B, against +11% on the [RTX 5090](#other-models-on-the-5090) and +20% / +13% on the [cluster](#speculative-decoding-on-the-cluster), at the same ~23–25% acceptance. No DFlash drafter exists for these two, so MTP is their only option.
- **31B + DFlash moves by up to 10% between starts, and FlashInfer's autotuner decides which way.** Six v0.30.0 starts on three Sparks gave 17.9 to 19.6 tok/s (three v0.29.0 starts gave 18.0 to 19.8), while every other config stayed within 3% between starts. On first start, vLLM has FlashInfer time several kernel tactics for each GEMM shape bucket and caches the winners under `/root/.cache/vllm/flashinfer_autotune_cache` in the `vllm_cache` volume, and later starts reuse them. For this config, the FP4 GEMM tactics for the 16-token bucket decide the speed. That is the bucket one stream's verify pass falls into: the current token plus 8 drafts. Starting from a slow Spark's cache, changing only those two entries made the next two starts fast (19.6 and 19.0 tok/s); the unmodified cache gave 17.9. Identical Sparks on the same image picked different tactics, so the choice is effectively timing noise at first start, and it sticks for as long as the cache does. The same spread is why this config has no version comparison.

`num_speculative_tokens: 8` was carried over from the RTX tuning rather than re-swept on GB10. Per-position acceptance decays steeply (0.81, 0.54, 0.36, 0.18, 0.12, 0.06, 0.05, 0.03, measured on v0.26.0), so positions 6–8 contribute only ~6% of accepted tokens. A shorter draft would likely trade a little throughput for meaningfully less verification work.

### 2x DGX Spark (TP=2 cluster)

Same method on two Sparks joined by [run_cluster.sh](run_cluster.sh) — TP=2 over the 200 GbE RoCE link, `--gpu-memory-utilization 0.78` per node, no speculative decoding. Re-measured 2026-10-03 on v0.30.0; the comparison columns are against the v0.30.0 single-Spark table above:

| Model | Single-stream decode | Aggregate, 8 streams | KV cache capacity | vs one Spark (single / agg / KV) |
| --- | --- | --- | --- | --- |
| gemma-4-31B | 15.5 tok/s | 116 tok/s | 1.10M tokens | +74% / +70% / 2.44x |
| **gemma-4-26B-A4B** (default) | **61.6 tok/s** | **299 tok/s** | 4.58M tokens | +29% / +38% / 2.22x |
| gemma-4-12b | 34.5 tok/s | 254 tok/s | 3.13M tokens | +60% / +49% / 1.86x |
| gemma-4-E4B | 59.4 tok/s | 422 tok/s | 9.75M tokens | +36% / +17% / 2.13x |
| gemma-4-E2B | 95.6 tok/s | 615 tok/s | 14.9M tokens | +21% / −2% / 1.05x |

Every figure is within 2.5% of the v0.27 pre-release nightly measurement this table replaces (2026-07-28), and the 26B-A4B gave 61.0 / 291 tok/s on v0.29.0. The gains over one Spark are a few points smaller than that table showed (+74% against +80% on the 31B) because the single-Spark baseline rose, not because the cluster slowed.

How much the second Spark buys tracks how starved the model was in the first place:

- **The dense models gain most, and the biggest gains the most of all.** Decode on a dense model is memory-bandwidth-bound, and TP=2 splits every weight read across two memory systems: +74% single-stream on the 31B (8.9 → 15.5 tok/s), +60% on the 12b. The 31B also frees the most weight memory per node, which is why its KV capacity scales furthest (2.44x).
- **The MoE gains +29% / +38% through expert parallelism.** With only 3.8B active parameters it is far less bandwidth-starved than the dense models, so there is less for the cluster to reclaim — but at 62 tok/s single-stream and 4.58M tokens of KV cache it is still the model to serve, now with 2.2x the capacity.
- **The E2B is the floor of the approach.** +21% single-stream, −2% aggregate — and its KV cache capacity barely grows (1.05x). That last one is architectural, not noise: the E2B has a single KV head (`num_key_value_heads: 1`), which TP=2 must replicate on both ranks, so each node still pays the full per-token KV cost and total capacity stays at one node's worth. The E4B's two KV heads split exactly, hence its 2.13x. Below ~4B effective parameters, the wire costs about what the second memory system pays back.

#### Speculative decoding on the cluster

Same method again, with the drafters from the single-node setup (`num_speculative_tokens` 8 for DFlash, 2 for MTP) served via `run_cluster.sh serve <model> <dflash|mtp>`, on v0.30.0:

| Model + drafter | Baseline | With drafter | Single | Aggregate | Acceptance | KV cache |
| --- | --- | --- | --- | --- | --- | --- |
| gemma-4-31B + DFlash | 15.5 / 116 | **33.5 / 167** | +116% | +44% | 21.5% | 1.10M → 973k |
| gemma-4-31B + MTP | 15.5 / 116 | 28.2 / 179 | +82% | +55% | 53.6% | 1.10M → 1.06M |
| gemma-4-26B-A4B + MTP † | 61.6 / 299 | **77.5 / 369** | +26% | +23% | 53.4% | 4.58M → 4.46M |
| gemma-4-26B-A4B + DFlash † | 61.6 / 299 | 77.3 / 286 | +25% | −4% | 16.0% | 4.58M → 3.92M |
| gemma-4-12b + DFlash | 34.5 / 254 | **61.8 / 296** | +79% | +17% | 18.2% | 3.13M → 2.74M |
| gemma-4-12b + MTP | 34.5 / 254 | 55.6 / 378 | +61% | +49% | 53.4% | 3.13M → 3.02M |
| gemma-4-E4B + MTP | 59.4 / 422 | **71.1 / 448** | +20% | +6% | 22.7% | 9.75M → 9.51M |
| gemma-4-E2B + MTP | 95.6 / 615 | **108.2 / 645** | +13% | +5% | 25.4% | 14.9M → 14.6M |

*single-stream tok/s / aggregate tok/s at 8 streams; baseline = cluster without a drafter. † Does not start on stock v0.30.0; measured with two upstream fixes patched into the image (see below). Stock v0.29.0 runs 26B-A4B + DFlash at 75.6 / 294.*

- **On stock v0.30.0 the 26B-A4B cannot start with either drafter on the cluster.** Its NVFP4 experts force `--enable-expert-parallel` (see [Two-Spark cluster](#two-spark-cluster)), and v0.30.0 copies that flag into the dense drafter's parallel config. `serve 26b-a4b dflash` and `serve 26b-a4b mtp` both exit within a minute with `Number of experts in the model must be greater than 0 when expert parallelism is enabled`. vLLM fixed this on main in vllm-project/vllm#56930 (2026-09-17), after the v0.30.0 branch was cut. v0.29.0 doesn't have the bug, so DFlash runs there as-is: start the head and every worker with `VLLM_IMAGE=vllm/vllm-openai:v0.29.0`. MTP fails on v0.29.0 too, later in startup and with the same message. The Gemma 4 speculator rebuilds the draft's `VllmConfig` around the assistant's model config, and that config fails the same check (vllm-project/vllm#56936, still open). The † rows ran on v0.30.0 with both fixes patched in.
- **With both fixes, MTP is the MoE's best drafter on the cluster too.** It matches DFlash single-stream (77.5 against 77.3 tok/s), delivers 29% more at 8 streams (369 against 286), and leaves more KV cache (4.46M tokens against 3.92M). Once a release carries both fixes, `serve 26b-a4b mtp` is the configuration to run.
- **Drafter and cluster gains multiply only roughly, and MTP's shrink most.** Multiply a config's single-Spark drafter gain by the model's cluster gain and you get a prediction for this table. DFlash lands within ±9% of it: the 31B inside its range, the 26B-A4B 9% over, the 12b 7% under. MTP lands 7–21% under on every model, the E-series most: the E4B and E2B assistants are worth +46% and +43% on one Spark but +20% and +13% here. The earlier finding that the two stacked almost perfectly came from three DFlash configs on a v0.27 nightly.
- **The dense 31B with DFlash at 33.5 tok/s is still the two-Spark flagship**: 3.8x its single-Spark baseline (8.9), and past the point where its quality is usable interactively. Four Sparks take it to 48.0 ([below](#4x-dgx-spark-tp4-cluster)). MTP gives up 16% single-stream on this model but wins at 8 streams (179 against 167).
- **The other rows are within 5% of the v0.27 nightly** except the 12b + DFlash, 7% below its 66.7 tok/s there with acceptance down from 20.5% to 18.2%. On one Spark that drafter accepts 18.3% on both v0.29.0 and v0.30.0.

### 4x DGX Spark (TP=4 cluster)

The same script spans all four Sparks: run `worker` on each of the other three, then give `serve` the size, e.g. `./run_cluster.sh serve 31b dflash 4`. Same method, measured 2026-10-04 on v0.30.0, with the four Sparks on one 200 GbE network. The 26B-A4B runs expert parallelism across all four (32 experts per rank):

| Model | Single-stream decode | Aggregate, 8 streams | KV cache capacity | vs TP=2 (single / agg / KV) | vs one Spark (single / agg / KV) |
| --- | --- | --- | --- | --- | --- |
| gemma-4-31B | 24.8 tok/s | 157 tok/s | 2.41M tokens | +60% / +35% / 2.18x | +179% / +131% / 5.32x |
| **gemma-4-26B-A4B** (default) | **65.6 tok/s** | **298 tok/s** | 7.57M tokens | +6% / ±0% / 1.65x | +38% / +38% / 3.67x |
| gemma-4-12b | 44.9 tok/s | 275 tok/s | 5.04M tokens | +30% / +8% / 1.61x | +109% / +61% / 2.99x |
| gemma-4-E4B | 65.9 tok/s | 379 tok/s | 9.98M tokens | +11% / −10% / 1.02x | +50% / +5% / 2.18x |
| gemma-4-E2B | 93.9 tok/s | 501 tok/s | 15.2M tokens | −2% / −19% / 1.02x | +19% / −20% / 1.07x |

With drafters, served as `serve <model> <dflash|mtp> 4`:

| Model + drafter | Baseline | With drafter | Single | Aggregate | Acceptance | KV cache | vs TP=2, same drafter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-4-31B + DFlash | 24.8 / 157 | **48.0 / 198** | +94% | +27% | 21.8% | 2.41M → 2.14M | +43% / +19% |
| gemma-4-31B + MTP | 24.8 / 157 | 39.8 / 221 | +60% | +41% | 53.1% | 2.41M → 2.31M | +41% / +23% |
| gemma-4-26B-A4B + MTP † | 65.6 / 298 | **78.1 / 381** | +19% | +28% | 52.4% | 7.57M → 7.42M | +1% / +3% |
| gemma-4-26B-A4B + DFlash † | 65.6 / 298 | 80.7 / 309 | +23% | +4% | 16.0% | 7.57M → 6.61M | +4% / +8% |
| gemma-4-12b + DFlash | 44.9 / 275 | **71.3 / 306** | +59% | +11% | 18.2% | 5.04M → 4.51M | +15% / +3% |
| gemma-4-12b + MTP | 44.9 / 275 | 62.4 / 370 | +39% | +34% | 51.7% | 5.04M → 4.91M | +12% / −2% |
| gemma-4-E4B + MTP | 65.9 / 379 | 66.0 / 388 | ±0% | +2% | 23.1% | 9.98M → 9.75M | −7% / −13% |
| gemma-4-E2B + MTP | 93.9 / 501 | 87.4 / 515 | −7% | +3% | 25.9% | 15.2M → 14.9M | −19% / −20% |

*Single-stream / aggregate tok/s, as in the TP=2 table. † With the same two upstream fixes patched in; neither 26B-A4B drafter starts on stock v0.30.0.*

- **The dense models keep scaling, by less with each doubling.** The 31B goes 8.9 → 15.5 → 24.8 tok/s single-stream from one Spark to two to four (+74%, then +60%), the 12b 21.5 → 34.5 → 44.9 (+60%, then +30%). At 8 streams the second doubling buys much less: +35% on the 31B and +8% on the 12b. Each doubling halves the weight reads per node, but every layer still ends in an all-reduce over the network, and that takes a growing share of each step, more so on the smaller model.
- **The 31B with DFlash reaches 48.0 tok/s**, 5.4x its single-Spark baseline and 43% faster than on two Sparks. For scale, one RTX PRO 6000 runs the same config at 120 tok/s. MTP gives up 17% single-stream on this model and again wins at 8 streams (221 against 198 tok/s).
- **The 26B-A4B has stopped scaling.** TP=4 adds 6% single-stream over TP=2 and nothing at 8 streams, and with a drafter it is 1–4% faster than on two Sparks. What the extra two Sparks buy the MoE is KV capacity, 7.57M tokens, not speed: one Spark with MTP already runs it at 65.8 tok/s, 16% short of four.
- **On the 26B-A4B, DFlash now edges MTP single-stream** (80.7 against 78.1 tok/s), but MTP still delivers 23% more at 8 streams (381 against 309) and leaves more KV cache, so it remains the drafter to run once a release carries both fixes.
- **The E-series runs better on two Sparks than on four.** The E4B gains 11% single-stream over TP=2 but loses 10% at 8 streams, and the E2B loses on both (−2% / −19%). With 2–4B effective parameters there is little work left per node to split, while every layer still waits on an all-reduce across four nodes. Their MTP assistants stop paying off as well: ±0% on the E4B and −7% on the E2B (86.5 tok/s on a second start), which leaves both 7–19% behind MTP on two Sparks. The fastest Spark setup for the E4B is two Sparks with MTP (71.1 tok/s), and for the E2B a single Spark with MTP (112.4). The E2B is also the one config here that wavered within a run: a second start held 94 tok/s for three of its eight runs, then sagged to 79–88.
- **Drafter gains shrink again from TP=2 to TP=4.** On the 31B, DFlash is worth +94% here against +116% on two Sparks, and MTP +60% against +82%; the 12b's drafters drop the same way.
- **KV capacity follows the models' global-attention KV heads.** The 31B's pool grows 2.18x over TP=2, more than the node count, because its four global KV heads split one per rank and each rank also holds a quarter of the weights. The 26B-A4B has two global KV heads and the 12b one, so every rank already stored one full head per global layer at TP=2 and still does at TP=4. Only their sliding-window layers split further, and their pools grow 1.65x and 1.61x. The E4B (two KV heads) and the E2B (one) were already at one head per rank at TP=2 and hold little weight memory to free, so their pools grow just 1.02x.

### Serving under concurrency

The tables above measure one client, or eight. This one sweeps concurrency and prompt length with `vllm bench serve` across three configurations and the two models worth serving, to answer what those tables cannot: how many people can this serve at once, and what does it feel like while it does.

Method: `--dataset-name random`, 1024 output tokens per request, `--ignore-eos` so every request generates exactly that many, `--seed 0`, all requests submitted at once (`--request-rate inf`). The prefix cache is flushed between every shape — the seeded dataset generates identical prompts for consecutive shapes, so without a flush each shape prefills out of the previous one's KV blocks, which inflated throughput by 37% and understated TTFT by 12x in testing. (The flush route needs the server started with `VLLM_SERVER_DEV_MODE=1`.) The 2x Spark rows ran on the v0.27 pre-release nightly the cluster pinned at the time, whose `vllm bench serve` is a Rust reimplementation; the classic "Serving Benchmark Result" table is what is reported, same as the other rows.

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
- On GPUs with less memory, lower `--max-num-batched-tokens` before `--max-model-len`: on a 32 GB card, going from 32768 to 8192 is enough to keep the full 262k context — see [RTX 5090 (32 GB)](#rtx-5090-32-gb). Below that, lower `--max-model-len`; the full 262k context is the main memory consumer after the weights.
- `--max-num-seqs 64` is sized for a workstation serving a handful of concurrent clients; raise it for heavier batch serving. `--max-num-batched-tokens 32768` is a different matter — it has been swept and should be left alone, for the reasons below.
- **`--max-num-batched-tokens 32768` is the right default on both machines, but for different reasons.** Swept across 4096/8192/32768 on all five models. On the RTX PRO 6000 lowering it is simply pointless: the best any lower value bought was +2.6% throughput. On the DGX Spark it is a real trade — 4096 is worth **+7% to +14%** on the dense models (31B +14%, 12b +12%, E2B +9.4%, E4B +7%), because a smaller GEMM is more efficient against unified LPDDR5X. The 26B-A4B MoE is the exception on both machines, gaining at most ~3% (two runs measured +0.3% and +3.1%, which brackets the run-to-run noise) — so the default model is the one with the least to gain from tuning this.
- **What lowering it costs is tail latency, everywhere: p99 ITL gets 5–15x worse.** The mechanism is that a chunked-prefill step blocks decoding requests only when it consumes the whole token budget. Once the budget exceeds the prompt, prefill and decode co-schedule in the same step and the stall stops existing rather than merely getting shorter — which is why 32768 is not on the same curve as 8192 and 4096 at all. Between those two the usual model does hold: halving the chunk halves the stall, measured at 1.84–2.39x across every model and both machines. Lowering MNBT is also a *capacity* lever, buying 1.9–3.7x the KV cache since peak activation memory falls with chunk size.
- **The one case where that trade is worth taking is the dense 31B**, which is the only model without comfortable KV headroom (1.6x its own 262k context on the RTX PRO 6000, 1.8x on the Spark; the other four have 6x–112x). On the Spark at 8k prompts, dropping it to 4096 gives 2.8x the KV cache, 19% lower TTFT and 14% more throughput at c32 — but takes p99 ITL from 234 ms to 3,233 ms. Reasonable for long-context batch work, wrong for interactive serving, which is what the default targets.
- **4096 is the floor for any Gemma 4 model.** These are multimodal, and vLLM refuses to start when `--max-num-batched-tokens` is below the encoder's per-item budget: `max_tokens_per_mm_item (2496) is larger than max_num_batched_tokens`. The exception is a text-only server: `--language-model-only` drops the vision encoder and the floor with it, which is how the 31B fits on a [32 GB card](#other-models-on-the-5090).
- Vision detail per image is tunable per request: `"mm_processor_kwargs": {"max_soft_tokens": 1120}` (default 280; 70 for cheap thumbnails).
- Speculative decoding is not enabled by default, but both a DFlash and an MTP drafter exist for these models and are worth adding — see [Enabling speculative decoding](#enabling-speculative-decoding).
- On both these GPUs vLLM serves Gemma 4 with the Triton attention backend, not FlashAttention. Gemma 4's head dimensions differ between sliding-window (256) and global (512) layers, which needs FA4; FA3/FA4 are built for neither sm_120 (RTX PRO 6000) nor sm_121 (GB10), so vLLM logs `FA4 not available, forcing TRITON_ATTN backend` and falls back. There is nothing to tune here — it is the only working backend for this combination — but it explains why `--attention-backend flash_attn` fails, and why a DFlash drafter needs its own `"attention_backend"` entry (see [Enabling speculative decoding](#enabling-speculative-decoding)).

## License

[MIT](LICENSE)
