#!/usr/bin/env bash
#
# Run Gemma 4 across two DGX Sparks as one Ray cluster (tensor parallel = 2).
#
# A single GB10 runs every model in this repo (see docker-compose.spark.yml);
# what a second Spark buys is headroom — twice the aggregate memory bandwidth
# and twice the KV-cache memory — at the price of putting every tensor-parallel
# all-reduce on the wire between the machines. That wire must be the dedicated
# 200 GbE link (~25 GB/s), not the management LAN: this script pins NCCL, Gloo,
# Ray and vLLM to it and takes the RDMA (RoCE) path rather than TCP.
#
# Usage — each command runs on the Spark it describes:
#
#   ./run_cluster.sh head                 # on the head node, first
#   ./run_cluster.sh worker [head_ip]     # on the second node, once the head is up
#   ./run_cluster.sh serve [MODEL] [SPEC] # on the head node: start vLLM
#                                         #   MODEL: 31b | 26b-a4b (default) |
#                                         #          12b | e4b | e2b
#                                         #   SPEC:  dflash | mtp (optional;
#                                         #          see README for which
#                                         #          drafter fits which model)
#   ./run_cluster.sh status               # any node: tmux/container/ray/API state
#   ./run_cluster.sh stop                 # any node: tear down this node's half
#
# Everything long-running lives in a detached tmux session, so an SSH drop
# doesn't take the cluster down: `ray-node` holds the Ray container, and on the
# head `vllm-serve` holds the engine (`tmux attach -t vllm-serve` to watch it,
# ctrl-b d to detach). Engine output is also mirrored to ~/vllm-cluster-serve.log.
#
# The worker finds the head via CLUSTER_HEAD_IP in .env (or the environment),
# or as the optional second argument. Overridable knobs, all with defaults that
# match a stock two-Spark pairing:
#
#   CLUSTER_HEAD_IP   head node's IP on the 200G link (worker + serve need it)
#   CLUSTER_IF        200G interface name        (default enP2p1s0f1np1)
#   CLUSTER_HCA       its RDMA device for RoCE   (default roceP2p1s0f1)
#   VLLM_IMAGE        container image            (default: same tag as compose)
#
# Multi-node needs v0.27.0 or later. v0.26.0's shm_broadcast message queue —
# which the executor uses to drive cross-node workers — could lose a reader
# wakeup notification, leaving the engine and both workers parked forever on
# queues that have data (nondeterministic; the engine then dies with "RPC
# call to sample_tokens timed out"). The fix (SHM_READER_RECHECK_INTERVAL_MS
# bounds the park time so a lost notify recovers within ~5 s) shipped in
# v0.27.0; this script briefly pinned a nightly by SHA while waiting for it.
#
# The image ships without Ray, so each node pip-installs ray[default] at
# container start (~1 min, needs internet) — same bootstrap as vLLM's own
# examples/run_cluster.sh. The same install pins transformers==5.14.1: the
# v0.27.x images bundle 5.15.0, which raises on Gemma 4's per-layer head_dim
# at startup (vllm-project/vllm#51744; fixed on main, not yet in a release).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

# .env (gitignored) supplies HF_TOKEN and optionally CLUSTER_* overrides.
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$SCRIPT_DIR/.env"
  set +a
fi

CLUSTER_IF="${CLUSTER_IF:-enP2p1s0f1np1}"
CLUSTER_HCA="${CLUSTER_HCA:-roceP2p1s0f1}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.27.1}"
RAY_PORT=6379
CONTAINER=ray-node
NODE_SESSION=ray-node
SERVE_SESSION=vllm-serve
SERVE_LOG="$HOME/vllm-cluster-serve.log"

usage() {
  sed -n '2,51p' "$SELF" | sed 's/^# \{0,1\}//'
  exit 1
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

node_ip() {
  ip -4 -o addr show "$CLUSTER_IF" 2>/dev/null | awk '{print $4}' | cut -d/ -f1
}

require_node_ip() {
  NODE_IP="$(node_ip)"
  [ -n "$NODE_IP" ] || die "no IPv4 address on $CLUSTER_IF — is this a Spark with the 200G link up? (override with CLUSTER_IF=...)"
}

# ---------------------------------------------------------------------------
# Node lifecycle: `head` / `worker` spawn a tmux session running `_node`,
# which blocks on the Ray container for the life of the cluster.
# ---------------------------------------------------------------------------

start_node() {
  local role="$1" head_ip="$2"
  require_node_ip
  command -v tmux >/dev/null || die "tmux is required"

  tmux has-session -t "$NODE_SESSION" 2>/dev/null && die "tmux session '$NODE_SESSION' already exists — './run_cluster.sh stop' first"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

  echo "Starting $role node: $NODE_IP on $CLUSTER_IF (head: $head_ip, image: $VLLM_IMAGE)"
  tmux new-session -d -s "$NODE_SESSION" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "$SELF _node $role $head_ip $NODE_IP $CLUSTER_IF $CLUSTER_HCA $VLLM_IMAGE"
  echo "Ray $role starting in tmux session '$NODE_SESSION' (attach: tmux attach -t $NODE_SESSION)"
}

_node() {
  local role="$1" head_ip="$2" this_ip="$3" ifname="$4" hca="$5" image="$6"

  local ray_cmd="pip install -q --root-user-action=ignore 'ray[default]>=2.9' transformers==5.14.1 && ray start --block"
  if [ "$role" = head ]; then
    ray_cmd+=" --head --node-ip-address=$this_ip --port=$RAY_PORT --include-dashboard=false"
  else
    ray_cmd+=" --address=$head_ip:$RAY_PORT --node-ip-address=$this_ip"
  fi

  # --network host        Ray + NCCL talk host-to-host directly
  # --device /dev/infiniband + memlock  the RoCE RDMA path (verbs needs to pin pages)
  # NCCL_IB_HCA / GID 3   the 200G port's RDMA device, RoCEv2 GID
  # NCCL_PROTO/ALGO       Simple/Ring measured stable on the GB10 pair; LL
  #                       protocols are tuned for NVLink-class latency
  # RAY_memory_monitor_refresh_ms=0     Ray's OOM killer reads *unified* memory
  #                       pressure and would kill healthy workers mid-run
  # No --rm: after a crash the stopped container keeps its logs for
  # `docker logs ray-node`; the next start removes it.
  docker run \
    --entrypoint /bin/bash \
    --network host \
    --name "$CONTAINER" \
    --shm-size 10.24g \
    --gpus all \
    --device /dev/infiniband \
    --ulimit memlock=-1 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v vllm_cluster_cache:/root/.cache/vllm \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e VLLM_HOST_IP="$this_ip" \
    -e MASTER_ADDR="$head_ip" \
    -e NCCL_SOCKET_IFNAME="$ifname" \
    -e GLOO_SOCKET_IFNAME="$ifname" \
    -e TP_SOCKET_IFNAME="$ifname" \
    -e UCX_NET_DEVICES="$ifname" \
    -e OMPI_MCA_btl_tcp_if_include="$ifname" \
    -e NCCL_IB_HCA="$hca" \
    -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_PROTO=Simple \
    -e NCCL_ALGO=Ring \
    -e RAY_memory_monitor_refresh_ms=0 \
    -e RAY_enable_worker_prestart=0 \
    "$image" -c "$ray_cmd"
}

# ---------------------------------------------------------------------------
# Serve: exec vLLM into the head's Ray container, in its own tmux session.
# ---------------------------------------------------------------------------

active_ray_nodes() {
  docker exec "$CONTAINER" ray status 2>/dev/null | awk '/^Active:/,/^Pending:/' | grep -c 'node_' || true
}

cmd_serve() {
  local model="${1:-26b-a4b}" spec="${2:-}"
  case "$model" in
    31b|26b-a4b|12b|e4b|e2b) ;;
    *) die "unknown model '$model' (want: 31b, 26b-a4b, 12b, e4b or e2b)" ;;
  esac
  case "$spec" in
    ""|dflash|mtp) ;;
    *) die "unknown speculative mode '$spec' (want: dflash or mtp)" ;;
  esac

  docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true \
    || die "Ray container isn't running here — run './run_cluster.sh head' on this node first"
  tmux has-session -t "$SERVE_SESSION" 2>/dev/null && die "tmux session '$SERVE_SESSION' already exists — './run_cluster.sh stop' first"

  local nodes
  nodes="$(active_ray_nodes)"
  [ "$nodes" -ge 2 ] || die "Ray reports $nodes active node(s), need 2 — start './run_cluster.sh worker' on the other Spark and wait for it to join (check: docker exec $CONTAINER ray status)"

  echo "Starting vLLM ($model${spec:+ +$spec}) on a $nodes-node Ray cluster"
  tmux new-session -d -s "$SERVE_SESSION" \
    -e VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-0}" \
    "$SELF _serve $model $spec 2>&1 | tee $SERVE_LOG"
  echo "Engine starting in tmux session '$SERVE_SESSION' (attach: tmux attach -t $SERVE_SESSION; log: $SERVE_LOG)"
  echo "First boot downloads weights and compiles — expect several minutes before http://localhost:8000/health goes green."
}

_serve() {
  local model="$1" spec="${2:-}" repo served len dflash_repo mtp_repo
  local extra=() exec_env=()
  case "$model" in
    31b)     repo=unsloth/gemma-4-31B-it-NVFP4     served=gemma-4-31b     len=262144
             dflash_repo=z-lab/gemma-4-31B-it-DFlash
             mtp_repo=google/gemma-4-31B-it-assistant ;;
    # The MoE must run its expert layers with expert parallelism (64 whole
    # experts per rank), not tensor parallelism: TP=2 would halve each
    # expert's intermediate size (704 -> 352), making the fused gate|up
    # weight 704 rows — not a multiple of the NVFP4 128-row scale tile —
    # and the fast FLASHINFER_CUTLASS backend refuses to pad gated MoE
    # weights (NotImplementedError at load). With EP each rank keeps the
    # single-GPU expert shape and the fast backend loads as-is; attention
    # is still TP=2.
    26b-a4b) repo=unsloth/gemma-4-26B-A4B-it-NVFP4 served=gemma-4-26b-a4b len=262144
             extra=(--enable-expert-parallel)
             dflash_repo=z-lab/gemma-4-26B-A4B-it-DFlash
             mtp_repo=google/gemma-4-26B-A4B-it-assistant ;;
    # MTP is broken on the 12b: its gemma4_unified compute_logits suppresses
    # tokens with a CPU index tensor, which CUDA graph capture rejects, and
    # --enforce-eager lands below baseline. DFlash is the right drafter here.
    12b)     repo=unsloth/gemma-4-12b-it-NVFP4     served=gemma-4-12b     len=262144
             dflash_repo=z-lab/gemma4-12B-it-DFlash ;;
    # E-series max out at 131,072 context; vLLM refuses to start above it.
    # No DFlash drafters exist for them — the assistant (MTP) is their only
    # speculative option.
    e4b)     repo=unsloth/gemma-4-E4B-it-NVFP4     served=gemma-4-e4b     len=131072
             mtp_repo=google/gemma-4-E4B-it-assistant ;;
    e2b)     repo=unsloth/gemma-4-E2B-it-NVFP4     served=gemma-4-e2b     len=131072
             mtp_repo=google/gemma-4-E2B-it-assistant ;;
  esac

  # Optional speculative decoding, mirroring the README's single-node flags:
  # DFlash pins "attention_backend": "triton_attn" (flash_attn cannot start
  # against fp8 KV + Gemma 4's PrefixLM attention, and vLLM only propagates
  # the target's forced backend to MTP drafters, not DFlash ones). MTP needs
  # the V2 model runner — the V1 proposer only shares target embeddings with
  # EAGLE-family drafters, and the assistant's pre_projection needs them.
  case "$spec" in
    dflash)
      [ -n "${dflash_repo:-}" ] || die "no DFlash drafter published for $model"
      extra+=(--speculative-config "{\"method\": \"dflash\", \"model\": \"$dflash_repo\", \"num_speculative_tokens\": 8, \"attention_backend\": \"triton_attn\"}")
      ;;
    mtp)
      [ -n "${mtp_repo:-}" ] || die "MTP not available for $model (broken on the 12b — see the case comment above)"
      extra+=(--speculative-config "{\"model\": \"$mtp_repo\", \"num_speculative_tokens\": 2}")
      exec_env+=(-e VLLM_USE_V2_MODEL_RUNNER=1)
      ;;
  esac

  # The engine flags mirror docker-compose.spark.yml — same model config, same
  # tuning rationale (see the comments there) — plus the two cluster flags:
  # tensor-parallel-size 2 across the machines, executed over Ray.
  #
  # --gpu-memory-utilization stays at the Spark's tuned 0.78: it is a
  # *per-node* fraction of unified memory and the OS-starvation ceiling it
  # protects doesn't move by adding a second machine.
  #
  # --max-num-batched-tokens stays at the swept 32768 (see README Tuning):
  # on Gemma 4 lowering it buys a few percent of throughput at best and costs
  # 5-15x in tail latency, and TP=2 doesn't change that trade.
  #
  # VLLM_SERVER_DEV_MODE=1 (export it, or set it in .env) adds vLLM's dev
  # routes — the serving benchmark POSTs /reset_prefix_cache between shapes.
  # Off by default: the routes are unauthenticated.
  exec docker exec \
    -e VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-0}" \
    "${exec_env[@]}" \
    "$CONTAINER" vllm serve \
    "$repo" \
    --served-model-name "$served" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.78 \
    --max-model-len "$len" \
    --max-num-seqs 64 \
    --max-num-batched-tokens 32768 \
    --reasoning-parser gemma4 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --limit-mm-per-prompt '{"image": 4}' \
    "${extra[@]}"
}

# ---------------------------------------------------------------------------
# stop / status
# ---------------------------------------------------------------------------

cmd_stop() {
  if tmux has-session -t "$SERVE_SESSION" 2>/dev/null; then
    tmux kill-session -t "$SERVE_SESSION"
    echo "killed tmux session $SERVE_SESSION"
  fi
  if docker rm -f "$CONTAINER" >/dev/null 2>&1; then
    echo "removed container $CONTAINER"
  fi
  if tmux has-session -t "$NODE_SESSION" 2>/dev/null; then
    tmux kill-session -t "$NODE_SESSION"
    echo "killed tmux session $NODE_SESSION"
  fi
}

cmd_status() {
  echo "== tmux =="
  tmux ls 2>/dev/null | grep -E "^($NODE_SESSION|$SERVE_SESSION):" || echo "(no cluster sessions)"
  echo "== container =="
  docker ps -a --filter "name=^$CONTAINER$" --format '{{.Names}}  {{.Status}}  {{.Image}}' | grep . || echo "(none)"
  if docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
    echo "== ray =="
    docker exec "$CONTAINER" ray status 2>&1 | sed -n '/^Node status/,/^Resources/p'
    echo "== api =="
    if curl -sf --max-time 3 http://localhost:8000/health >/dev/null 2>&1; then
      echo "http://localhost:8000 healthy"
    else
      echo "not serving on :8000 (normal on the worker, or while the engine boots)"
    fi
  fi
}

# ---------------------------------------------------------------------------

case "${1:-}" in
  head)
    require_node_ip
    start_node head "${CLUSTER_HEAD_IP:-$NODE_IP}"
    ;;
  worker)
    head_ip="${2:-${CLUSTER_HEAD_IP:-}}"
    [ -n "$head_ip" ] || die "worker needs the head's 200G IP: './run_cluster.sh worker <head_ip>' or CLUSTER_HEAD_IP in .env"
    start_node worker "$head_ip"
    ;;
  serve)  cmd_serve "${2:-}" "${3:-}" ;;
  status) cmd_status ;;
  stop)   cmd_stop ;;
  _node)  shift; _node "$@" ;;
  _serve) shift; _serve "$@" ;;
  *) usage ;;
esac
