# LongCat-Video on RunPod — self-hosted, with web UI

A complete, **respin-friendly** setup for running Meituan's
[LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) (13.6B) on a
rented GPU, with the built-in **Streamlit web UI**.

**The whole point:** you do the heavy install **once** onto a persistent
**network volume**. After that, killing the pod costs nothing and a fresh pod is
generating again in ~2 minutes — **no reinstalling**.

Supports everything: **text→video, image→video, video continuation, long video,
and avatar (audio-driven)**.

---

## How it works (the model)

```
┌─────────────────────────────────────────────────────────────┐
│  RunPod NETWORK VOLUME  (persists forever, ~$6/mo for 100GB) │
│   /workspace/                                                 │
│     ├── miniconda3/      conda + env + COMPILED flash-attn    │
│     ├── LongCat-Video/   the cloned repo                      │
│     ├── weights/         ~60GB model weights (3 repos)        │
│     └── hf_cache/                                             │
└─────────────────────────────────────────────────────────────┘
              ▲  attach / detach
              │
        ┌─────┴───────┐   ephemeral, pay-per-hour, delete anytime
        │  GPU POD    │   (A100/H100) → runs ./start.sh → Streamlit UI
        └─────────────┘
```

- **Pod** = the GPU. Pay per hour. Delete it the second you're done.
- **Volume** = your installed world. Cheap, always there. Reattach to any new pod.

---

## One-time setup (do this once per volume)

### 1. Create the network volume
RunPod → **Storage** → **New Network Volume**.
- Size: **100 GB** (60GB weights + env + headroom).
- Region: pick one that has the GPUs you want (A100/H100).

### 2. Deploy a pod against the volume
RunPod → **Pods** → **Deploy**.
- **GPU:** A100 80GB (recommended) or H100 80GB. *(see `docs/GPU_AND_COSTS.md`)*
- **Template:** any **PyTorch / CUDA 12.4** template (e.g. a `runpod/pytorch`
  base). We install our own conda env on the volume, so the base just needs
  CUDA drivers + git/wget.
- **Network Volume:** attach the one from step 1 → it mounts at `/workspace`.
- **Expose HTTP port:** `8080` (this is the Streamlit UI port).

### 3. Get these scripts onto the pod
Open the pod's **Web Terminal** (or SSH). Easiest is to keep this folder in a Git
repo and clone it:

```bash
cd /workspace
git clone <your-repo-with-this-folder> longcat-ui
cd longcat-ui
# Safety: strip any Windows CRLF line endings so bash won't choke
sed -i 's/\r$//' runpod/*.sh && chmod +x runpod/*.sh
```

> No repo? Just recreate the four files in `runpod/` on the pod with a text
> editor — they're short.

### 4. Install (the slow part — once)
```bash
./runpod/01_setup.sh            # ~15–35 min (flash-attn compiles)
./runpod/02_download_weights.sh # ~60GB download, a few minutes on datacenter bw
```

### 5. Launch the UI
```bash
./runpod/start.sh
```
Then in RunPod, click the pod's **Connect → HTTP 8080** to open the Streamlit UI
in your browser. 🎬

---

## Every time after that (the respin — ~2 min, no reinstall)

1. Deploy a new pod, **attach the same network volume**, expose port **8080**.
2. Web Terminal:
   ```bash
   cd /workspace/longcat-ui && ./runpod/start.sh
   ```
3. Connect → HTTP 8080 → generate.
4. **Done? Stop/delete the pod.** GPU billing stops. Volume keeps everything.

That's it. The conda env, compiled flash-attn, repo, and weights are all already
on the volume.

---

## Custom-prompt generation + long continuous video (`generate.sh`)

The Streamlit UI generates **single clips** (T2V/I2V) and can continue a video you
upload (VC), but continuity across clips there is **manual** (download → re-upload),
and the repo's own long-video demo has a **hardcoded** prompt.

`runpod/generate.sh` fixes that: your prompt from the CLI, plus a **`long`** mode
that makes a base clip and autoregressively continues it, stitching the segments
into one **continuous** video (each segment reuses the last 13 frames for a seamless
seam). It calls the same pipeline API as the repo's `run_demo_long_video.py`.

```bash
# single clip from your prompt
./runpod/generate.sh --mode t2v \
  --prompt "A red fox trotting through a snowy forest at golden hour" --output fox.mp4

# long, continuous video: base clip + 5 continuations stitched together
./runpod/generate.sh --mode long --segments 5 \
  --prompt "A drone shot gliding over a misty mountain valley at sunrise" --output valley.mp4

# faster (lower quality): distill mode
./runpod/generate.sh --mode t2v --distill --prompt "..." --output out.mp4

# 2-GPU pod
GPUS=2 ./runpod/generate.sh --mode long --context-parallel-size 2 --prompt "..." --output out.mp4
```

Per-segment files (`*_seg00.mp4`, `*_seg01.mp4`, …) are also saved, so a bad stitch
never loses a render. **First run:** smoke-test `--mode long` once — it mirrors the
upstream API but hasn't been run on a live pod yet.

### Or do it in the browser — continuous-clip UI

Prefer clicking to typing CLI flags? `runpod/start_chain.sh` launches our custom
Streamlit app that auto-chains in the browser: generate a base clip, then press
**🔗 Continue** as many times as you like — each press extends the same video
seamlessly and re-stitches it, no download/re-upload. Use this instead of
`start.sh` when you want continuity:

```bash
./runpod/start_chain.sh        # then RunPod Connect -> HTTP 8080
```

(`start.sh` still launches the stock UI for I2V / avatar / single clips.)

---

## CLI generation (power users / batch / avatar)

The Streamlit UI covers the main tasks. For full control (multi-GPU, avatar
flags), use the repo's demo scripts directly. Activate the env first:

```bash
source runpod/env.sh && activate_env
cd "$REPO_DIR"
```

**Text-to-Video**
```bash
torchrun run_demo_text_to_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile
```
**Image-to-Video**
```bash
torchrun run_demo_image_to_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile
```
**Video continuation**
```bash
torchrun run_demo_video_continuation.py --checkpoint_dir=./weights/LongCat-Video --enable_compile
```
**Long video**
```bash
torchrun run_demo_long_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile
```
**Avatar 1.5 — single audio → video** (INT8 + distill, 2 GPUs)
```bash
torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py \
  --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar-1.5 \
  --stage_1=at2v --input_json=assets/avatar/single_example_1.json \
  --num_segments=5 --ref_img_index=10 --mask_frame_range=3 \
  --use_distill --model_type avatar-v1.5 --use_int8
```

Multi-GPU on any task: prefix with `--nproc_per_node=2` and add
`--context_parallel_size=2` (deploy a 2-GPU pod for this).

---

## Files in this project

| Path | What |
|------|------|
| `runpod/env.sh` | Shared paths + `activate_env` helper |
| `runpod/01_setup.sh` | One-time install onto the volume (idempotent) |
| `runpod/02_download_weights.sh` | One-time ~60GB weight download |
| `runpod/start.sh` | **Respin entrypoint** — launches the web UI |
| `runpod/generate.sh` | Custom-prompt CLI generation + long continuous video |
| `runpod/longcat_generate.py` | Parameterized generator behind `generate.sh` |
| `runpod/start_chain.sh` | Launch the **continuous-clip UI** (auto-chaining in the browser) |
| `runpod/app_chain.py` | Our custom Streamlit app: Generate → Continue → Continue… |
| `docker/Dockerfile` | Optional: bake the env into an image for instant cold-starts |
| `docs/GPU_AND_COSTS.md` | GPU pick + cost model + money-saving habits |

---

## Notes / gotchas

- **Line endings:** these `.sh` files must be LF, not CRLF. The `sed` in step 3
  handles it; if you ever see `$'\r': command not found`, re-run that `sed`.
- **Weights path:** `start.sh` symlinks `$WEIGHTS_DIR` → `LongCat-Video/weights`
  so the Streamlit app finds them. If the UI asks for a checkpoint path, point it
  at `./weights/LongCat-Video`.
- **VRAM:** if you OOM on a 24GB card, you need the INT8/distill flags (avatar) or
  a bigger GPU. A100 80GB avoids all of this.
- **Stop the pod**, don't just close the tab — otherwise the GPU keeps billing.
- The optional `docker/Dockerfile` is for later: build once, push to a registry,
  and skip `01_setup.sh` entirely on future pods (weights still come from the volume).
```
