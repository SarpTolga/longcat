# What to do next — full RunPod setup

This is the step-by-step to take the project from "code in a GitHub repo" to
"generating video in a browser." Follow it top to bottom the first time. After
that, only **Part D (respin)** matters.

> Read `GPU_AND_COSTS.md` first if you want the GPU/cost reasoning.
> Read `PROJECT_STATUS.md` for the big picture.

---

## Part 0 — Prerequisites (one time, on your own machine)

- [ ] A **RunPod account** with some credit loaded → https://runpod.io
- [ ] This project **pushed to a GitHub repo** (you're doing this now). You'll
      `git clone` it onto the pod.
- [ ] (Optional) A **Hugging Face account** — the LongCat weights are public, so
      usually no token is needed; only add one if a download asks for auth.

---

## Part A — Create the persistent network volume (one time)

This is the disk that survives forever and saves you from reinstalling.

1. RunPod dashboard → **Storage** → **Network Volume** → **New**.
2. **Size: 100 GB** (≈ 60GB weights + env + headroom).
3. **Region:** choose one that has A100/H100 availability (check the GPU list in
   that region first).
4. Name it e.g. `longcat-vol`. Create it.

> Cost: ~$5–8/month while it exists, billed even with no pod running. This is the
> price of never reinstalling. Delete it later if you stop using the project.

---

## Part B — Deploy a pod against that volume (one time for setup)

1. RunPod → **Pods** → **Deploy**.
2. **GPU:** `A100 80GB` (recommended) or `H100 80GB`.
3. **Template:** any **PyTorch + CUDA 12.4** template (e.g. a `runpod/pytorch`
   image). We install our own conda env on the volume, so the base only needs
   CUDA drivers + git/wget.
4. **Network Volume:** attach `longcat-vol` → it mounts at **`/workspace`**.
5. **Expose HTTP Ports:** add **`8080`** (this is the Streamlit UI).
6. Deploy. Wait for it to be "Running", then open the **Web Terminal** (or SSH).

---

## Part C — Install everything (one time, ~20–40 min total)

In the pod's Web Terminal:

```bash
# 1. Get this project onto the pod
cd /workspace
git clone <YOUR_GITHUB_REPO_URL> longcat-ui
cd longcat-ui

# 2. Fix Windows line endings + make scripts executable (important!)
sed -i 's/\r$//' runpod/*.sh && chmod +x runpod/*.sh

# 3. Install env + deps onto the volume (SLOW: flash-attn compiles, ~15–35 min)
./runpod/01_setup.sh

# 4. Download the ~60GB of weights onto the volume (a few minutes)
./runpod/02_download_weights.sh

# 5. Launch the web UI
./runpod/start.sh
```

Then in the RunPod pod panel: **Connect → HTTP Service [Port 8080]** → opens the
real LongCat-Video Streamlit UI in your browser. Generate something to confirm. 🎬

> If the UI's "Model Dir" field isn't auto-resolved, set it to
> `./weights/LongCat-Video`.

---

## Part D — The respin (every time after setup, ~2 min, NO reinstall)

When you come back later:

1. RunPod → **Deploy** a new pod (or restart a stopped one), **attach the same
   `longcat-vol` volume**, expose port **8080**.
2. Web Terminal:
   ```bash
   cd /workspace/longcat-ui && ./runpod/start.sh
   ```
3. **Connect → HTTP 8080** → generate.
4. **When done: STOP or DELETE the pod** (don't just close the browser tab) —
   GPU billing stops; the volume keeps everything.

That's the whole loop. The conda env, compiled flash-attn, repo, and weights are
already on the volume, so there's nothing to reinstall.

---

## Power-user / advanced

CLI generation (batch, multi-GPU, avatar) — activate the env first:

```bash
cd /workspace/longcat-ui && source runpod/env.sh && activate_env
cd "$REPO_DIR"
```

- **Text-to-Video:**
  `torchrun run_demo_text_to_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile`
- **Image-to-Video:**
  `torchrun run_demo_image_to_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile`
- **Video continuation:**
  `torchrun run_demo_video_continuation.py --checkpoint_dir=./weights/LongCat-Video --enable_compile`
- **Long video:**
  `torchrun run_demo_long_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile`
- **Avatar 1.5 (single audio → video, INT8 + distill, 2 GPUs):**
  ```bash
  torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py \
    --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video-Avatar-1.5 \
    --stage_1=at2v --input_json=assets/avatar/single_example_1.json \
    --num_segments=5 --ref_img_index=10 --mask_frame_range=3 \
    --use_distill --model_type avatar-v1.5 --use_int8
  ```

Multi-GPU on any task: deploy a 2-GPU pod, prefix with `--nproc_per_node=2`, add
`--context_parallel_size=2`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `$'\r': command not found` | Line endings. Re-run `sed -i 's/\r$//' runpod/*.sh`. |
| `flash_attn` build fails | Ensure CUDA 12.4 base image; `01_setup.sh` installs `ninja` first. Re-run the script (idempotent). |
| CUDA out of memory | Use a bigger GPU (A100/H100 80GB), or for avatar add `--use_int8 --use_distill`. |
| UI loads but generate errors on weights path | Set Model Dir to `./weights/LongCat-Video`. |
| Pod gone but I didn't lose work | Correct — that's the design. Reattach the volume and run `start.sh`. |
| Still being billed | You only stopped Streamlit, not the pod. **Stop/Delete the pod** in RunPod. |

---

## Cleanup / cost control

- **Done for the day:** stop/delete the pod. Keep the volume.
- **Done for weeks:** optionally back up/delete the volume to stop the ~$6/mo
  (you'd re-run Part C next time).
- See `GPU_AND_COSTS.md` for the full cost model.
