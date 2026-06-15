# Project status & overview

_Last updated: 2026-06-15_

## What this project is

A **self-hosted deployment of [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video)**
(Meituan's 13.6B text/image/video generation model) on a **rented RunPod GPU**,
using the model's **built-in Streamlit web UI**.

It is NOT a fork of the model. It's a thin **ops/deployment layer** — scripts +
docs that make the model cheap and painless to run, with the key property that you
**install once and respin in ~2 minutes** (no reinstalling every time).

## Why it's built this way (decisions made)

| Decision | Choice | Reason |
|----------|--------|--------|
| Run location | **RunPod**, self-hosted | Local PC (RTX 2060, 6GB) can't run a 13.6B model (needs 24GB+). RunPod network volumes make "install once" clean. |
| Persistence | **Network volume** at `/workspace` | Holds conda env (with compiled flash-attn) + repo + ~60GB weights. Survives pod deletion → no reinstall. |
| GPU | **A100 80GB** (default), H100 for speed | 13.6B dense model is comfortable at 80GB; avoids INT8/OOM hassle. |
| UI | **Model's built-in Streamlit** (`run_streamlit.py`) | Ships with the repo; no need to build our own. |
| Features | T2V, I2V, continuation, long video, **avatar** | All weights downloaded so every mode works. |
| Docker | Optional `docker/Dockerfile` | For later: bake env for instant cold-starts. Not required for the volume workflow. |

## Current state

- ✅ Deployment scripts written (`runpod/`)
- ✅ Optional Dockerfile written (`docker/`)
- ✅ Local UI preview built & verified (`preview_ui.py`) — confirms the layout
- ✅ Docs written (`docs/`, root `README.md`)
- ⏳ **Not yet deployed to RunPod** — that's the next step (see `SETUP_RUNPOD.md`)
- ⏳ Scripts written from the repo's documented commands but **not yet run on a
  live pod** — first real run is where to watch for version hiccups.

## The "install once, respin fast" model (core idea)

```
NETWORK VOLUME (/workspace) — persists, ~$6/mo for 100GB
  miniconda3/  (env + compiled flash-attn)   <- the slow part, done once
  LongCat-Video/  (repo)
  weights/  (~60GB, done once)
        ▲ attach/detach
  GPU POD — disposable, pay-per-hour, delete when idle
        runs ./runpod/start.sh -> Streamlit UI on port 8080
```

## File map

| Path | Purpose |
|------|---------|
| `README.md` | Top-level playbook / quick reference |
| `preview_ui.py` | **Local** UI mock (no GPU/torch needed) — preview the layout only |
| `runpod/env.sh` | Shared paths + `activate_env` helper |
| `runpod/01_setup.sh` | One-time install onto the volume (idempotent) |
| `runpod/02_download_weights.sh` | One-time ~60GB weight download (3 repos) |
| `runpod/start.sh` | **Respin entrypoint** — launches the real Streamlit UI |
| `docker/Dockerfile` | Optional baked-image alternative |
| `docs/PROJECT_STATUS.md` | This file — overview & decisions |
| `docs/SETUP_RUNPOD.md` | **What to do next** — full RunPod walkthrough |
| `docs/GPU_AND_COSTS.md` | GPU pick + cost model |

## Local preview (free, no GPU)

To re-see the UI layout anytime on this PC:
```powershell
pip install streamlit
streamlit run preview_ui.py   # -> http://localhost:8501
```
It cannot generate (no model) — it's purely the interface. The real generation
happens on the pod.

## Open questions / possible future work

- Optionally build a single **custom unified UI** (Gradio) instead of the stock
  Streamlit, if the default UX feels limiting.
- Optionally finish + push the `docker/Dockerfile` image for instant cold-starts.
- Decide whether to keep the volume long-term (~$6/mo) or tear down between
  heavy-use periods.
