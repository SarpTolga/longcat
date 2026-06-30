"""
LongCat-Video — continuous-clip Streamlit UI (auto-chaining in the browser).

The stock run_streamlit.py generates single clips and only continues a video you
manually download and re-upload. This app keeps the last generated clip in session
state and adds a "Continue this video" button, so you build one long, continuous
video without ever leaving the browser. Each continuation reuses the last 13 frames
for a seamless seam (the model's generate_vc), and everything is stitched into a
single growing video shown in the page.

It reuses the SAME pipeline loading + generate_t2v / generate_vc calls as
longcat_generate.py (which mirror the repo's run_demo_long_video.py).

Launched by runpod/start_chain.sh on the pod. Single-GPU.
"""
import os
import sys
import tempfile

import torch
import streamlit as st
from torchvision.io import write_video

# Reuse the proven loader + helpers from the sibling CLI generator.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longcat_generate import (  # noqa: E402
    load_pipe,
    NUM_FRAMES,
    NUM_COND_FRAMES,
    DEFAULT_NEGATIVE,
    _resolution_to_hw,
    _to_uint8_cpu,
    _as_tensor,
)


class _Cfg:
    """Minimal stand-in for the argparse Namespace load_pipe expects."""
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.context_parallel_size = 1
        self.enable_compile = False
        self.seed = 42


@st.cache_resource(show_spinner="Loading the 13.6B model (first time only, ~1-2 min)…")
def get_pipe(checkpoint_dir):
    pipe, generator, global_rank = load_pipe(_Cfg(checkpoint_dir))
    return pipe, generator


def _write_temp(video):
    path = os.path.join(tempfile.gettempdir(), f"chain_{video.shape[0]}f.mp4")
    write_video(path, _to_uint8_cpu(video), fps=16)
    return path


st.set_page_config(page_title="🎬 LongCat — Continuous", page_icon="🎬", layout="wide")
st.title("🎬 LongCat-Video — Continuous clips")
st.caption("Generate a clip, then keep pressing **Continue** to extend it seamlessly. "
           "Everything is stitched into one growing video.")

# --- session state ---
ss = st.session_state
ss.setdefault("video", None)       # full stitched video tensor (T, H, W, C)
ss.setdefault("tail", None)        # last clip, fed into the next continuation
ss.setdefault("n_segments", 0)

with st.sidebar:
    st.header("⚙️ Settings")
    model_dir = st.text_input("Model Dir", "./weights/LongCat-Video")
    resolution = st.selectbox("Resolution", ["480p", "720p"], index=0)
    distill = st.checkbox("Distill mode (faster, lower quality)", value=False)
    seed = st.number_input("Seed", min_value=0, max_value=2**32 - 1, value=42, step=1)
    steps = 16 if distill else st.number_input("Inference steps", 1, 100, 50)
    guidance = 1.0 if distill else st.number_input("Guidance scale", 0.0, 20.0, 4.0)
    st.markdown("---")
    if st.button("🗑️ Reset (start a new video)", use_container_width=True):
        ss.video = ss.tail = None
        ss.n_segments = 0
        st.rerun()

prompt = st.text_area("Prompt", height=100,
                      placeholder="Describe the video (and how it should continue)…")
negative = "" if distill else st.text_area("Negative prompt", value=DEFAULT_NEGATIVE, height=70)

col1, col2 = st.columns(2)
start_clicked = col1.button("🚀 Generate base clip", type="primary",
                            use_container_width=True, disabled=not prompt.strip())
continue_clicked = col2.button("🔗 Continue this video", use_container_width=True,
                               disabled=ss.tail is None or not prompt.strip())

if start_clicked or continue_clicked:
    pipe, generator = get_pipe(model_dir)
    height, width = _resolution_to_hw(resolution)
    generator.manual_seed(int(seed) + ss.n_segments)  # vary per segment

    with st.spinner("Generating… (this is the GPU working — minutes, not seconds)"):
        if start_clicked:
            clip = pipe.generate_t2v(
                prompt=prompt, negative_prompt=negative,
                height=height, width=width, num_frames=NUM_FRAMES,
                num_inference_steps=int(steps), guidance_scale=float(guidance),
                generator=generator,
            )[0]
            clip = _as_tensor(clip)
            ss.video = clip
            ss.tail = clip
            ss.n_segments = 1
        else:  # continue
            clip = pipe.generate_vc(
                video=ss.tail, prompt=prompt, negative_prompt=negative,
                resolution=resolution, num_frames=NUM_FRAMES,
                num_cond_frames=NUM_COND_FRAMES,
                num_inference_steps=int(steps), guidance_scale=float(guidance),
                generator=generator, use_kv_cache=True, offload_kv_cache=False,
                enhance_hf=True,
            )[0]
            clip = _as_tensor(clip)
            # Drop the conditioning overlap so we don't repeat frames in the stitch.
            ss.video = torch.cat([ss.video, clip[NUM_COND_FRAMES:]], dim=0)
            ss.tail = clip
            ss.n_segments += 1

# --- output ---
if ss.video is not None:
    st.subheader(f"🎥 Current video — {ss.n_segments} segment(s), {ss.video.shape[0]} frames")
    path = _write_temp(ss.video)
    st.video(path)
    with open(path, "rb") as f:
        st.download_button("⬇️ Download", f, file_name="continuous.mp4",
                           mime="video/mp4", use_container_width=True)
    st.info("Tweak the prompt and press **🔗 Continue** to extend it, or **Reset** to start over.")
else:
    st.markdown("_Generate a base clip to begin, then chain continuations._")
