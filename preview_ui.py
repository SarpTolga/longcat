"""
LOCAL UI PREVIEW of LongCat-Video's Streamlit app.

This is a faithful visual mock of the real `run_streamlit.py` interface so you can
click around the layout WITHOUT torch / flash_attn / CUDA / weights. It does NOT
generate anything — the Generate button just shows a fake message.

Run locally (only one tiny package needed):
    pip install streamlit
    streamlit run preview_ui.py

The real app on the GPU pod is run_streamlit.py — same layout, but it actually
loads the 13.6B model and generates video.
"""

import streamlit as st

st.set_page_config(page_title="🎬 LongCatVideo Generator", page_icon="🎬", layout="wide")

# --- preview-only banner (not in the real app) ---
st.warning(
    "👀 **UI PREVIEW ONLY** — this is a local mock to show the layout. "
    "No model is loaded and nothing will generate. The real app runs on the GPU pod."
)

# --- Header ---
st.title("🎬 LongCatVideo Generator")
st.caption("Supports Text-to-Video (T2V), Image-to-Video (I2V), and Video Continuation (VC) generation")

st.text_input("Model Dir", "./weights/LongCat-Video")

with st.expander("📋 Example Prompts"):
    st.markdown("**T2V:** A red fox trotting through a snowy forest at golden hour, cinematic, shallow depth of field.")
    st.markdown("**I2V:** The person in the image slowly turns their head and smiles at the camera.")
    st.markdown("**VC:** Continue the motion smoothly — the car keeps driving down the coastal road.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.selectbox("Mode", ["T2V", "I2V", "VC"], index=0)

    distill = st.checkbox("Enable Distill Mode (Faster Generation)", value=False)
    st.checkbox("Enable Super-Resolution Mode (Low-res first, then upsample)", value=False)

    if mode == "T2V":
        st.number_input("Height", min_value=256, max_value=1024, value=480, step=16)
        st.number_input("Width", min_value=256, max_value=1024, value=832, step=16)
    else:  # I2V / VC
        st.selectbox("Resolution", ["480p", "720p"], index=0)

    # Fixed params (shown read-only to mirror the real app's behavior)
    num_frames = 93
    num_inference_steps = 16 if distill else 50
    guidance_scale = 1.0 if distill else 4.0
    st.markdown("**Fixed parameters**")
    st.text(f"num_frames        = {num_frames}")
    st.text(f"inference_steps   = {num_inference_steps}")
    st.text(f"guidance_scale    = {guidance_scale}")

    st.number_input("Random Seed", min_value=0, max_value=2**32 - 1, value=42, step=1)

# --- Main area: two columns ---
left, right = st.columns(2)

with left:
    st.subheader("📝 Input")
    st.text_area("Positive Prompt", height=100,
                 placeholder="Describe the video you want to generate...")
    st.text_area("Negative Prompt", height=80, disabled=distill,
                 value="" if not distill else "(disabled in distill mode)")

    if mode == "I2V":
        st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    elif mode == "VC":
        st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    generate = st.button("🚀 Generate", type="primary", use_container_width=True)

with right:
    st.subheader("🎥 Output")
    placeholder = st.empty()
    if generate:
        placeholder.info(
            "This is the **preview mock** — no model is loaded, so nothing is "
            "generated. On the GPU pod, the rendered video would appear here with "
            "a Download button below."
        )
    else:
        placeholder.markdown("_Generated video will appear here._")
        st.download_button("⬇️ Download", data=b"", file_name="output.mp4", disabled=True)
