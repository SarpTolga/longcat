# OPTIONAL alternative to the network-volume workflow.
# Bakes the entire environment (incl. compiled flash-attn) into an image so
# cold-starts are instant and the setup is shareable/reproducible.
# Weights are NOT baked in (too big) — still mount them from a volume or
# download at runtime.
#
# Build (needs Docker + ~30 min for flash-attn; no GPU required to BUILD):
#   docker build -t <dockerhub-user>/longcat-video:latest docker/
#   docker push  <dockerhub-user>/longcat-video:latest
# Then point your RunPod template at that image instead of a stock pytorch one.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates build-essential ninja-build ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda \
    && rm /tmp/mc.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda create -y -n longcat-video python=3.10
SHELL ["conda", "run", "--no-capture-output", "-n", "longcat-video", "/bin/bash", "-c"]

RUN git clone --single-branch --branch main \
        https://github.com/meituan-longcat/LongCat-Video /app
WORKDIR /app

RUN pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124
RUN pip install ninja psutil packaging \
    && pip install flash_attn==2.7.4.post1 --no-build-isolation
RUN pip install -r requirements.txt \
    && pip install -r requirements_avatar.txt
RUN conda install -y -c conda-forge librosa ffmpeg
RUN pip install "huggingface_hub[cli]" hf_transfer

ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=8080 \
    WEIGHTS_DIR=/workspace/weights
EXPOSE 8080

# Weights still come from a mounted volume at /workspace/weights
CMD bash -lc 'ln -sfn "$WEIGHTS_DIR" /app/weights && \
  conda run --no-capture-output -n longcat-video \
  streamlit run ./run_streamlit.py --server.address=0.0.0.0 --server.port=${PORT} \
  --server.fileWatcherType none --server.headless=true'
