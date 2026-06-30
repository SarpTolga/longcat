#!/usr/bin/env python3
"""
Parameterized LongCat-Video generator — YOUR prompt, from the command line.

The repo ships only fixed demos (run_demo_*.py): the prompt, length, and output
path are hardcoded, so you can't make your own video without editing source. This
wrapper exposes them as CLI flags and adds an auto-chained LONG mode that keeps
continuity across segments (text->video, then repeatedly continue the last clip).

It uses the SAME pipeline + method calls as the repo's run_demo_long_video.py
(LongCatVideoPipeline.generate_t2v / generate_vc), so the model code path is the
upstream one — only the orchestration around it is ours.

Run via torchrun (it sets RANK / WORLD_SIZE):

  # single clip (~93 frames) from your prompt
  torchrun longcat_generate.py --mode t2v \
    --prompt "A red fox trotting through a snowy forest at golden hour" \
    --output out.mp4

  # long, continuous video: base clip + N continuations stitched together
  torchrun longcat_generate.py --mode long --segments 5 \
    --prompt "A drone shot gliding over a misty mountain valley at sunrise" \
    --output long.mp4

  # 2-GPU pod (faster): add --nproc_per_node=2 and --context-parallel-size 2
  torchrun --nproc_per_node=2 longcat_generate.py --mode long \
    --context-parallel-size 2 --prompt "..." --output long.mp4

NOTE: there is no 80GB GPU to test this on locally; the API calls mirror the
upstream demo, but smoke-test --mode long on your first pod run (T2V is the
low-risk path). Per-segment files are also saved so a bad stitch never loses work.
"""
import os
import argparse
import datetime

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel

# Standard Wan-family negative prompt (same intent as the repo demos). Override
# with --negative-prompt. Ignored automatically when --distill is set.
DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, "
    "still picture, cluttered background, three legs, many people in the background, "
    "walking backwards"
)

NUM_FRAMES = 93        # per clip — matches the repo demos
NUM_COND_FRAMES = 13   # overlap reused to keep continuity between segments


def _resolution_to_hw(resolution: str):
    # LongCat's native aspect: 480p -> 480x832, 720p -> 720x1280.
    return (720, 1280) if resolution == "720p" else (480, 832)


def load_pipe(args):
    # Works both under torchrun (which sets these) and under a plain process such
    # as `streamlit run` (single-GPU): fill in sane single-process defaults.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    rank = int(os.environ.get("RANK", "0"))
    num_gpus = max(torch.cuda.device_count(), 1)
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24)
        )
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    init_context_parallel(
        context_parallel_size=args.context_parallel_size,
        global_rank=global_rank,
        world_size=world_size,
    )
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    ckpt = args.checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt, subfolder="tokenizer", torch_dtype=torch.bfloat16
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        ckpt, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLWan.from_pretrained(
        ckpt, subfolder="vae", torch_dtype=torch.bfloat16
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        ckpt, subfolder="scheduler", torch_dtype=torch.bfloat16
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        ckpt, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16
    )
    if args.enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(args.seed + global_rank)
    return pipe, generator, global_rank


def _to_uint8_cpu(video):
    """write_video wants a CPU uint8 (T, H, W, C) tensor; be defensive about dtype."""
    v = video.detach().cpu()
    if v.dtype != torch.uint8:
        v = (v.clamp(0, 1) * 255).round().to(torch.uint8)
    return v


def _save(video, path, fps, is_writer):
    if is_writer:
        write_video(path, _to_uint8_cpu(video), fps=fps)
        print(f"==> wrote {path}  ({video.shape[0]} frames)")


def main():
    args = _parse_args()
    pipe, generator, global_rank = load_pipe(args)
    is_writer = global_rank == 0

    steps = 16 if args.distill else args.num_inference_steps
    guidance = 1.0 if args.distill else args.guidance_scale
    negative = "" if args.distill else args.negative_prompt
    height, width = _resolution_to_hw(args.resolution)

    # Base text-to-video clip (used directly for t2v, or as segment 0 for long).
    clip = pipe.generate_t2v(
        prompt=args.prompt,
        negative_prompt=negative,
        height=height,
        width=width,
        num_frames=NUM_FRAMES,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )[0]

    if args.mode == "t2v":
        _save(clip, args.output, args.fps, is_writer)
        return

    # --- long mode: autoregressively continue the last clip and stitch ---
    base, ext = os.path.splitext(args.output)
    segments = [clip]
    _save(clip, f"{base}_seg00{ext}", args.fps, is_writer)  # keep raw segments as a safety net

    cur = clip
    for i in range(1, args.segments + 1):
        cur = pipe.generate_vc(
            video=cur,
            prompt=args.prompt,
            negative_prompt=negative,
            resolution=args.resolution,
            num_frames=NUM_FRAMES,
            num_cond_frames=NUM_COND_FRAMES,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=True,
        )[0]
        _save(cur, f"{base}_seg{i:02d}{ext}", args.fps, is_writer)
        # Drop the conditioning overlap so the stitched video doesn't repeat frames.
        segments.append(cur[NUM_COND_FRAMES:])

    final = torch.cat(segments, dim=0)
    _save(final, args.output, args.fps, is_writer)


def _parse_args():
    p = argparse.ArgumentParser(description="Parameterized LongCat-Video generator")
    p.add_argument("--mode", choices=["t2v", "long"], default="t2v",
                   help="t2v = single clip; long = base clip + continuations stitched")
    p.add_argument("--prompt", required=True, help="your positive prompt")
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    p.add_argument("--segments", type=int, default=5,
                   help="(long mode) number of continuation segments after the base clip")
    p.add_argument("--resolution", choices=["480p", "720p"], default="480p")
    p.add_argument("--output", default="output.mp4")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--distill", action="store_true",
                   help="faster: 16 steps, guidance 1.0, no negative prompt")
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir",
                   default="./weights/LongCat-Video")
    p.add_argument("--context-parallel-size", dest="context_parallel_size",
                   type=int, default=1)
    p.add_argument("--enable-compile", dest="enable_compile", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
