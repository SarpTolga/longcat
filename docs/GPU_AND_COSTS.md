# GPU choice & cost notes

> Prices are rough RunPod community-cloud ballparks as of mid-2026 and **change
> constantly** — always check the live RunPod pricing page before you commit.

## Which GPU?

LongCat-Video is **13.6B params, dense**. Practical VRAM picture:

| GPU | VRAM | Verdict for LongCat-Video |
|-----|------|---------------------------|
| **H100** | 80GB | ✅ Best. Full 720p, all features, fastest. Top pick. |
| **A100** | 80GB | ✅ Great value. Slightly slower than H100, much cheaper. **Recommended default.** |
| A100 / A6000 | 48GB | ✅ Works for most tasks; tighter on long-video / avatar. |
| RTX 4090 | 24GB | ⚠️ Bare minimum. Needs `--use_int8` + distill; slow, may OOM on long video. |
| anything < 24GB | — | ❌ Don't bother. |

**Recommendation:** start on a single **A100 80GB**. Move to **H100** if you want
speed, or to **2×** GPUs (`--context_parallel_size=2`) only if you need faster
long-video / higher throughput.

## Cost shape (the important mental model)

Two separate meters:

1. **GPU (per-hour, only while the pod runs)** — stop the pod and this stops.
   - A100 80GB: ~$1.5–2.5/hr · H100 80GB: ~$2.5–4/hr (community vs secure cloud).
2. **Network volume (per-GB-month, billed even when no pod is running)** —
   this is the price of "never reinstall." ~$0.05–0.07/GB/month.
   - You need ~**100GB** (60GB weights + conda env + repo + scratch) →
     roughly **$5–8/month** to keep everything warm between sessions.

So the workflow cost = (hours you actually generate × GPU rate) + ~$6/mo standing
storage. The standing storage is what buys you 2-minute respins instead of
40-minute reinstalls. Worth it.

## Money-saving habits

- **Stop the pod** the moment you're done generating. GPU billing stops; the
  volume keeps your setup.
- Keep the volume; kill the pod. Recreate a pod against the same volume anytime.
- If you go weeks without using it and want to stop even the ~$6/mo: download the
  weights elsewhere / back up, delete the volume, re-run setup next time.
- Consider **spot/interruptible** pods for big batch jobs (cheaper, can be killed).
- The 720p clips render "in minutes" — a typical session is short; you're mostly
  paying for a handful of GPU-hours, not a 24/7 server.
