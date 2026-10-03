# peepy-vast-worker

Vast.ai serverless PyWorker for peepy-chat.ai's SD Forge image workers.

> **Branch `video`** (this branch) is the image-to-video worker for the
> `peepy-anima` endpoint: ComfyUI only, provisioned by
> `vast-provisioning-video.sh`, template env `PYWORKER_REF=video`. The image
> fleet runs branch `neo` and never loads this code.
>
> The model is **MiniMax H3** (fl2va), not Wan 2.2 any more (Oct 2026). The
> script fetches five files (~41.6 GB): from Comfy-Org/MiniMax-H3 at revision
> `e5eb578a`, `minimax_h3_fl2va_pruned_int8_convrot` (diffusion_models),
> `qwen3vl_32b_minimax_h3_nvfp4_awq` (text_encoders),
> `minimax_h3_video_vae_int8_convrot` (vae) and
> `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16` (loras); from Civitai,
> `yaoi_h3_lora_000002500` (loras, sha256-checked). Downloads resume after a
> drop. Changes to `worker.py` reach a running worker at its next restart (the
> pyworker pulls the branch at start); model changes reach only freshly
> provisioned workers.

- `worker.py` — fronts Forge's `/sdapi/v1/txt2img` (internal port 17860 on the
  `vastai/sd-forge` image). Readiness = probe `/sdapi/v1/sd-models` and append
  `FORGE_READY` to `/var/log/portal/forge.log`. Benchmark = a real peepy-style
  render (checkpoint + LoRAs + TI negatives + ADetailer), so a mis-provisioned
  worker fails its benchmark and never gets traffic.
- `client_test.py` — one-shot endpoint test, saves `test_out.png`.

Deployed via the serverless template's env:

```
PYWORKER_REPO=https://github.com/thirdku/peepy-vast-worker
PYWORKER_REF=main
```

This repo is public and contains **no secrets** — payloads (and the hidden
prompt recipe) arrive per-request from the app's API; the benchmark prompt is
a generic stand-in. Model provisioning lives in the app repo's
`scripts/vast-provisioning.sh` (served from R2), not here.
