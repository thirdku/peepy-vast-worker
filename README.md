# peepy-vast-worker

Vast.ai serverless PyWorker for peepy-chat.ai's SD Forge image workers.

> **Branch `video`** (this branch) is the image-to-video worker for the
> `peepy-anima` endpoint: ComfyUI only, Wan 2.2 I2V A14B, provisioned by
> `vast-provisioning-video.sh`, template env `PYWORKER_REF=video`. The image
> fleet runs branch `neo` and never loads this code.

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
