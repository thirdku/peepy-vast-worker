"""Peepy Forge pyworker — makes SD Forge serverless-compatible on Vast.ai.

Fronts Forge's /sdapi/v1/txt2img on the peepy worker template
(vastai/sd-forge image, Forge on internal port 17860, supervisor piping its
output to /var/log/portal/forge.log). Deployed by setting PYWORKER_REPO to
this repo on the serverless template; Vast's startup clones it, installs
requirements.txt, and runs `python worker.py`.

Readiness: instead of matching Forge's stdout wording (fragile across
builds), a background shim probes /sdapi/v1/sd-models until Forge answers
with a non-empty model list, then appends FORGE_READY to the tailed log —
the same pattern the official comfyui-json worker uses. on_error matches
ONLY our own FORGE_START_FAILED token: Forge prints tracebacks for bad
per-request input too, and an on_error match permanently kills the worker.

The benchmark payload mirrors a real peepy selfie render (checkpoint +
LoRAs + TI embeddings in the negative + ADetailer face pass), so it doubles
as an end-to-end canary: a worker whose models didn't provision correctly
fails its benchmark and never receives traffic.
"""

import base64
import os
import random
import threading
import time

import aiohttp
import requests
from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

FORGE_PORT = int(os.environ.get("FORGE_INTERNAL_PORT", "17860"))
FORGE_URL = f"http://127.0.0.1:{FORGE_PORT}"
LOG_FILE = os.environ.get("MODEL_LOG_FILE", "/var/log/portal/forge.log")
BENCHMARK_CHECKPOINT = os.environ.get("FORGE_MODEL", "homosimileXLPony_v40NAIXLEPS")
# First boot loads a ~7GB checkpoint from disk after provisioning; be patient.
STARTUP_TIMEOUT_S = int(os.environ.get("FORGE_STARTUP_TIMEOUT", "1800"))

READY_TOKEN = "FORGE_READY"
FAIL_TOKEN = "FORGE_START_FAILED"


# ── readiness shim ───────────────────────────────────────────────────────────

def _append_log(line: str) -> None:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _probe_render() -> bool:
    """Prove the API can serve PRODUCTION payloads, not just list models.

    Forge Neo answers /sdapi before its extension script-arg tables settle —
    any request carrying alwayson_scripts in that window 500s with
    "list assignment index out of range" (seen live, Jul 19 2026). A tiny real
    render with a disabled ADetailer block exercises exactly that table, and
    doubles as the checkpoint pre-warm before the benchmark runs.
    """
    payload = {
        "prompt": "1boy", "negative_prompt": "girl",
        "steps": 1, "width": 256, "height": 320, "cfg_scale": 5,
        "sampler_name": "Euler a", "seed": 1,
        "send_images": False, "save_images": False,
        "override_settings": {"sd_model_checkpoint": BENCHMARK_CHECKPOINT},
        "override_settings_restore_afterwards": False,
        # Disabled block: parses through the arg table without running detection.
        "alwayson_scripts": {"ADetailer": {"args": [False, False, {"ad_model": "None"}]}},
    }
    try:
        r = requests.post(f"{FORGE_URL}/sdapi/v1/txt2img", json=payload, timeout=300)
        return r.ok
    except Exception:
        return False


def _readiness_shim() -> None:
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        try:
            r = requests.get(f"{FORGE_URL}/sdapi/v1/sd-models", timeout=5)
            if r.ok and isinstance(r.json(), list) and len(r.json()) > 0 and _probe_render():
                _append_log(READY_TOKEN)
                return
        except Exception:
            pass
        time.sleep(5)
    _append_log(FAIL_TOKEN)


threading.Thread(target=_readiness_shim, daemon=True).start()


# ── benchmark: a real peepy-style render (fixed recipe, random seed) ─────────
# Prompt pieces mirror api/_lib/chat-core.ts buildForgePayload so the
# benchmark exercises exactly what production requests will: both LoRA
# loads, the TI negatives, clip-skip, checkpoint override, and ADetailer.

QUALITY_HEAD = ("masterpiece, best quality, amazing quality, very aesthetic, "
                "high resolution, ultra-detailed, absurdres, newest")
ARTIST_TAGS = "(( pnk_crow )), (( domo_(domo_kizusuki) ))"
LORA_BLOCK = ("Expressiveh <lora:Expressive_H:1> "
              "<lora:noobaiXLNAIXL_epsilonPred11Version-lora:0.15>")
MALE_STYLING = ("natural light, light on body, male focus, male only, "
                "(blush:0.2), smooth skin, large pectorals")
APPEARANCE = "1boy, solo, adult male, short black hair, athletic build"
SELFIE_BEAT = ("dynamic angle, looking at viewer, ( blush :0.4), "
               "light details, light on face, natural light")
OUTFIT = "casual t-shirt <lora:Undressing_Male:0.3>"

NEGATIVE = ("modern, recent, oldest, cartoon, graphic, text, painting, crayon, "
            "graphite, abstract, glitch, deformed, mutated, ugly, disfigured, "
            "long body, lowres, bad anatomy, missing fingers, extra digit, "
            "fewer digits, cropped, very displeasing, (worst quality, bad "
            "quality:1.2), sketch, jpeg artifacts, signature, watermark, "
            "username, chibi, simple background, conjoined, ai-generated, "
            "female, girl, feminine, AissistXLv2-neg , unaestheticXL_bp5")

AD_FACE_NEGATIVE = ("cartoon, graphic, text, deformed, ugly, disfigured, lowres, "
                    "bad anatomy, cropped, (worst quality, bad quality:1.2), "
                    "jpeg artifacts, watermark, female, girl, feminine")


def make_benchmark_payload() -> dict:
    prompt = ", ".join([QUALITY_HEAD, ARTIST_TAGS, LORA_BLOCK, MALE_STYLING,
                        APPEARANCE, SELFIE_BEAT, OUTFIT])
    return {
        "prompt": prompt,
        "negative_prompt": NEGATIVE,
        "steps": 35,
        "cfg_scale": 5,
        "width": 832,
        "height": 1216,
        "sampler_name": "Euler a",
        "seed": random.randint(0, 2**31 - 1),
        "send_images": True,
        "save_images": False,
        "override_settings": {
            "CLIP_stop_at_last_layers": 2,
            "sd_model_checkpoint": BENCHMARK_CHECKPOINT,
            "enable_pnginfo": False,
        },
        "override_settings_restore_afterwards": False,
        "alwayson_scripts": {
            "ADetailer": {
                "args": [
                    True,   # ad_enable
                    False,  # skip_img2img
                    {
                        "ad_model": "face_yolov8n.pt",
                        "ad_prompt": ", ".join([QUALITY_HEAD, APPEARANCE]),
                        "ad_negative_prompt": AD_FACE_NEGATIVE,
                        "ad_confidence": 0.6,
                        "ad_dilate_erode": 4,
                        "ad_mask_blur": 4,
                        "ad_denoising_strength": 0.3,
                        "ad_inpaint_only_masked": True,
                        "ad_inpaint_only_masked_padding": 32,
                    },
                ],
            },
        },
    }


def workload_calculator(payload: dict) -> float:
    """Request cost in comparable units: steps x megapixels, with multipliers
    for the hires pass and the ADetailer face pass. A default peepy selfie
    (35 steps, 832x1216, ADetailer) ~= 49.6 units."""
    steps = float(payload.get("steps", 35))
    mp = float(payload.get("width", 832)) * float(payload.get("height", 1216)) / 1e6
    cost = steps * mp
    if payload.get("enable_hr"):
        cost *= 2.5
    scripts = payload.get("alwayson_scripts") or {}
    if "ADetailer" in scripts:
        cost *= 1.4
    return cost


# ── live progress passthrough ────────────────────────────────────────────────
# Forge's /sdapi/v1/progress is GET-only, so a plain forwarding handler can't
# front it (the pyworker forwards POST). A remote-dispatch handler runs this
# coroutine instead of forwarding — it answers WHILE a render is in flight
# because allow_parallel_requests=True bypasses the txt2img FIFO queue, and
# the caller reuses the render's own auth_data (signatures aren't single-use).

async def forge_progress() -> dict:
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(f"{FORGE_URL}/sdapi/v1/progress",
                             params={"skip_current_image": "true"}) as r:
                if r.status == 200:
                    return await r.json()
    except Exception:
        pass
    return {}


# ── worker config ────────────────────────────────────────────────────────────

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=FORGE_PORT,
    model_log_file=LOG_FILE,
    handlers=[
        HandlerConfig(
            route="/sdapi/v1/txt2img",
            allow_parallel_requests=False,   # one render at a time per GPU
            max_queue_time=120.0,
            workload_calculator=workload_calculator,
            benchmark_config=BenchmarkConfig(
                generator=make_benchmark_payload,
                concurrency=1,
                runs=2,
            ),
        ),
        HandlerConfig(
            route="/sdapi/v1/progress",
            allow_parallel_requests=True,   # must answer during a render
            max_queue_time=None,            # cosmetic polls never 429
            workload_calculator=lambda payload: 0.0,  # zero autoscaler load
            remote_function=forge_progress,
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=[READY_TOKEN],
        # ONLY our own fatal token — never generic "Traceback": Forge logs
        # tracebacks for recoverable per-request errors, and on_error is fatal.
        on_error=[FAIL_TOKEN],
        on_info=[],
    ),
)

if __name__ == "__main__":
    Worker(worker_config).run()
