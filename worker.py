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
from http.server import BaseHTTPRequestHandler, HTTPServer

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

# ── health (Sep 2 2026) ───────────────────────────────────────────────────────
# The framework's readiness used to be "an on_load token appeared in the log" + a cached
# benchmark. The log was never rotated, so every RESTART re-read a stale FORGE_READY from
# a previous boot and the worker advertised itself ready ~8s after boot while Forge was
# still 15-30s from serving (empty 500s / 404s / hang-ups for every render routed in that
# window — and a worker whose Forge could not start at all, e.g. a host with a broken GPU
# driver, sat "ready" forever and black-holed the queue). Two fixes:
#   • the template now sets ROTATE_MODEL_LOG=true (start_server.sh truncates the log per
#     start) so on_load can only fire from THIS boot's probe render;
#   • model_healthcheck_url below: the framework will not mark the model loaded until this
#     endpoint answers 200, and marks the worker ERRORED if it later stops answering — so
#     a dead Forge takes the worker out of routing instead of eating renders.
# /health = 200 only when (a) this boot's probe render has passed and (b) Forge's API has
# answered within the last LIVENESS_GRACE_S. The grace window rides out supervisor
# restarting Forge after a crash (~15s) without convicting a worker that will recover.
# Rotate the model log OURSELVES at import time (Sep 2 2026). Vast's start_server.sh only
# rotates when the template sets ROTATE_MODEL_LOG=true + MODEL_LOG, and the account API key
# can't edit templates — so do the equivalent here, BEFORE the framework opens the file: the
# tailer reads from the START of the file, so any FORGE_READY left by a previous boot would be
# consumed as "loaded" ~8s after boot. This runs at module import, ahead of the readiness thread
# and ahead of Worker().run(), so the framework can only ever see THIS boot's token.
def _rotate_model_log() -> None:
    try:
        if not os.path.isfile(LOG_FILE):
            return
        old = LOG_FILE + ".old"
        with open(LOG_FILE, "rb") as src, open(old, "ab") as dst:
            dst.write(src.read())
        with open(LOG_FILE, "r+b") as f:
            f.truncate(0)
        # keep .old bounded (~20 MB) — supervisor's own 10 MB×1 rotation only covers the live file
        if os.path.getsize(old) > 20 * 1024 * 1024:
            with open(old, "rb") as f:
                f.seek(-10 * 1024 * 1024, os.SEEK_END)
                tail = f.read()
            with open(old, "wb") as f:
                f.write(tail)
    except Exception:
        pass  # a failed rotation just means the old behaviour (healthcheck still gates readiness)


_rotate_model_log()

HEALTH_PORT = int(os.environ.get("FORGE_HEALTH_PORT", "17870"))
LIVENESS_GRACE_S = int(os.environ.get("FORGE_LIVENESS_GRACE_S", "90"))
_probe_passed = False
_last_forge_ok = 0.0


def _liveness_loop() -> None:
    global _last_forge_ok
    while True:
        try:
            r = requests.get(f"{FORGE_URL}/sdapi/v1/progress", params={"skip_current_image": "true"}, timeout=5)
            if r.ok:
                _last_forge_ok = time.time()
        except Exception:
            pass
        time.sleep(5)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server API
        alive = _probe_passed and (time.time() - _last_forge_ok) < LIVENESS_GRACE_S
        body = b"ok" if alive else b"not ready"
        self.send_response(200 if alive else 503)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:  # keep the pyworker log clean (polled every 10s)
        return


def _serve_health() -> None:
    HTTPServer(("127.0.0.1", HEALTH_PORT), _HealthHandler).serve_forever()


threading.Thread(target=_liveness_loop, daemon=True).start()
threading.Thread(target=_serve_health, daemon=True).start()


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
    global _probe_passed
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        try:
            r = requests.get(f"{FORGE_URL}/sdapi/v1/sd-models", timeout=5)
            if r.ok and isinstance(r.json(), list) and len(r.json()) > 0 and _probe_render():
                _probe_passed = True  # /health may now answer 200 (liveness permitting)
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
    # Readiness + liveness gate (see the health block above): the framework polls this every
    # 10s, needs one 200 before marking the model loaded, and errors the worker on a later
    # non-200 — a Forge that dies is taken out of routing instead of 500ing every render.
    model_healthcheck_url=f"http://127.0.0.1:{HEALTH_PORT}/health",
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
