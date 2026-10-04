"""Peepy VIDEO pyworker — ComfyUI-only image-to-video worker for Vast.ai serverless.

Branch `video` of this repo (Oct 2026). The peepy-anima endpoint's template sets
PYWORKER_REF=video and provisions with vast-provisioning-video.sh from this branch;
the image fleet runs branch `neo` and never loads this file. The model is MiniMax H3
fl2va (int8 transformer, nvfp4 Qwen3-VL text encoder, 4-step turbo LoRA); it replaced
Wan 2.2 I2V A14B in Oct 2026.

How it differs from neo's worker.py:
  • No Forge. ComfyUI is the model server, so readiness, liveness and the benchmark
    all go through ComfyUI (a neo worker's health gates on a Forge probe render,
    which a ComfyUI-only box can never pass).
  • /comfy/render returns every saved output with its type, so a SaveVideo mp4
    comes back next to (or instead of) still images.
  • A render that times out, or whose caller disconnects, is interrupted inside
    ComfyUI. Otherwise the GPU keeps rendering an orphaned clip while the pyworker
    reports itself idle, and the next clip queues behind it inside ComfyUI.
  • A CUDA fault seen in any render latches /health to 503 until the process
    restarts: a poisoned GPU context fails every later render, but ComfyUI's HTTP
    API keeps answering, so liveness alone never notices.
"""

import asyncio
import base64
import os
import random
import re
import shutil
import struct
import threading
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer

import aiohttp
import requests
from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

COMFY_PORT = int(os.environ.get("COMFY_INTERNAL_PORT", "8188"))
COMFY_URL = f"http://127.0.0.1:{COMFY_PORT}"
COMFY_DIR = os.environ.get("COMFY_DIR", "/workspace/ComfyUI")
LOG_FILE = os.environ.get("MODEL_LOG_FILE", "/workspace/comfy.log")
# A fresh worker stages ~38 GB of H3 weights on its first render; be patient.
STARTUP_TIMEOUT_S = int(os.environ.get("VIDEO_STARTUP_TIMEOUT", "2400"))

READY_TOKEN = "PEEPY_VIDEO_READY"
FAIL_TOKEN = "PEEPY_VIDEO_START_FAILED"

# The files vast-provisioning-video.sh installs under ComfyUI/models/: MiniMax H3 from
# Comfy-Org/MiniMax-H3 (revision e5eb578a), plus the yaoi LoRA from Civitai, which the
# app adds at 0.8 to two-man clips.
H3_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
H3_TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
H3_TURBO = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
H3_YAOI = "yaoi_h3_lora_000002500.safetensors"
H3_NODE = "MiniMaxH3ImageToVideo"
REQUIRED_MODELS = {
    "UNETLoader": ("unet_name", [H3_UNET]),
    "CLIPLoader": ("clip_name", [H3_TE]),
    "VAELoader": ("vae_name", [H3_VAE]),
    "LoraLoaderModelOnly": ("lora_name", [H3_TURBO, H3_YAOI]),
}


# Rotate the model log at import, before the framework opens it: the tailer reads from
# the start of the file, so a READY_TOKEN left by a previous boot would mark this boot
# loaded before ComfyUI is up (the same fix neo's worker.py carries).
def _rotate_model_log() -> None:
    try:
        if not os.path.isfile(LOG_FILE):
            return
        old = LOG_FILE + ".old"
        with open(LOG_FILE, "rb") as src, open(old, "ab") as dst:
            dst.write(src.read())
        with open(LOG_FILE, "r+b") as f:
            f.truncate(0)
        if os.path.getsize(old) > 20 * 1024 * 1024:
            with open(old, "rb") as f:
                f.seek(-10 * 1024 * 1024, os.SEEK_END)
                tail = f.read()
            with open(old, "wb") as f:
                f.write(tail)
    except Exception:
        pass


_rotate_model_log()


# The framework skips the benchmark when it finds .has_benchmark in its working dir,
# and that file survives a restart. Remove it so every boot renders a proof clip: a
# worker whose GPU or weights went bad since the last boot must fail before it serves.
def _forget_benchmark() -> None:
    for d in {os.getcwd(), os.path.dirname(os.path.abspath(__file__))}:
        try:
            os.remove(os.path.join(d, ".has_benchmark"))
        except FileNotFoundError:
            pass
        except Exception:
            pass


_forget_benchmark()


# ── health ───────────────────────────────────────────────────────────────────
# /health = 200 only when this boot's readiness check passed, ComfyUI answered within
# the last LIVENESS_GRACE_S (rides out a supervisor restart), and no render has hit a
# CUDA fault. The framework errors the worker out of routing on a non-200.

HEALTH_PORT = int(os.environ.get("VIDEO_HEALTH_PORT", "17870"))
LIVENESS_GRACE_S = int(os.environ.get("VIDEO_LIVENESS_GRACE_S", "90"))
_ready = False
_last_comfy_ok = 0.0
_gpu_fault = ""

# Errors that leave the CUDA context unusable for the rest of the process. "Fault
# failed" is how the same fault surfaced through ComfyUI on Sep 26 2026. Out-of-memory
# is recoverable and must never latch.
CUDA_FATAL = re.compile(
    r"illegal memory access|cudaErrorIllegalAddress|cudaErrorUnknown|CUDA error: unknown error"
    r"|unspecified launch failure|cudaErrorLaunchFailure|misaligned address|device-side assert"
    r"|uncorrectable ECC|no CUDA-capable device|No CUDA GPUs are available|fallen off the bus"
    r"|CUBLAS_STATUS_EXECUTION_FAILED|CUDNN_STATUS_EXECUTION_FAILED|AcceleratorError|Fault failed",
    re.I,
)
NOT_FATAL = re.compile(r"out of memory|OutOfMemoryError|ALLOC_FAILED", re.I)


def _note_render_error(detail) -> None:
    global _gpu_fault
    # Only the error fields: ComfyUI's execution_error also carries the node's inputs,
    # i.e. the user's prompt text, which must never be able to trip (or mask) the latch.
    if isinstance(detail, dict):
        text = " ".join(str(detail.get(k, "")) for k in ("exception_type", "exception_message", "traceback"))
    else:
        text = str(detail)
    if CUDA_FATAL.search(text) and not NOT_FATAL.search(text):
        if not _gpu_fault:
            _gpu_fault = text[:300]
            _append_log(f"PEEPY_VIDEO_GPU_FAULT {_gpu_fault}")


def _liveness_loop() -> None:
    global _last_comfy_ok
    while True:
        try:
            if requests.get(f"{COMFY_URL}/system_stats", timeout=5).ok:
                _last_comfy_ok = time.time()
        except Exception:
            pass
        time.sleep(5)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server API
        alive = _ready and not _gpu_fault and (time.time() - _last_comfy_ok) < LIVENESS_GRACE_S
        body = b"ok" if alive else (b"gpu fault" if _gpu_fault else b"not ready")
        self.send_response(200 if alive else 503)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return


def _serve_health() -> None:
    HTTPServer(("127.0.0.1", HEALTH_PORT), _HealthHandler).serve_forever()


threading.Thread(target=_liveness_loop, daemon=True).start()
threading.Thread(target=_serve_health, daemon=True).start()


# ── readiness shim ───────────────────────────────────────────────────────────
# ComfyUI up + every H3 file visible to its loaders + the H3 node registered. The real
# proof — a rendered clip — is the framework's benchmark, which runs right after
# READY_TOKEN.

def _append_log(line: str) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _combo_options(spec) -> list:
    """The options of a combo input, in either shape this ComfyUI pin serves: the loaders'
    legacy [["a", "b"], {...}] or the newer ["COMBO", {"options": ["a", "b"]}]."""
    if isinstance(spec, list) and spec:
        if isinstance(spec[0], list):
            return spec[0]
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            return spec[1].get("options") or []
    return []


def _models_visible() -> bool:
    for node, (field, names) in REQUIRED_MODELS.items():
        r = requests.get(f"{COMFY_URL}/object_info/{node}", timeout=15)
        if not r.ok:
            return False
        spec = (((r.json().get(node) or {}).get("input") or {}).get("required") or {}).get(field)
        options = _combo_options(spec)
        if not all(n in options for n in names):
            return False
    # ComfyUI answers an unknown node with 200 {}, so check the key itself.
    r = requests.get(f"{COMFY_URL}/object_info/{H3_NODE}", timeout=15)
    return r.ok and H3_NODE in r.json()


def _readiness_shim() -> None:
    global _ready
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        # The base image runs the provisioning script with on_failure=continue: a failed
        # script leaves this marker and starts the pyworker anyway. Never go ready then.
        if os.path.exists("/.provisioning_failed"):
            _append_log("provisioning failed: /.provisioning_failed exists")
            break
        try:
            if requests.get(f"{COMFY_URL}/system_stats", timeout=5).ok and _models_visible():
                _ready = True
                _append_log(READY_TOKEN)
                return
        except Exception:
            pass
        time.sleep(5)
    _append_log(FAIL_TOKEN)


threading.Thread(target=_readiness_shim, daemon=True).start()


# ── the clip graph (benchmark + reference for the app's graph builder) ───────
# MiniMax H3 fl2va, following Comfy-Org's video_minimax_h3_i2v template at this pin:
# the turbo LoRA's 4 steps of res_multistep through SamplerCustomAdvanced, no audio.
# For a loop the still is pinned as both first and last frame and the last frame is
# dropped, because it duplicates frame 0 at the seam; otherwise the ending is free.
# `yaoi` > 0 chains the yaoi LoRA after the turbo LoRA at that strength.

def make_clip_graph(image_name: str, prompt: str, width: int = 512, height: int = 736,
                    frames: int = 73, fps: float = 24.0, steps: int = 4, seed: int = 0,
                    loop: bool = True, yaoi: float = 0.0) -> dict:
    model = ["3", 0]
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_UNET, "weight_dtype": "default"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": H3_TURBO,
                                                              "strength_model": 1.0}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_TE, "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE}},
        "6": {"class_type": H3_NODE, "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "prompt": prompt, "width": width, "height": height,
            "length": frames, "first_frame": ["1", 0], **({"last_frame": ["1", 0]} if loop else {})}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["5", 0]}},
        "13": {"class_type": "ImageFromBatch", "inputs": {"image": ["12", 0], "batch_index": 0,
                                                          "length": frames - 1 if loop else frames}},
        "14": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "fps": fps}},
        "15": {"class_type": "SaveVideo", "inputs": {
            "video": ["14", 0], "filename_prefix": "peepy/clip", "format": "mp4",
            "format.codec": "h264", "format.codec.encoding": "re-encode",
            "format.codec.encoding.crf": 20.0}},
    }
    if yaoi > 0:
        graph["16"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": H3_YAOI,
                                                                       "strength_model": yaoi}}
        model = ["16", 0]
    graph["9"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple",
                                                             "steps": steps, "denoise": 1.0}}
    graph["10"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["6", 0]}}
    graph["11"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["7", 0], "guider": ["10", 0], "sampler": ["8", 0], "sigmas": ["9", 0], "latent_image": ["6", 1]}}
    return graph


def _png(width: int, height: int) -> bytes:
    """A plain gradient PNG built in pure Python, so the benchmark needs no image library
    and the public repo ships no picture."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        g = int(255 * y / max(1, height - 1))
        for x in range(width):
            rows += bytes((int(255 * x / max(1, width - 1)), g, 128))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
            + chunk(b"IEND", b""))


BENCH_IMAGE = "peepy_bench.png"
BENCH_W, BENCH_H, BENCH_FRAMES = 512, 736, 73   # ~32 s warm on a 3090
_BENCH_PNG_B64 = base64.b64encode(_png(BENCH_W, BENCH_H)).decode()
# H3's own prompt layout (the one the app sends), so the text encoder does real work.
BENCH_PROMPT = (
    "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n"
    "integrated_multimodal_description: [Shot 1] A static shot of the soft colour gradient shown in <Picture 1>, "
    "preserving its colours and composition. The camera is completely still for the whole clip: no push-in, no pan, "
    "no zoom. Throughout the clip the colours shimmer gently and a soft glow drifts slowly across the frame. "
    "No people, no new objects, no text, no cuts.\n"
    "overall_soundscape: N/A\n"
    "non_diegetic_music: N/A"
)


def make_benchmark_payload() -> dict:
    return {
        "prompt": make_clip_graph(BENCH_IMAGE, BENCH_PROMPT, width=BENCH_W, height=BENCH_H,
                                  frames=BENCH_FRAMES, seed=random.randint(0, 2**31 - 1)),
        "timeout_s": 900,
        "images": {BENCH_IMAGE: _BENCH_PNG_B64},
        # The framework counts any returned value as a passing benchmark run; an
        # exception is what fails it, so a worker that can't render never goes live.
        "strict": True,
    }


def h3_est_seconds(width: float, height: float, frames: float, steps: float = 4.0) -> float:
    """GPU seconds for one H3 clip on an RTX 3090 with the 4-step turbo LoRA: a fit to
    every measured render (SD 512x672 5 s = 45 s ... Full 1024x1376 5 s = 318-352 s), within
    ~10%. The app's clipEstSeconds is the same formula and sends it as the /route/ cost."""
    fmp = float(frames) * float(width) * float(height) / 1e6
    return 6.0 + float(steps) * fmp * (0.18 + 0.0016 * fmp)


# Anything the estimate can't read (no video node, a linked width...) is priced as the
# benchmark clip.
DEFAULT_WORKLOAD = h3_est_seconds(BENCH_W, BENCH_H, BENCH_FRAMES)
# Wan 2.2 graphs (the pre-H3 app): frames x megapixels x steps at the 1.50 units/s the Wan
# benchmark measured on a 3090. Only a box provisioned before H3 still holds the Wan
# weights; a fresh H3 worker refuses such a graph at validation in under a second.
WAN_UNITS_PER_S = 1.5


def clip_workload(payload: dict) -> float:
    """Estimated GPU seconds, read from the graph (the H3 node's width/height/length and
    BasicScheduler's steps). The benchmark measures throughput in these same units
    (~1 per second), so wait_time = queued workload / throughput comes out in seconds and
    max_queue_time means what it says."""
    try:
        graph = payload.get("prompt") or {}
        h3 = wan = None
        steps, wan_steps = 4.0, 4.0
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            ct = node.get("class_type")
            inp = node.get("inputs") or {}
            if ct == H3_NODE:
                h3 = (float(inp.get("width", 1344)), float(inp.get("height", 768)), float(inp.get("length", 124)))
            elif ct == "BasicScheduler":
                steps = float(inp.get("steps", steps))
            elif ct in ("WanFirstLastFrameToVideo", "WanImageToVideo"):
                wan = (float(inp.get("width", 480)), float(inp.get("height", 704)), float(inp.get("length", 49)))
            elif ct in ("KSamplerAdvanced", "KSampler"):
                wan_steps = float(inp.get("steps", wan_steps))
        if h3:
            return h3_est_seconds(h3[0], h3[1], h3[2], steps)
        if wan:
            return wan[2] * (wan[0] * wan[1] / 1e6) * wan_steps / WAN_UNITS_PER_S
        return DEFAULT_WORKLOAD
    except Exception:
        return DEFAULT_WORKLOAD


# ── ComfyUI render (remote-dispatch) ─────────────────────────────────────────
# {payload: {prompt: <API graph>, timeout_s, images: {filename: base64}}} →
# {result: {files: [{filename, mime, animated, b64}], images: [b64 of stills]}} or
# {result: {error, detail}}.

MIME = {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska", ".webp": "image/webp",
        ".gif": "image/gif", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


async def _comfy_post(path: str, body: dict, timeout: float = 15) -> None:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
            async with s.post(f"{COMFY_URL}{path}", json=body) as r:
                await r.read()
    except Exception:
        pass


async def _cancel(pid: str | None) -> None:
    """Stop a prompt whether it is running or still queued."""
    if pid:
        await _comfy_post("/queue", {"delete": [pid]})
    await _comfy_post("/interrupt", {"prompt_id": pid} if pid else {})


async def _get_json(path: str, timeout: float = 10):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
        async with s.get(f"{COMFY_URL}{path}") as r:
            return await r.json(content_type=None)


async def _drain_orphans(deadline: float) -> None:
    """The pyworker runs one request at a time, so anything already in ComfyUI's queue
    when a request starts is left over from a caller that went away. Clear it first."""
    try:
        q = await _get_json("/queue")
    except Exception:
        return
    if not (q.get("queue_running") or q.get("queue_pending")):
        return
    await _comfy_post("/queue", {"clear": True})
    await _comfy_post("/interrupt", {})
    stop = min(deadline, time.time() + 30)
    while time.time() < stop:
        await asyncio.sleep(1)
        try:
            if not (await _get_json("/queue", 5)).get("queue_running"):
                return
        except Exception:
            pass


# LoRAs the app's graph may name that a fresh worker can lack (Oct 4 2026): the two Civitai files
# behind a sign-in (VIDEO_EXTRAS in vast-provisioning-video.sh — fetched only from the R2 mirror or
# with CIVITAI_TOKEN) and the newer turbo. A graph naming one that isn't on disk renders WITHOUT it
# (its consumers re-wired to its input model) instead of failing ComfyUI's validation. The required
# models (REQUIRED_MODELS) are never dropped — a worker without them never reports ready.
OPTIONAL_LORAS = {
    "NSFW_ANIME_V7_H3-step00019500.safetensors",
    "H3_Motion_Booster_anime.safetensors",
    "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors",
}


def _drop_missing_loras(prompt: dict) -> dict:
    lora_dir = os.path.join(COMFY_DIR, "models", "loras")
    drop = {}
    for nid, node in prompt.items():
        if not (isinstance(node, dict) and node.get("class_type") == "LoraLoaderModelOnly"):
            continue
        inputs = node.get("inputs") or {}
        name = inputs.get("lora_name")
        if name in OPTIONAL_LORAS and not os.path.isfile(os.path.join(lora_dir, name)):
            drop[str(nid)] = inputs.get("model")
    if not drop:
        return prompt

    def resolve(ref):
        for _ in range(len(drop) + 1):  # a chain of dropped loaders resolves to the first kept input
            if isinstance(ref, list) and len(ref) == 2 and str(ref[0]) in drop and ref[1] == 0:
                ref = drop[str(ref[0])]
            else:
                break
        return ref

    out = {}
    for nid, node in prompt.items():
        if str(nid) in drop:
            continue
        if isinstance(node, dict):
            node = {**node, "inputs": {k: resolve(v) for k, v in (node.get("inputs") or {}).items()}}
        out[nid] = node
    names = sorted(str((prompt[n].get("inputs") or {}).get("lora_name")) for n in drop)
    print(f"[video] optional LoRA(s) not on this worker, rendering without: {', '.join(names)}", flush=True)
    return out


# Every save node writes under its own peepy/<request id>/ folder. A fresh prefix changes
# the node's inputs, so ComfyUI's cache can't answer a repeat graph with files this
# worker already deleted, and the whole folder goes once the bytes are in hand.
SAVE_NODES = ("SaveVideo", "SaveImage", "SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveWEBM")


def _scope_outputs(prompt: dict, tag: str) -> dict:
    scoped = {}
    for nid, node in prompt.items():
        if isinstance(node, dict) and node.get("class_type") in SAVE_NODES:
            inputs = dict(node.get("inputs") or {})
            base = os.path.basename(str(inputs.get("filename_prefix") or "").replace("\\", "/")) or "out"
            inputs["filename_prefix"] = f"peepy/{tag}/{base}"
            node = {**node, "inputs": inputs}
        scoped[nid] = node
    return scoped


async def comfy_render(prompt: dict = None, timeout_s: float = 600.0, images: dict = None,
                       strict: bool = False, **_extra) -> dict:
    result = await _comfy_render(prompt, timeout_s, images)
    if strict and "error" in result:
        raise RuntimeError(f"{result['error']}: {str(result.get('detail', ''))[:300]}")
    return result


async def _comfy_render(prompt: dict, timeout_s: float, images: dict) -> dict:
    if not isinstance(prompt, dict) or not prompt:
        return {"error": "missing prompt graph"}
    if _gpu_fault:
        return {"error": "gpu fault", "detail": _gpu_fault}
    try:
        timeout_s = float(timeout_s)
    except Exception:
        timeout_s = 600.0
    # One budget for the whole request: draining, rendering and fetching.
    deadline = time.time() + timeout_s
    staged = []
    if isinstance(images, dict):
        input_dir = os.path.join(COMFY_DIR, "input")
        try:
            for name, b64 in images.items():
                if not isinstance(name, str) or not isinstance(b64, str):
                    continue
                path = os.path.join(input_dir, os.path.basename(name))
                with open(path, "wb") as f:
                    f.write(base64.b64decode(b64))
                staged.append(path)
        except Exception as ex:
            for p in staged:
                try: os.remove(p)
                except Exception: pass
            return {"error": f"comfy image staging failed: {ex}"}
    tag = uuid.uuid4().hex
    holder: dict = {}
    try:
        await _drain_orphans(deadline)
        return await _render_inner(_scope_outputs(_drop_missing_loras(prompt), tag), timeout_s, deadline, holder)
    except asyncio.CancelledError:
        await asyncio.shield(_cancel(holder.get("pid")))
        raise
    finally:
        for p in staged:
            try: os.remove(p)
            except Exception: pass
        shutil.rmtree(os.path.join(COMFY_DIR, "output", "peepy", tag), ignore_errors=True)


def _in_queue(q: dict, pid: str) -> bool:
    for key in ("queue_running", "queue_pending"):
        for item in q.get(key) or []:
            if isinstance(item, list) and len(item) > 1 and item[1] == pid:
                return True
    return False


async def _render_inner(prompt: dict, timeout_s: float, deadline: float, holder: dict) -> dict:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
            async with s.post(f"{COMFY_URL}/prompt", json={"prompt": prompt}) as r:
                sub = await r.json(content_type=None)
                if r.status != 200:
                    return {"error": f"comfy submit {r.status}", "detail": sub}
    except Exception as ex:
        return {"error": f"comfy unreachable: {ex}"}
    pid = sub.get("prompt_id")
    if not pid:
        return {"error": "comfy returned no prompt_id", "detail": sub}
    holder["pid"] = pid
    entry = None
    missing = 0
    while time.time() < deadline:
        await asyncio.sleep(2)
        try:
            data = await _get_json(f"/history/{pid}", 15)
        except Exception:
            continue
        e = (data or {}).get(pid)
        if not e:
            # In neither the history nor the queue twice in a row: ComfyUI restarted
            # (or dropped the prompt) and nothing will ever finish it. Fail now rather
            # than at the deadline.
            try:
                missing = 0 if _in_queue(await _get_json("/queue", 10), pid) else missing + 1
            except Exception:
                continue
            if missing >= 2:
                return {"error": "comfy lost the prompt (restarted?)"}
            continue
        missing = 0
        status = e.get("status") or {}
        if status.get("status_str") == "error":
            msgs = [m for m in status.get("messages") or [] if m and m[0] == "execution_error"]
            detail = msgs[-1][1] if msgs else status
            _note_render_error(detail)
            return {"error": "comfy execution failed", "detail": detail}
        if status.get("completed"):
            entry = e
            break
    if entry is None:
        await _cancel(pid)
        return {"error": f"comfy timeout after {timeout_s}s"}
    files, stills = [], []
    try:
        # A finished clip is worth fetching even right at the deadline.
        fetch_s = max(15.0, deadline - time.time())
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=fetch_s)) as s:
            for out in (entry.get("outputs") or {}).values():
                animated = bool((out.get("animated") or [False])[0])
                for key in ("images", "gifs", "videos"):
                    for item in out.get(key) or []:
                        if item.get("type") != "output":
                            continue
                        name = item.get("filename", "")
                        q = {"filename": name, "subfolder": item.get("subfolder", ""), "type": "output"}
                        async with s.get(f"{COMFY_URL}/view", params=q) as r:
                            if r.status != 200:
                                continue
                            b64 = base64.b64encode(await r.read()).decode()
                        ext = os.path.splitext(name)[1].lower()
                        is_anim = animated or key != "images" or ext in (".mp4", ".webm", ".mkv", ".gif")
                        files.append({"filename": name, "mime": MIME.get(ext, "application/octet-stream"),
                                      "animated": is_anim, "b64": b64})
                        if not is_anim:
                            stills.append(b64)
    except Exception as ex:
        return {"error": f"comfy output fetch failed: {ex}"}
    if not files:
        return {"error": "comfy produced no output files"}
    return {"files": files, "images": stills}


# ── worker config ────────────────────────────────────────────────────────────

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=COMFY_PORT,
    model_log_file=LOG_FILE,
    model_healthcheck_url=f"http://127.0.0.1:{HEALTH_PORT}/health",
    handlers=[
        HandlerConfig(
            route="/comfy/render",
            allow_parallel_requests=False,   # one clip at a time per GPU
            # A clip runs minutes, so one queued clip already means a long wait; the
            # app caps clips in flight itself, this only stops a pile-up.
            max_queue_time=600.0,
            workload_calculator=clip_workload,
            remote_function=comfy_render,
            benchmark_config=BenchmarkConfig(
                generator=make_benchmark_payload,
                concurrency=1,
                runs=1,          # each run is a full clip (~32 s warm); the warmup also stages the ~38 GB of weights
            ),
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=[READY_TOKEN],
        on_error=[FAIL_TOKEN],
        on_info=[],
    ),
)

if __name__ == "__main__":
    Worker(worker_config).run()
