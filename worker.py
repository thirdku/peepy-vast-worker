"""Peepy HYBRID pyworker — video clips AND photos on one GPU (Vast.ai serverless).

Branch `hybrid` of this repo (Oct 2026), provisioned by vast-provisioning-hybrid.sh. The box
runs both stacks: Forge Neo (SDXL photos) and ONE ComfyUI (video clips: MiniMax H3 + LTX 2.3;
Anima photos). Built from branch `video`'s worker.py (the ComfyUI side, unchanged) and branch
`neo`'s (the Forge side), with one addition — the GPU gate:

  • ONE job at a time on the GPU, across every route. A clip needs the whole 24 GB (an H3 Full
    clip peaked at 23.8 of 24.5 GB on a 3090), so a photo must never start beside one.
  • Clips first: when a clip and a photo both wait, the clip goes next. (A photo already
    running finishes first — 15–40 s.)
  • Before a clip, Forge unloads its checkpoint (Forge and ComfyUI are separate processes,
    and ComfyUI's memory manager can't see Forge's VRAM). After a clip, ComfyUI unloads the
    clip's models before a Forge photo. ComfyUI-to-ComfyUI switches (clip ↔ Anima photo) are
    left to ComfyUI's own memory manager.

Routes:
  /comfy/render      — a ComfyUI graph: a clip (any video node in it) or an Anima photo. Same
                       payload / reply as branch `video` ({files, images}), so the app's
                       renderClip and renderComfy both read it.
  /forge/txt2img     — a Forge txt2img body; replies Forge's own JSON under `result`. Forge is
                       reached only through here (never forwarded), so the gate covers it.
  /sdapi/v1/progress — Forge's live progress (parallel, zero load), as on `neo`.

Readiness: ComfyUI up with the H3 files visible (as `video`), then Forge's probe render tried
(as `neo`) and its checkpoint unloaded, so the benchmark clip starts on a clean GPU. A Forge
that never comes up leaves the worker serving clips; /forge/txt2img then answers
{"error": "forge not ready"} and the app keeps that photo on the photo pool.
"""

import asyncio
import base64
import json
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
FORGE_PORT = int(os.environ.get("FORGE_INTERNAL_PORT", "17860"))
FORGE_URL = f"http://127.0.0.1:{FORGE_PORT}"
FORGE_PROBE_CHECKPOINT = os.environ.get("FORGE_MODEL", "homosimileXLPony_v40NAIXLEPS")
LOG_FILE = os.environ.get("MODEL_LOG_FILE", "/workspace/comfy.log")
# A fresh worker stages ~90 GB of video weights plus the photo stack; be patient.
STARTUP_TIMEOUT_S = int(os.environ.get("VIDEO_STARTUP_TIMEOUT", "2400"))
FORGE_STARTUP_TIMEOUT_S = int(os.environ.get("FORGE_STARTUP_TIMEOUT", "1200"))
FORGE_RENDER_TIMEOUT_S = float(os.environ.get("FORGE_RENDER_TIMEOUT", "300"))

READY_TOKEN = "PEEPY_VIDEO_READY"
FAIL_TOKEN = "PEEPY_VIDEO_START_FAILED"

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


# Rotate the model log at import, before the framework opens it (see branch `video`).
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


def _forget_benchmark() -> None:
    for d in {os.getcwd(), os.path.dirname(os.path.abspath(__file__))}:
        try:
            os.remove(os.path.join(d, ".has_benchmark"))
        except Exception:
            pass


_forget_benchmark()


# ── health (as branch `video`: ComfyUI liveness + the CUDA-fault latch) ─────

HEALTH_PORT = int(os.environ.get("VIDEO_HEALTH_PORT", "17870"))
LIVENESS_GRACE_S = int(os.environ.get("VIDEO_LIVENESS_GRACE_S", "90"))
_ready = False
_last_comfy_ok = 0.0
_gpu_fault = ""
_forge_ready = False

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
    def do_GET(self) -> None:  # noqa: N802
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


def _append_log(line: str) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _log(msg: str) -> None:
    print(f"[hybrid] {msg}", flush=True)


# ── the GPU gate ─────────────────────────────────────────────────────────────
# What the GPU holds now: "video" (a clip's models in ComfyUI), "comfy" (an Anima photo's in
# ComfyUI), "forge" (Forge's checkpoint) or None (clean, after readiness).
_mode = None
_forge_loaded = False   # Forge holds a checkpoint in VRAM


# Photos first (owner, Oct 9 2026): a photo takes seconds, a clip minutes, so a clip waits
# behind the photos — but one that has waited CLIP_MAX_WAIT_S goes before any photo that
# hasn't started, so a steady stream of photos can't starve it. A photo that would wait
# longer than PHOTO_MAX_WAIT_S (a long clip rendering) is turned away at once: the app then
# sends it to the photo pool. Estimates are the workloads (3090 GPU-seconds) × HYBRID_SPEED.
CLIP_MAX_WAIT_S = float(os.environ.get("HYBRID_CLIP_MAX_WAIT_S", "120"))
PHOTO_MAX_WAIT_S = float(os.environ.get("HYBRID_PHOTO_MAX_WAIT_S", "45"))
HYBRID_SPEED = float(os.environ.get("HYBRID_SPEED", "0.75"))


class GpuGate:
    """One job at a time; photos first, a clip that has waited CLIP_MAX_WAIT_S next."""

    def __init__(self) -> None:
        self._busy = False
        self._cur = None      # (started_at, est_s) of the running job
        self._photos = []     # waiting photos: [arrived_at, est_s], in arrival order
        self._clips = []      # waiting clips: [arrived_at, est_s], in arrival order
        self._cond = None

    def _c(self) -> asyncio.Condition:
        if self._cond is None:   # made inside the framework's running loop
            self._cond = asyncio.Condition()
        return self._cond

    def photo_wait(self) -> float:
        """Seconds a photo arriving now would wait before it starts."""
        now = time.time()
        w = 0.0
        if self._busy and self._cur:
            w += max(2.0, self._cur[1] - (now - self._cur[0]))
        w += sum(e[1] for e in self._photos)
        for arrived, est in self._clips:   # a clip that is overdue by then goes first
            if now + w - arrived >= CLIP_MAX_WAIT_S:
                w += est
        return w

    async def acquire(self, video: bool, est: float) -> None:
        c = self._c()
        async with c:
            entry = [time.time(), est]
            line = self._clips if video else self._photos
            line.append(entry)

            def my_turn() -> bool:
                if self._busy:
                    return False
                overdue = bool(self._clips) and time.time() - self._clips[0][0] >= CLIP_MAX_WAIT_S
                if video:
                    return self._clips[0] is entry and (not self._photos or overdue)
                return self._photos[0] is entry and not overdue

            try:
                while not my_turn():
                    try:   # wake on release, and every second for the clip's clock
                        await asyncio.wait_for(c.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
            finally:
                line.remove(entry)
            self._busy = True
            self._cur = (time.time(), est)

    async def release(self) -> None:
        c = self._c()
        async with c:
            self._busy = False
            self._cur = None
            c.notify_all()


_gate = GpuGate()


async def _post(url: str, body: dict | None = None, timeout: float = 30) -> int:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
            async with s.post(url, json=body if body is not None else {}) as r:
                await r.read()
                return r.status
    except Exception:
        return 0


FREE_TARGET_MB = int(os.environ.get("HYBRID_FREE_TARGET_MB", "16000"))


async def _wait_vram_free(target_mb: int, limit_s: float) -> int:
    """Poll ComfyUI's /system_stats until the GPU has `target_mb` free (or the limit passes);
    returns the last reading in MB (-1 when unreadable)."""
    stop = time.time() + limit_s
    free_mb = -1
    while True:
        try:
            dev = ((await _get_json("/system_stats", 5)).get("devices") or [{}])[0]
            free_mb = int((dev.get("vram_free") or 0) / 1048576)
        except Exception:
            pass
        if free_mb >= target_mb or time.time() >= stop:
            return free_mb
        await asyncio.sleep(0.5)


async def _enter(kind: str) -> None:
    """Make the GPU ready for a `kind` job ("video" | "comfy" | "forge")."""
    global _mode, _forge_loaded
    if kind == _mode:
        return
    t0 = time.time()
    steps = []
    if kind == "video":
        # Forge's checkpoint out; ComfyUI evicts an Anima photo's models itself
        if _forge_loaded:
            st = await _post(f"{FORGE_URL}/sdapi/v1/unload-checkpoint", timeout=120)
            steps.append(f"forge unload {st}")
            _forge_loaded = False
    elif kind == "forge":
        if _mode == "video":
            # the clip's weights leave the GPU (ComfyUI keeps them in RAM for the next clip)
            st = await _post(f"{COMFY_URL}/free", {"unload_models": True}, timeout=120)
            # /free only sets a flag; ComfyUI's prompt loop unloads a moment later. Wait for the
            # VRAM to actually come back before Forge loads into it.
            freed = await _wait_vram_free(FREE_TARGET_MB, 20.0)
            steps.append(f"comfy free {st} ({freed} MB free)")
        # No reload call: Neo's unload-checkpoint clears forge_hash, so the photo's own
        # txt2img reloads the checkpoint (forge_model_reload) — from the page cache after
        # the first time. Neo has no /sdapi/v1/reload-checkpoint route anyway.
        _forge_loaded = True
    # kind == "comfy": nothing — ComfyUI swaps its own models, and a loaded Forge checkpoint
    # fits beside an Anima photo (the photo fleet runs exactly that)
    _log(f"gpu {_mode} -> {kind} in {time.time() - t0:.1f}s ({', '.join(steps) or 'no-op'})")
    _mode = kind


# ── readiness ───────────────────────────────────────────────────────────────

def _combo_options(spec) -> list:
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
        if not all(n in _combo_options(spec) for n in names):
            return False
    r = requests.get(f"{COMFY_URL}/object_info/{H3_NODE}", timeout=15)
    return r.ok and H3_NODE in r.json()


def _forge_probe() -> bool:
    """A tiny real render (branch `neo`'s probe): Neo answers /sdapi before its extension
    arg tables settle, and a request carrying alwayson_scripts 500s in that window."""
    payload = {
        "prompt": "1boy", "negative_prompt": "girl",
        "steps": 1, "width": 256, "height": 320, "cfg_scale": 5,
        "sampler_name": "Euler a", "seed": 1,
        "send_images": False, "save_images": False,
        "override_settings": {"sd_model_checkpoint": FORGE_PROBE_CHECKPOINT},
        "override_settings_restore_afterwards": False,
        "alwayson_scripts": {"ADetailer": {"args": [False, False, {"ad_model": "None"}]}},
    }
    try:
        return requests.post(f"{FORGE_URL}/sdapi/v1/txt2img", json=payload, timeout=300).ok
    except Exception:
        return False


def _readiness_shim() -> None:
    global _ready, _forge_ready, _forge_loaded, _mode
    deadline = time.time() + STARTUP_TIMEOUT_S
    comfy_ok = False
    while time.time() < deadline:
        if os.path.exists("/.provisioning_failed"):
            _append_log("provisioning failed: /.provisioning_failed exists")
            break
        try:
            if requests.get(f"{COMFY_URL}/system_stats", timeout=5).ok and _models_visible():
                comfy_ok = True
                break
        except Exception:
            pass
        time.sleep(5)
    if not comfy_ok:
        _append_log(FAIL_TOKEN)
        return
    # Forge second, before the benchmark clip runs, so nothing else holds the GPU then
    fdeadline = time.time() + FORGE_STARTUP_TIMEOUT_S
    while time.time() < fdeadline:
        try:
            r = requests.get(f"{FORGE_URL}/sdapi/v1/sd-models", timeout=5)
            if r.ok and isinstance(r.json(), list) and r.json() and _forge_probe():
                _forge_ready = True
                break
        except Exception:
            pass
        time.sleep(5)
    if _forge_ready:
        try:
            requests.post(f"{FORGE_URL}/sdapi/v1/unload-checkpoint", timeout=120)
        except Exception:
            pass
        _log("forge ready (checkpoint unloaded for the benchmark clip)")
    else:
        _log("FORGE NOT READY — serving clips only; /forge/txt2img refuses")
    _forge_loaded = False
    _mode = None
    _ready = True
    _append_log(READY_TOKEN)


threading.Thread(target=_readiness_shim, daemon=True).start()


# ── the benchmark clip (branch `video`'s, unchanged) ─────────────────────────

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
BENCH_W, BENCH_H, BENCH_FRAMES = 512, 736, 73
_BENCH_PNG_B64 = base64.b64encode(_png(BENCH_W, BENCH_H)).decode()
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
        "strict": True,
    }


# ── workloads (all in ~GPU-seconds on a 3090, the unit the clip benchmark measures) ─

def h3_est_seconds(width: float, height: float, frames: float, steps: float = 4.0) -> float:
    fmp = float(frames) * float(width) * float(height) / 1e6
    return 6.0 + float(steps) * fmp * (0.18 + 0.0016 * fmp)


def ltx_est_seconds(width: float, height: float, frames: float) -> float:
    """The app's ltxEstSeconds (api/_lib/clip-graph.ts), on the full canvas."""
    fmp = float(frames) * float(width) * float(height) / 1e6
    return 42.0 + 0.70 * fmp + 0.00085 * fmp * fmp


DEFAULT_WORKLOAD = h3_est_seconds(BENCH_W, BENCH_H, BENCH_FRAMES)
PHOTO_SECONDS = 31.0   # a photo's median render on a 3090 (the Oct 2 – 9 2026 week)
VIDEO_NODES = {
    H3_NODE, "SaveVideo", "CreateVideo",
    "EmptyLTXVLatentVideo", "LTXVConditioning", "LTXVImgToVideoInplace", "LTXVEmptyLatentAudio",
    "LTXVLatentUpsampler", "WanImageToVideo", "WanFirstLastFrameToVideo",
}


def _is_video(graph) -> bool:
    return isinstance(graph, dict) and any(
        isinstance(n, dict) and n.get("class_type") in VIDEO_NODES for n in graph.values())


def clip_workload(payload: dict) -> float:
    """/comfy/render: a clip priced as branch `video` does (H3) or by the app's LTX fit; an
    Anima photo as one photo."""
    try:
        graph = payload.get("prompt") or {}
        if not _is_video(graph):
            return PHOTO_SECONDS
        h3 = ltx = None
        steps = 4.0
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            ct = node.get("class_type")
            inp = node.get("inputs") or {}
            if ct == H3_NODE:
                h3 = (float(inp.get("width", 1344)), float(inp.get("height", 768)), float(inp.get("length", 124)))
            elif ct == "BasicScheduler":
                steps = float(inp.get("steps", steps))
            elif ct == "EmptyLTXVLatentVideo":
                # stage 1 runs at half size; the clip is twice that
                ltx = (2 * float(inp.get("width", 512)), 2 * float(inp.get("height", 384)),
                       float(inp.get("length", 121)))
        if h3:
            return h3_est_seconds(h3[0], h3[1], h3[2], steps)
        if ltx:
            return ltx_est_seconds(ltx[0], ltx[1], ltx[2])
        return DEFAULT_WORKLOAD
    except Exception:
        return DEFAULT_WORKLOAD


def forge_workload(payload: dict) -> float:
    try:
        steps = float(payload.get("steps", 35))
        mp = float(payload.get("width", 832)) * float(payload.get("height", 1216)) / 1e6
        return PHOTO_SECONDS * (steps * mp) / (35 * 1.012)
    except Exception:
        return PHOTO_SECONDS


# ── ComfyUI render (branch `video`'s, behind the gate) ───────────────────────

MIME = {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska", ".webp": "image/webp",
        ".gif": "image/gif", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


async def _comfy_post(path: str, body: dict, timeout: float = 15) -> None:
    await _post(f"{COMFY_URL}{path}", body, timeout)


async def _cancel(pid: str | None) -> None:
    if pid:
        await _comfy_post("/queue", {"delete": [pid]})
    await _comfy_post("/interrupt", {"prompt_id": pid} if pid else {})


async def _get_json(path: str, timeout: float = 10):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
        async with s.get(f"{COMFY_URL}{path}") as r:
            return await r.json(content_type=None)


async def _drain_orphans(deadline: float) -> None:
    """Behind the gate only one job submits to ComfyUI at a time, so anything already in its
    queue is left over from a caller that went away."""
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


OPTIONAL_LORAS = {
    "NSFW_ANIME_V7_H3-step00019500.safetensors",
    "H3_Motion_Booster_anime.safetensors",
    "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors",
    "epic_cumshots-MiniMaxH3-ALPHA-CUMSH0T.safetensors",
    "HMCumshot_v1_e120.safetensors",
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
        for _ in range(len(drop) + 1):
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
    video = _is_video(prompt)
    est = clip_workload({"prompt": prompt}) * HYBRID_SPEED
    if not video:
        wait = _gate.photo_wait()
        if wait > PHOTO_MAX_WAIT_S:
            _log(f"comfy photo turned away: {wait:.0f}s wait")
            return {"error": "busy", "wait_s": round(wait)}
    t_wait = time.time()
    await _gate.acquire(video, est)
    waited = time.time() - t_wait
    t0 = time.time()
    try:
        await _enter("video" if video else "comfy")
        try:
            budget = float(timeout_s) - waited
        except Exception:
            budget = 600.0
        result = await _comfy_render(prompt, budget, images)
    finally:
        await _gate.release()
    _log(f"{'clip' if video else 'comfy photo'} waited {waited:.1f}s, ran {time.time() - t0:.1f}s"
         f"{' ERROR ' + str(result.get('error')) if 'error' in result else ''}")
    if strict and "error" in result:
        raise RuntimeError(f"{result['error']}: {str(result.get('detail', ''))[:300]}")
    return result


async def _comfy_render(prompt: dict, timeout_s: float, images: dict) -> dict:
    if not isinstance(prompt, dict) or not prompt:
        return {"error": "missing prompt graph"}
    if _gpu_fault:
        return {"error": "gpu fault", "detail": _gpu_fault}
    timeout_s = max(30.0, float(timeout_s))
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


# ── Forge (branch `neo`'s txt2img, reached only through here, behind the gate) ─

async def forge_txt2img(**body) -> dict:
    if not _forge_ready:
        return {"error": "forge not ready"}
    if _gpu_fault:
        return {"error": "gpu fault", "detail": _gpu_fault}
    wait = _gate.photo_wait()
    if wait > PHOTO_MAX_WAIT_S:
        _log(f"forge photo turned away: {wait:.0f}s wait")
        return {"error": "busy", "wait_s": round(wait)}
    t_wait = time.time()
    await _gate.acquire(False, forge_workload(body) * HYBRID_SPEED)
    waited = time.time() - t_wait
    t0 = time.time()
    try:
        await _enter("forge")
        try:
            timeout = aiohttp.ClientTimeout(total=FORGE_RENDER_TIMEOUT_S)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.post(f"{FORGE_URL}/sdapi/v1/txt2img", json=body) as r:
                    text = await r.text()
                    if r.status != 200:
                        _note_render_error(text)
                        out = {"error": f"forge {r.status}", "detail": text[:500]}
                    else:
                        out = json.loads(text)
        except asyncio.CancelledError:
            await asyncio.shield(_post(f"{FORGE_URL}/sdapi/v1/interrupt", timeout=10))
            raise
        except Exception as ex:
            out = {"error": f"forge unreachable: {ex}"}
    finally:
        await _gate.release()
    _log(f"forge photo waited {waited:.1f}s, ran {time.time() - t0:.1f}s"
         f"{' ERROR ' + str(out.get('error')) if 'error' in out else ''}")
    return out


async def forge_progress() -> dict:
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(f"{FORGE_URL}/sdapi/v1/progress", params={"skip_current_image": "true"}) as r:
                if r.status == 200:
                    return await r.json()
    except Exception:
        pass
    return {}


# ── worker config ────────────────────────────────────────────────────────────

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=COMFY_PORT,
    model_log_file=LOG_FILE,
    model_healthcheck_url=f"http://127.0.0.1:{HEALTH_PORT}/health",
    handlers=[
        # Both GPU routes are "parallel" to the framework: its own line is one FIFO for every
        # route, so the GpuGate above does the ordering (photos first) and the turning away.
        HandlerConfig(
            route="/comfy/render",
            allow_parallel_requests=True,
            max_queue_time=600.0,
            workload_calculator=clip_workload,
            remote_function=comfy_render,
            benchmark_config=BenchmarkConfig(
                generator=make_benchmark_payload,
                concurrency=1,
                runs=1,
            ),
        ),
        HandlerConfig(
            route="/forge/txt2img",
            allow_parallel_requests=True,
            max_queue_time=None,
            workload_calculator=forge_workload,
            remote_function=forge_txt2img,
        ),
        HandlerConfig(
            route="/sdapi/v1/progress",
            allow_parallel_requests=True,
            max_queue_time=None,
            workload_calculator=lambda payload: 0.0,
            remote_function=forge_progress,
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
