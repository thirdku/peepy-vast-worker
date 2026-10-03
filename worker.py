"""Peepy VIDEO pyworker — ComfyUI-only image-to-video worker for Vast.ai serverless.

Branch `video` of this repo (Oct 2026). The peepy-anima endpoint's template sets
PYWORKER_REF=video and provisions with vast-provisioning-video.sh from this branch;
the image fleet runs branch `neo` and never loads this file.

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
# A fresh worker loads ~37 GB of Wan weights on its first render; be patient.
STARTUP_TIMEOUT_S = int(os.environ.get("VIDEO_STARTUP_TIMEOUT", "2400"))

READY_TOKEN = "PEEPY_VIDEO_READY"
FAIL_TOKEN = "PEEPY_VIDEO_START_FAILED"

# The files vast-provisioning-video.sh installs under ComfyUI/models/.
WAN_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
WAN_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
WAN_TE = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE = "wan_2.1_vae.safetensors"
LORA_HIGH = "Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_260412_rank_64_fp16.safetensors"
LORA_LOW = "Wan_2_2_I2V_A14B_LOW_lightx2v_4step_lora_260412_rank_64_fp16.safetensors"
REQUIRED_MODELS = {
    "UNETLoader": ("unet_name", [WAN_HIGH, WAN_LOW]),
    "CLIPLoader": ("clip_name", [WAN_TE]),
    "VAELoader": ("vae_name", [WAN_VAE]),
    "LoraLoaderModelOnly": ("lora_name", [LORA_HIGH, LORA_LOW]),
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
# ComfyUI up + every Wan file visible to its loaders. The real proof — a rendered
# clip — is the framework's benchmark, which runs right after READY_TOKEN.

def _append_log(line: str) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _models_visible() -> bool:
    for node, (field, names) in REQUIRED_MODELS.items():
        r = requests.get(f"{COMFY_URL}/object_info/{node}", timeout=15)
        if not r.ok:
            return False
        spec = (((r.json().get(node) or {}).get("input") or {}).get("required") or {}).get(field)
        options = spec[0] if isinstance(spec, list) and spec and isinstance(spec[0], list) else []
        if not all(n in options for n in names):
            return False
    r = requests.get(f"{COMFY_URL}/object_info/WanFirstLastFrameToVideo", timeout=15)
    return r.ok and "WanFirstLastFrameToVideo" in r.json()


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
# Wan 2.2 I2V A14B: the high-noise half runs the first half of the steps and hands
# its leftover noise to the low-noise half (KSamplerAdvanced pair). The still is
# pinned as both first and last frame, so the clip loops; the last frame is dropped
# because it duplicates frame 0 at the seam.

def make_clip_graph(image_name: str, prompt: str, width: int = 480, height: int = 704,
                    frames: int = 49, fps: float = 16.0, steps: int = 4, seed: int = 0) -> dict:
    split = steps // 2
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": WAN_HIGH, "weight_dtype": "default"}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": WAN_LOW, "weight_dtype": "default"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": LORA_HIGH, "strength_model": 1.0}},
        "5": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["3", 0], "lora_name": LORA_LOW, "strength_model": 1.0}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": 5.0}},
        "7": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["5", 0], "shift": 5.0}},
        "8": {"class_type": "CLIPLoader", "inputs": {"clip_name": WAN_TE, "type": "wan", "device": "default"}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["8", 0], "text": prompt}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["8", 0], "text": ""}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
        "12": {"class_type": "WanFirstLastFrameToVideo", "inputs": {
            "positive": ["9", 0], "negative": ["10", 0], "vae": ["11", 0],
            "width": width, "height": height, "length": frames, "batch_size": 1,
            "start_image": ["1", 0], "end_image": ["1", 0]}},
        "13": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["6", 0], "add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple",
            "positive": ["12", 0], "negative": ["12", 1], "latent_image": ["12", 2],
            "start_at_step": 0, "end_at_step": split, "return_with_leftover_noise": "enable"}},
        "14": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["7", 0], "add_noise": "disable", "noise_seed": seed, "steps": steps, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple",
            "positive": ["12", 0], "negative": ["12", 1], "latent_image": ["13", 0],
            "start_at_step": split, "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        "16": {"class_type": "ImageFromBatch", "inputs": {"image": ["15", 0], "batch_index": 0, "length": frames - 1}},
        "17": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "fps": fps}},
        "18": {"class_type": "SaveVideo", "inputs": {
            "video": ["17", 0], "filename_prefix": "peepy/clip", "format": "mp4",
            "format.codec": "h264", "format.codec.encoding": "re-encode",
            "format.codec.encoding.crf": 20.0}},
    }


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
_BENCH_PNG_B64 = base64.b64encode(_png(480, 704)).decode()


def make_benchmark_payload() -> dict:
    return {
        "prompt": make_clip_graph(BENCH_IMAGE, "a man breathes slowly, subtle natural motion, static camera",
                                  seed=random.randint(0, 2**31 - 1)),
        "timeout_s": 900,
        "images": {BENCH_IMAGE: _BENCH_PNG_B64},
        # The framework counts any returned value as a passing benchmark run; an
        # exception is what fails it, so a worker that can't render never goes live.
        "strict": True,
    }


def clip_workload(payload: dict) -> float:
    """frames × megapixels × sampler steps, read from the graph. The benchmark measures
    throughput in these same units, so wait_time = queued workload ÷ throughput comes out
    in seconds. The default 49-frame 480×704 4-step clip = ~66 units."""
    try:
        graph = payload.get("prompt") or {}
        frames, w, h, steps = 49.0, 480.0, 704.0, 4.0
        for node in graph.values():
            ct = node.get("class_type")
            inp = node.get("inputs") or {}
            if ct in ("WanFirstLastFrameToVideo", "WanImageToVideo"):
                frames = float(inp.get("length", frames))
                w = float(inp.get("width", w))
                h = float(inp.get("height", h))
            elif ct in ("KSamplerAdvanced", "KSampler"):
                steps = float(inp.get("steps", steps))
        return frames * (w * h / 1e6) * steps
    except Exception:
        return 66.0


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
        return await _render_inner(_scope_outputs(prompt, tag), timeout_s, deadline, holder)
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
                runs=1,          # each run is a full clip (~2 min); the warmup also loads the models
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
