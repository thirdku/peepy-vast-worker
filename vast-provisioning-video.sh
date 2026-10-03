#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Peepy Vast.ai provisioning — VIDEO workers (Oct 2026).
# Branch `video` only. The peepy-anima endpoint's template fetches this file and
# runs PYWORKER_REF=video; the image fleet keeps vast-provisioning-neo.sh on `neo`.
#
# A video worker is ComfyUI ONLY: no Forge, no image models, no R2 image sync. The
# whole card goes to Wan 2.2 I2V A14B (two fp8 halves, ~14.3 GB each, swapped in
# and out per clip), so nothing else may hold VRAM.
#
# Models come from Hugging Face (all public). If R2 ever holds a mirror under
# peepy-models/video/<dir>/, it is tried first and Hugging Face only fills gaps.
# Video weights NEVER go into the image folders of the bucket (checkpoints/, lora/,
# text_encoders/, vae/…): every image worker syncs those at boot.
#
# Template env: PYWORKER_REPO=https://github.com/thirdku/peepy-vast-worker,
# PYWORKER_REF=video, and optionally the R2 rclone block + R2_BUCKET, HF_TOKEN.
# ─────────────────────────────────────────────────────────────────────────────

source /venv/main/bin/activate 2>/dev/null || true
COMFY_DIR=${WORKSPACE:-/workspace}/ComfyUI
COMFY_VENV=${WORKSPACE:-/workspace}/comfy-venv
COMFY_PORT=${COMFY_INTERNAL_PORT:-8188}
# Same pin as the image fleet's sidecar: it already ships the Wan, CreateVideo and
# SaveVideo nodes (checked on the live workers, Oct 3 2026). Roll forward only by
# canarying a new SHA on a disposable box first.
COMFY_PIN=b16023b004d3b1bfbbd6463414dc20da1b36cc4d

HF=https://huggingface.co
WAN_REPO="$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
LORA_REPO="$HF/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_Lightx2v"
# dir under ComfyUI/models | file name | url | minimum size in bytes (catches truncated downloads)
VIDEO_MODELS=(
    "diffusion_models|wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors|$WAN_REPO/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors|14000000000"
    "diffusion_models|wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors|$WAN_REPO/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors|14000000000"
    "text_encoders|umt5_xxl_fp8_e4m3fn_scaled.safetensors|$WAN_REPO/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors|6500000000"
    "vae|wan_2.1_vae.safetensors|$WAN_REPO/vae/wan_2.1_vae.safetensors|240000000"
    "loras|Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_260412_rank_64_fp16.safetensors|$LORA_REPO/Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_260412_rank_64_fp16.safetensors|600000000"
    "loras|Wan_2_2_I2V_A14B_LOW_lightx2v_4step_lora_260412_rank_64_fp16.safetensors|$LORA_REPO/Wan_2_2_I2V_A14B_LOW_lightx2v_4step_lora_260412_rank_64_fp16.safetensors|600000000"
)

function provisioning_start() {
    printf "\n##############################################\n#     Peepy VIDEO worker provisioning        #\n##############################################\n\n"
    # The pyworker service goes in FIRST: it waits while /.provisioning exists, and if a
    # later step fails the base image marks /.provisioning_failed and starts services
    # anyway, so worker.py is there to read the marker and report PEEPY_VIDEO_START_FAILED.
    # Installed last (as neo does), a failed worker would run no pyworker and never report.
    provisioning_install_pyworker
    provisioning_install_comfyui
    provisioning_get_models
    provisioning_install_comfy_service
    provisioning_verify
    printf "\nProvisioning complete:  Application will start now\n\n"
}

# ── ComfyUI in its own venv (no Forge venv to borrow torch from here) ────────
function provisioning_install_comfyui() {
    if ! command -v uv >/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"

    if [[ ! -d "$COMFY_DIR" ]]; then
        git clone https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"             || { echo "[provision] FATAL: ComfyUI clone failed"; exit 1; }
    fi
    git -C "$COMFY_DIR" checkout --quiet "$COMFY_PIN" \
        || echo "[provision] WARN: ComfyUI pin checkout FAILED — running HEAD (unvalidated!)"

    if [[ ! -d "$COMFY_VENV" ]]; then
        uv venv "$COMFY_VENV" --python 3.13 || { echo "[provision] FATAL: comfy venv"; exit 1; }
    fi
    uv pip install --python "$COMFY_VENV/bin/python" pip || { echo "[provision] FATAL: pip"; exit 1; }
    # The image fleet's proven torch build (cu130); torchaudio is only for ComfyUI's
    # optional audio nodes, so its failure is not fatal.
    uv pip install --python "$COMFY_VENV/bin/python" "torch==2.11.0" "torchvision==0.26.0" \
        --index-url https://download.pytorch.org/whl/cu130         || { echo "[provision] FATAL: torch install failed"; exit 1; }
    uv pip install --python "$COMFY_VENV/bin/python" "torchaudio==2.11.0" \
        --index-url https://download.pytorch.org/whl/cu130 \
        || echo "[provision] WARN: torchaudio not installed (audio nodes unavailable)"
    grep -viE '^(torch|torchvision|torchaudio)([=<>~ ]|$)' "$COMFY_DIR/requirements.txt" > /tmp/comfy-req.txt
    "$COMFY_VENV/bin/pip" install -q -r /tmp/comfy-req.txt         || { echo "[provision] FATAL: ComfyUI requirements failed"; exit 1; }
    "$COMFY_VENV/bin/pip" cache purge >/dev/null 2>&1 || true
    mkdir -p "$COMFY_DIR/input" "$COMFY_DIR/output"
    echo "[provision] ComfyUI installed at ${COMFY_DIR} (pin ${COMFY_PIN:0:8})"
}

# ── Models: R2 mirror first (if any), then Hugging Face, all four big files in parallel ─
function fetch_one() {
    local dir="$1" name="$2" url="$3" min="$4"
    local dest="$COMFY_DIR/models/$dir/$name"
    mkdir -p "$COMFY_DIR/models/$dir"
    if [[ -f "$dest" ]] && (( $(stat -c %s "$dest") >= min )); then
        echo "[provision] have $name"
        return 0
    fi
    if command -v rclone >/dev/null && [[ -n "$R2_BUCKET" && -n "$RCLONE_CONFIG_R2_ENDPOINT" ]]; then
        rclone copyto "r2:${R2_BUCKET}/video/${dir}/${name}" "$dest" \
            --multi-thread-streams 8 --retries 3 -q 2>/dev/null || true
        if [[ -f "$dest" ]] && (( $(stat -c %s "$dest") >= min )); then
            echo "[provision] $name from R2"
            return 0
        fi
    fi
    local auth=()
    [[ -n "$HF_TOKEN" ]] && auth=(--header="Authorization: Bearer $HF_TOKEN")
    local i
    for i in 1 2 3; do
        rm -f "$dest.part"
        if wget -q "${auth[@]}" -O "$dest.part" "$url" && (( $(stat -c %s "$dest.part") >= min )); then
            mv "$dest.part" "$dest"
            echo "[provision] $name from Hugging Face"
            return 0
        fi
        echo "[provision] $name download attempt $i failed — retrying"
        sleep 10
    done
    rm -f "$dest.part"
    echo "[provision] FATAL: could not fetch $name"
    return 1
}

function provisioning_get_models() {
    local entry pids=() failed=0
    for entry in "${VIDEO_MODELS[@]}"; do
        IFS='|' read -r dir name url min <<< "$entry"
        fetch_one "$dir" "$name" "$url" "$min" &
        pids+=($!)
    done
    local p
    for p in "${pids[@]}"; do
        wait "$p" || failed=1
    done
    if [[ $failed -ne 0 ]]; then
        echo "[provision] FATAL: model download failed"
        exit 1
    fi
}

# ── Supervisor service ───────────────────────────────────────────────────────
# --disable-metadata: SaveVideo otherwise writes the whole prompt graph into every mp4,
# and clips are served from a public bucket.
function provisioning_install_comfy_service() {
    cat > /etc/supervisor/conf.d/comfyui.conf <<COMFYC
[program:comfyui]
environment=PROC_NAME="%(program_name)s"
command=${COMFY_VENV}/bin/python ${COMFY_DIR}/main.py --listen 127.0.0.1 --port ${COMFY_PORT} --disable-auto-launch --disable-metadata
directory=${COMFY_DIR}
autostart=true
autorestart=true
startsecs=10
startretries=10
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=10
stdout_logfile=${WORKSPACE:-/workspace}/comfy.log
redirect_stderr=true
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=1
COMFYC
    supervisorctl reread && supervisorctl update
    echo "[provision] comfyui supervisor service installed (127.0.0.1:${COMFY_PORT} → comfy.log)"
}

# ── Verify: a worker missing pieces fails provisioning here, loudly ──────────
# exit 1 → the base image retries the script, then marks /.provisioning_failed and
# carries on; worker.py sees that marker and reports PEEPY_VIDEO_START_FAILED, so the
# worker errors out instead of going ready. The benchmark clip is the final proof.
function provisioning_verify() {
    local ok=1 entry
    [[ -f "${COMFY_DIR}/main.py" ]] || { echo "[provision] FATAL: ComfyUI missing"; ok=0; }
    "${COMFY_VENV}/bin/python" -c "import torch, aiohttp, av" 2>/dev/null \
        || { echo "[provision] FATAL: comfy venv broken (torch/aiohttp/av)"; ok=0; }
    "${COMFY_VENV}/bin/python" -c "import av, sys; sys.exit(0 if 'libx264' in av.codecs_available else 1)" 2>/dev/null \
        || { echo "[provision] FATAL: PyAV has no libx264 (SaveVideo can't write mp4)"; ok=0; }
    "${COMFY_VENV}/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
        || { echo "[provision] FATAL: torch sees no CUDA device"; ok=0; }
    for entry in "${VIDEO_MODELS[@]}"; do
        IFS='|' read -r dir name url min <<< "$entry"
        local f="$COMFY_DIR/models/$dir/$name"
        [[ -f "$f" ]] && (( $(stat -c %s "$f") >= min )) || { echo "[provision] FATAL: $dir/$name missing or short"; ok=0; }
    done
    # ComfyUI itself must come up and register the nodes the clip graph uses.
    local node i up
    for node in SaveVideo CreateVideo WanFirstLastFrameToVideo KSamplerAdvanced; do
        up=0
        for i in $(seq 1 60); do
            if curl -sf --max-time 10 "http://127.0.0.1:${COMFY_PORT}/object_info/${node}" | grep -q "\"${node}\""; then
                up=1; break
            fi
            sleep 5
        done
        (( up )) || { echo "[provision] FATAL: ComfyUI never served node ${node}"; ok=0; break; }
    done
    if [[ $ok -eq 0 ]]; then
        echo "[provision] VERIFICATION FAILED — worker is not usable"
        exit 1
    fi
    echo "[provision] verification passed"
}

# ── Serverless pyworker (same bootstrap as neo; PYWORKER_REF=video) ──────────
function provisioning_install_pyworker() {
    if [[ -z "$PYWORKER_REPO" ]]; then
        echo "[provision] PYWORKER_REPO not set — skipping pyworker service"
        return 0
    fi
    cat > /opt/supervisor-scripts/pyworker.sh <<'PYW'
#!/bin/bash
set -a; . /etc/environment 2>/dev/null; set +a
trap 'exit 0' TERM INT
while [ -f "/.provisioning" ]; do
    echo "pyworker startup paused (provisioning)..."
    sleep 5
done
for i in $(seq 1 60); do
    getent hosts github.com >/dev/null 2>&1 && break
    echo "pyworker waiting for network ($i)..."
    sleep 2
done
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -f /workspace/vast-pyworker/worker.py ]; then
    for i in $(seq 1 30); do
        rm -rf /workspace/vast-pyworker
        if git clone "${PYWORKER_REPO:-https://github.com/vast-ai/pyworker}" /workspace/vast-pyworker; then
            if [ -n "${PYWORKER_REF:-}" ]; then (cd /workspace/vast-pyworker && git checkout "${PYWORKER_REF}"); fi
            break
        fi
        echo "pyworker clone failed (attempt $i) — retrying..."
        sleep 5
    done
fi
curl -L https://raw.githubusercontent.com/vast-ai/pyworker/main/start_server.sh | bash
PYW
    chmod +x /opt/supervisor-scripts/pyworker.sh

    cat > /etc/supervisor/conf.d/pyworker.conf <<'PYC'
[program:pyworker]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/pyworker.sh
autostart=true
autorestart=unexpected
exitcodes=0
startsecs=0
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=10
stdout_logfile=/var/log/portal/pyworker.log
redirect_stderr=true
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=1
PYC
    supervisorctl reread && supervisorctl update
    supervisorctl restart pyworker 2>/dev/null || true
    echo "[provision] pyworker supervisor service installed"
}

if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
