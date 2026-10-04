#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Peepy Vast.ai provisioning — VIDEO workers (Oct 2026).
# Branch `video` only. The peepy-anima endpoint's template fetches this file and
# runs PYWORKER_REF=video; the image fleet keeps vast-provisioning-neo.sh on `neo`.
#
# A video worker is ComfyUI ONLY: no Forge, no image models, no R2 image sync. The
# whole card goes to MiniMax H3 fl2va (image-to-video): the int8 transformer (~21 GB),
# the nvfp4 Qwen3-VL text encoder (~15.7 GB) and the int8 video VAE (~2.8 GB), with the
# 4-step turbo LoRA and the yaoi LoRA for two-man clips: ~41.6 GB in all. A 5 s clip at
# 1024x1376 peaks at 23.8 of 24.5 GB on a 3090, so nothing else may hold VRAM.
# (Until Oct 2026 this branch provisioned Wan 2.2 I2V A14B; H3 replaced it.)
#
# Models come from Hugging Face (Comfy-Org/MiniMax-H3, pinned to one revision) and
# Civitai (the yaoi LoRA), all public. If R2 ever holds a mirror under
# peepy-models/video/<dir>/, it is tried first and the public hosts only fill gaps.
# Video weights NEVER go into the image folders of the bucket (checkpoints/, lora/,
# text_encoders/, vae/…): every image worker syncs those at boot.
#
# Template env: PYWORKER_REPO=https://github.com/thirdku/peepy-vast-worker,
# PYWORKER_REF=video, and optionally the R2 rclone block + R2_BUCKET, HF_TOKEN, CIVITAI_TOKEN.
# ─────────────────────────────────────────────────────────────────────────────

source /venv/main/bin/activate 2>/dev/null || true
COMFY_DIR=${WORKSPACE:-/workspace}/ComfyUI
COMFY_VENV=${WORKSPACE:-/workspace}/comfy-venv
COMFY_PORT=${COMFY_INTERNAL_PORT:-8188}
# Same pin as the image fleet's sidecar. It ships everything H3 needs (checked on the
# live video worker, Oct 3 2026): the core MiniMaxH3ImageToVideo node, CLIPLoader type
# `minimax`, and comfy-kitchen (from ComfyUI's own requirements) with the int8_convrot
# and nvfp4 kernels. int8_convrot needs torch cu130, hence a driver >= 580. Roll
# forward only by canarying a new SHA on a disposable box first.
COMFY_PIN=b16023b004d3b1bfbbd6463414dc20da1b36cc4d

HF=https://huggingface.co
# Pinned revision: a moved `main` can never swap the weights under a running fleet.
H3_REPO="$HF/Comfy-Org/MiniMax-H3/resolve/e5eb578a89295337b8ff433a035929ce0279e0b6"
# dir under ComfyUI/models | file name | url | exact size in bytes | sha256 (optional)
# Hugging Face files are pinned by revision and checked by exact size; the Civitai file
# also by sha256 (it comes from a less trustworthy host, and from a signed redirect).
VIDEO_MODELS=(
    "diffusion_models|minimax_h3_fl2va_pruned_int8_convrot.safetensors|$H3_REPO/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors|20970379616|"
    "text_encoders|qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors|$H3_REPO/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors|15687142551|"
    "vae|minimax_h3_video_vae_int8_convrot.safetensors|$H3_REPO/vae/minimax_h3_video_vae_int8_convrot.safetensors|2811065184|"
    "loras|minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors|$H3_REPO/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors|1956192992|"
    "loras|yaoi_h3_lora_000002500.safetensors|https://civitai.com/api/download/models/3234992?fileId=3117398|155110336|F769A344264F620957C24916F376F5C1A83DEFDE45FD43351BECA2ECA0F6E381"
)
# Optional extras (Oct 4 2026). Same format and checks, but a worker starts without any of
# them — a failed fetch only warns, and provisioning_verify doesn't look at them. The app's
# clip graph loads the anime LoRA (0.6) and the Motion Booster (0.4) on every clip since the
# motion builder; worker.py drops either from the graph when it isn't on disk, so a clip still
# renders (without that look). Civitai serves those two to signed-in accounts only: they come
# from the R2 mirror (peepy-models/video/loras/) or, with CIVITAI_TOKEN in the template env, from
# Civitai itself (the token goes to civitai.com only).
#   turbo v1.2 (lightx2v 4-step, 768p) — the newer turbo, to compare with v1.0
#   2D Anime Style NSFW (v0.4, the 19.5k-step file) — trigger "2d anime style"
#   Faster! Harder! Shake Harder! (Motion Booster, Anime Edition)
VIDEO_EXTRAS=(
    "loras|minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors|https://civitai.com/api/download/models/3297123?fileId=3181960|1956193000|C8168EBC17BBACC4296103DDA2FEC1BA85B24392FA08CF2BFBCEF0CFF0DC3CC8"
    "loras|NSFW_ANIME_V7_H3-step00019500.safetensors|https://civitai.com/api/download/models/3286171?fileId=3170500|596450480|C69A8E719B6784A8E475004CD47D34D1DDEFBB5DAA2D7670632CD3B459490B8D"
    "loras|H3_Motion_Booster_anime.safetensors|https://civitai.com/api/download/models/3299686?fileId=3184584|155110280|CF23F3F8AC3D663DD3EA49482F90DD4321EFF6A052AD6DFC7518321C01B78B9B"
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

# ── Models: R2 mirror first (if any), then the public hosts, all five files in parallel ─
# 0 when $1 exists with exactly $2 bytes and, when $3 is set, that sha256 (any case).
function file_ok() {
    local f="$1" size="$2" sha="$3"
    [[ -f "$f" ]] && (( $(stat -c %s "$f") == size )) || return 1
    [[ -z "$sha" ]] && return 0
    [[ "$(sha256sum "$f" | cut -d' ' -f1)" == "${sha,,}" ]]
}

# A Civitai token (optional, template env CIVITAI_TOKEN) unlocks the sign-in-only files. wget
# would resend an Authorization header to every redirect hop, so the token is used here only to
# walk the civitai.com redirects (curl, no follow); the signed delivery URL that comes back is
# fetched without it. Any other URL — or no token — passes through unchanged.
function civitai_src() {
    local u="$1" next i
    if [[ -z "$CIVITAI_TOKEN" || "$u" != "https://civitai.com/"* ]]; then
        echo "$u"; return
    fi
    for i in 1 2 3 4; do
        [[ "$u" == "https://civitai.com/"* ]] || break
        next=$(curl -s -o /dev/null -w '%{redirect_url}' --max-time 60 -A "Mozilla/5.0" -H "Authorization: Bearer $CIVITAI_TOKEN" "$u")
        [[ -n "$next" ]] || break
        u="$next"
    done
    echo "$u"
}

function fetch_one() {
    local dir="$1" name="$2" url="$3" size="$4" sha="$5"
    local dest="$COMFY_DIR/models/$dir/$name"
    mkdir -p "$COMFY_DIR/models/$dir"
    if file_ok "$dest" "$size" "$sha"; then
        echo "[provision] have $name"
        return 0
    fi
    rm -f "$dest"
    if command -v rclone >/dev/null && [[ -n "$R2_BUCKET" && -n "$RCLONE_CONFIG_R2_ENDPOINT" ]]; then
        rclone copyto "r2:${R2_BUCKET}/video/${dir}/${name}" "$dest" \
            --multi-thread-streams 8 --retries 3 -q 2>/dev/null || true
        if file_ok "$dest" "$size" "$sha"; then
            echo "[provision] $name from R2"
            return 0
        fi
        rm -f "$dest"
    fi
    if [[ -z "$url" ]]; then
        echo "[provision] $name is not in the R2 mirror (it has no public source)"
        return 1
    fi
    # The Hugging Face token (optional) goes to Hugging Face only, never to Civitai.
    local auth=()
    [[ -n "$HF_TOKEN" && "$url" == "$HF/"* ]] && auth=(--header="Authorization: Bearer $HF_TOKEN")
    # -c resumes $dest.part, which is kept between attempts AND between runs of this
    # script (the base image re-runs it after a failure), so a drop 18 GB into the
    # 21 GB transformer costs the last 3 GB, not all 21. Both hosts honour Range. Civitai
    # refuses wget's default user agent, hence Mozilla/5.0. The file check decides, not
    # wget's exit code.
    local i got
    for i in 1 2 3; do
        if [[ -f "$dest.part" ]] && (( $(stat -c %s "$dest.part") > size )); then
            rm -f "$dest.part"
        fi
        wget -c -q --read-timeout=120 --user-agent="Mozilla/5.0" "${auth[@]}" -O "$dest.part" "$(civitai_src "$url")"
        if file_ok "$dest.part" "$size" "$sha"; then
            mv "$dest.part" "$dest"
            echo "[provision] $name downloaded"
            return 0
        fi
        got=$(stat -c %s "$dest.part" 2>/dev/null || echo 0)
        # Full length but the wrong bytes (sha256 mismatch): resuming can't fix that.
        if (( got >= size )); then
            echo "[provision] $name: $got bytes but the wrong checksum — starting over"
            rm -f "$dest.part"
        fi
        echo "[provision] $name download attempt $i failed ($got of $size bytes) — resuming"
        sleep 10
    done
    echo "[provision] FATAL: could not fetch $name (partial file kept for the next run to resume)"
    return 1
}

function provisioning_get_models() {
    local entry pids=() failed=0
    for entry in "${VIDEO_MODELS[@]}"; do
        IFS='|' read -r dir name url size sha <<< "$entry"
        fetch_one "$dir" "$name" "$url" "$size" "$sha" &
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
    # the optional extras, after the required set (never fatal)
    pids=()
    for entry in "${VIDEO_EXTRAS[@]}"; do
        IFS='|' read -r dir name url size sha <<< "$entry"
        ( fetch_one "$dir" "$name" "$url" "$size" "$sha"             || echo "[provision] WARN: optional $name not fetched — the worker runs without it" ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
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

# 0 when ComfyUI's /object_info lists $3 among the options of input $2 on node $1.
# This pin serves two schema shapes: the loaders still send [["a","b"], {...}], newer
# nodes (KSamplerSelect, BasicScheduler) send ["COMBO", {"options": ["a","b"]}].
function comfy_option() {
    "${COMFY_VENV}/bin/python" - "$COMFY_PORT" "$1" "$2" "$3" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
port, node, field, want = sys.argv[1:5]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/object_info/{node}", timeout=15) as r:
    inputs = (json.load(r).get(node) or {}).get("input") or {}
spec = (inputs.get("required") or {}).get(field) or (inputs.get("optional") or {}).get(field) or []
opts = []
if spec and isinstance(spec[0], list):
    opts = spec[0]
elif spec and spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
    opts = spec[1].get("options") or []
sys.exit(0 if want in opts else 1)
PY
}

# The loader node + input that must list a file of each models/ folder.
function loader_for() {
    case "$1" in
        diffusion_models) echo "UNETLoader unet_name" ;;
        text_encoders)    echo "CLIPLoader clip_name" ;;
        vae)              echo "VAELoader vae_name" ;;
        loras)            echo "LoraLoaderModelOnly lora_name" ;;
    esac
}

function provisioning_verify() {
    local ok=1 entry dir name url size sha
    [[ -f "${COMFY_DIR}/main.py" ]] || { echo "[provision] FATAL: ComfyUI missing"; ok=0; }
    "${COMFY_VENV}/bin/python" -c "import torch, aiohttp, av" 2>/dev/null \
        || { echo "[provision] FATAL: comfy venv broken (torch/aiohttp/av)"; ok=0; }
    "${COMFY_VENV}/bin/python" -c "import av, sys; sys.exit(0 if 'libx264' in av.codecs_available else 1)" 2>/dev/null \
        || { echo "[provision] FATAL: PyAV has no libx264 (SaveVideo can't write mp4)"; ok=0; }
    "${COMFY_VENV}/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
        || { echo "[provision] FATAL: torch sees no CUDA device"; ok=0; }
    "${COMFY_VENV}/bin/python" -c "import comfy_kitchen" 2>/dev/null \
        || { echo "[provision] FATAL: comfy-kitchen missing (H3's int8_convrot / nvfp4 kernels)"; ok=0; }
    for entry in "${VIDEO_MODELS[@]}"; do
        IFS='|' read -r dir name url size sha <<< "$entry"
        file_ok "$COMFY_DIR/models/$dir/$name" "$size" "$sha" \
            || { echo "[provision] FATAL: $dir/$name missing, wrong size or wrong checksum"; ok=0; }
    done
    # ComfyUI itself must come up and register the nodes the clip graph uses.
    local node i up served=1
    for node in MiniMaxH3ImageToVideo SamplerCustomAdvanced BasicGuider BasicScheduler KSamplerSelect \
                RandomNoise ImageFromBatch ImageCrop CreateVideo SaveVideo \
                LoraLoaderModelOnly UNETLoader CLIPLoader VAELoader; do
        up=0
        for i in $(seq 1 60); do
            if curl -sf --max-time 10 "http://127.0.0.1:${COMFY_PORT}/object_info/${node}" | grep -q "\"${node}\""; then
                up=1; break
            fi
            sleep 5
        done
        (( up )) || { echo "[provision] FATAL: ComfyUI never served node ${node}"; ok=0; served=0; break; }
    done
    if (( served )); then
        comfy_option CLIPLoader type minimax \
            || { echo "[provision] FATAL: CLIPLoader has no 'minimax' type (ComfyUI too old for H3)"; ok=0; }
        comfy_option KSamplerSelect sampler_name res_multistep \
            || { echo "[provision] FATAL: KSamplerSelect has no 'res_multistep' sampler"; ok=0; }
        local loader field
        for entry in "${VIDEO_MODELS[@]}"; do
            IFS='|' read -r dir name url size sha <<< "$entry"
            read -r loader field <<< "$(loader_for "$dir")"
            comfy_option "$loader" "$field" "$name" \
                || { echo "[provision] FATAL: ComfyUI's ${loader} doesn't list ${name}"; ok=0; }
        done
    fi
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
