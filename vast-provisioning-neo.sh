#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Peepy Vast.ai provisioning — FORGE NEO fleet (Jul 2026 migration).
# NEW FILE on purpose: the old template fetches vast-provisioning.sh from this
# repo's main on every provision — editing that in place would Neo-ify the old
# fleet mid-flight. The Neo template's PROVISIONING_SCRIPT points HERE instead.
#
# Runs on vastai/base-image (NO webui baked in — unlike vastai/sd-forge, we
# install the webui ourselves): clone Haoming02/sd-webui-forge-classic branch
# `neo`, Python 3.13 venv via uv, extensions ADetailer-Neo + sd-forge-couple,
# model stack mirrored from R2, then a dep-install warm boot so the supervisor
# start is fast. Verified end-to-end on golden box 45278938 (Jul 19 2026).
#
# Golden-box learnings baked in (each bit us live):
#   • Neo's model scanner IGNORES symlinked dirs — sync into REAL dirs.
#   • uv venvs ship without pip — extension install.py (run_pip) dies without
#     it; seed pip + ultralytics explicitly.
#   • ${NEO_DIR}/tmp must exist or gradio component init crashes extension
#     ui() during API init — which silently corrupts the script-args table
#     (every alwayson request then 500s "list assignment index out of range").
#   • Neo names upscalers by FILENAME and only scans models/ESRGAN —
#     RealESRGAN_x4plus_anime_6B.pth goes there (hr_upscaler name in
#     chat-core switches with it, env FORGE_NEO=1).
#   • /sdapi answers BEFORE extension arg tables settle — readiness must do a
#     real render (worker.py neo branch does; sd-models alone is NOT enough).
#   • Anima (homosimileAnima_v10) needs the Qwen TE + VAE modules, selected
#     per-request via override_settings.forge_additional_modules with the
#     absolute paths under ${NEO_DIR}/models/{text_encoder,VAE}/.
#
# Template env REQUIRED (same R2 read-only rclone block as the old template):
#   RCLONE_CONFIG_R2_TYPE=s3  RCLONE_CONFIG_R2_PROVIDER=Cloudflare
#   RCLONE_CONFIG_R2_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
#   RCLONE_CONFIG_R2_ACCESS_KEY_ID / RCLONE_CONFIG_R2_SECRET_ACCESS_KEY
#   R2_BUCKET=peepy-models
#   PYWORKER_REPO=https://github.com/thirdku/peepy-vast-worker
#   PYWORKER_REF=neo        ← the neo-aware worker.py branch
#   FORGE_MODEL=homosimileXLPony_v40NAIXLEPS   (benchmark checkpoint)
#
# Bucket layout additions for Neo (upload once from anywhere):
#   peepy-models/checkpoints/     → models/Stable-diffusion/   (homosimile SDXL + homosimileAnima_v10; illustrious REMOVED at cutover)
#   peepy-models/lora/            → models/Lora/               (SDXL loras — homosimile renders only)
#   peepy-models/embeddings/      → models/embeddings/         (SDXL TI negatives — note: INSIDE models/ on Neo)
#   peepy-models/text_encoders/   → models/text_encoder/       (qwen_3_06b_base.safetensors)
#   peepy-models/vae/             → models/VAE/                (qwen_image_vae.safetensors)
# ─────────────────────────────────────────────────────────────────────────────

source /venv/main/bin/activate 2>/dev/null || true
NEO_DIR=${WORKSPACE:-/workspace}/sd-webui-forge-neo
NEO_PORT=${FORGE_INTERNAL_PORT:-17860}

# PINNED to the exact SHAs the fleet was proven-clean on (Jul 23 2026). The whole
# neo incident was riding upstream HEAD — these three extensions are moving targets
# too, so freeze them. `repo@sha`; checkout happens in provisioning_get_extensions.
# ls-remote confirmed each SHA == current HEAD (dormant repos), so this changes
# nothing today and guards against a FUTURE upstream regression. Roll forward only
# by canarying a new SHA on a disposable box, then bumping the pin here.
EXTENSIONS=(
    "https://github.com/Haoming02/ADetailer-Neo@ac06b8a98505fae6ae43b03491fd6ca7ca983f81"
    "https://github.com/Haoming02/sd-forge-couple@c7884e81623d7fcbf4c92e3aea14ced8b5b6aa74"
    # Extra samplers (SA Solver / SA Solver PECE, SEEDS, Gradient Estimation, RES
    # Multistep variants…) — reForge/Comfy samplers missing from Neo (owner, Jul 20).
    "https://github.com/Panchovix/sd_forge_neo_extra_samplers@7e9e2bf4e4d8956e39004fcdc3845cb0e5bca257"
    # Extra samplers + schedulers incl. the CFG++ family ("Euler a CFG++" — the FurryMusk/
    # AnthroGlaze/FurryGloss site styles ride it) and extra schedulers (owner, Aug 17).
    # Pinned to HEAD @ Dec 5 2025 ("determinism (with sigma churn Setting)").
    "https://github.com/DenOfEquity/webUI_ExtraSchedulers@705c39ab6774c245d5ded5ff7213cba600320886"
)

ADETAILER_MODELS=(
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
)

# Neo scans models/ESRGAN only; filename becomes the API upscaler name.
ESRGAN_MODELS=(
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
)

# Anima stack fallbacks — public sources, fetched only when the R2 mirror didn't
# deliver them (wget -nc skips existing files). Keeps provisioning independent of
# the mirror's write-token situation; upload to R2 later and these become no-ops.
ANIMA_TE_MODELS=(
    "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors"
)
ANIMA_VAE_MODELS=(
    "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors"
)
ANIMA_CKPT_MODELS=(
    "https://civitai.com/api/download/models/3041842"   # homosimileAnima_v10.safetensors (content-disposition)
)

function provisioning_start() {
    provisioning_print_header
    provisioning_install_neo
    provisioning_get_extensions
    provisioning_sync_models
    provisioning_get_files "${NEO_DIR}/models/adetailer" "${ADETAILER_MODELS[@]}"
    provisioning_get_files "${NEO_DIR}/models/ESRGAN"    "${ESRGAN_MODELS[@]}"
    provisioning_get_files "${NEO_DIR}/models/text_encoder"     "${ANIMA_TE_MODELS[@]}"
    provisioning_get_files "${NEO_DIR}/models/VAE"              "${ANIMA_VAE_MODELS[@]}"
    provisioning_get_files "${NEO_DIR}/models/Stable-diffusion" "${ANIMA_CKPT_MODELS[@]}"
    # Warm boot BEFORE the config patch: a config.json that exists before Neo's
    # first launch trips verify_version() ("updating from an old version") which
    # blocks on an interactive input() and EOFErrors headless. Neo stamps its own
    # fresh config on first boot; we merge-patch our keys into it afterwards.
    provisioning_warm_boot
    provisioning_patch_forge_config

    export GIT_CONFIG_GLOBAL=/tmp/temporary-git-config
    git config --file $GIT_CONFIG_GLOBAL --add safe.directory '*'

    provisioning_install_forge_service
    provisioning_verify
    provisioning_install_pyworker
    provisioning_print_end
}

# ── Neo install ──────────────────────────────────────────────────────────────
function provisioning_install_neo() {
    if ! command -v uv >/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"

    if [[ ! -d "$NEO_DIR" ]]; then
        git clone --branch neo https://github.com/Haoming02/sd-webui-forge-classic "$NEO_DIR"
    fi
    # PIN (Jul 22 2026): the fleet used to ride Neo's moving HEAD and inherited
    # upstream regressions within HOURS of each push (fleet-wide CUDA
    # illegal-address outage — workers were running a commit pushed the same
    # morning). 025bbdda = the Jul 19 golden-box-validated cutover commit.
    # Roll forward ONLY by canarying a new SHA on a disposable box first,
    # then editing this line. Never un-pin.
    git -C "$NEO_DIR" checkout --quiet 025bbdda \
        || echo "[provision] WARN: Neo pin checkout FAILED — running branch HEAD (unvalidated!)"
    mkdir -p "$NEO_DIR/tmp"   # missing tmp = gradio init crash = corrupted API arg table

    cd "$NEO_DIR"
    if [[ ! -d .venv ]]; then
        uv venv .venv --python 3.13
    fi
    # uv venvs have no pip; ADetailer-Neo's install.py shells out to `python -m pip`.
    # Seed pip, then PIN torch cu130 into the venv OURSELVES: launch.py's --uv flow
    # installed deps into an ephemeral env that didn't persist (and its GPU check
    # then failed against the wrong torch — seen live on test box 45303289), so the
    # fleet boots WITHOUT --uv and everything must already be, or get pip-installed,
    # IN the venv. torch pinned to the version Neo's own resolver picked (Jul 2026).
    uv pip install --python .venv/bin/python pip
    uv pip install --python .venv/bin/python "torch==2.11.0" "torchvision==0.26.0" \
        --index-url https://download.pytorch.org/whl/cu130
}

# ── Extensions ───────────────────────────────────────────────────────────────
function provisioning_get_extensions() {
    mkdir -p "${NEO_DIR}/extensions"
    for entry in "${EXTENSIONS[@]}"; do
        repo="${entry%@*}"        # strip trailing @sha (GitHub https URLs have no other @)
        sha="${entry##*@}"        # the pinned commit
        dir="${repo##*/}"
        path="${NEO_DIR}/extensions/${dir}"
        if [[ ! -d $path ]]; then
            printf "Downloading extension: %s @ %s...\n" "${repo}" "${sha}"
            git clone "${repo}" "${path}" --recursive
        fi
        # Freeze to the pinned SHA + realign submodules to it. Never un-pin.
        git -C "$path" checkout --quiet "$sha" \
            && git -C "$path" submodule update --init --recursive --quiet \
            || echo "[provision] WARN: extension pin ${dir}@${sha} FAILED — running HEAD (unvalidated!)"
    done
}

# ── Model mirror (REAL dirs — Neo ignores symlinked dirs) ────────────────────
function provisioning_sync_models() {
    if [[ -z "$R2_BUCKET" || -z "$RCLONE_CONFIG_R2_ENDPOINT" ]]; then
        echo "[provision] FATAL: R2_BUCKET / RCLONE_CONFIG_R2_* env vars not set"
        exit 1
    fi
    if ! command -v rclone >/dev/null; then
        echo "[provision] installing rclone..."
        curl -fsSL https://rclone.org/install.sh | bash
    fi
    sync_pair "checkpoints"   "${NEO_DIR}/models/Stable-diffusion"
    sync_pair "lora"          "${NEO_DIR}/models/Lora"
    sync_pair "embeddings"    "${NEO_DIR}/models/embeddings"
    sync_pair "text_encoders" "${NEO_DIR}/models/text_encoder"
    sync_pair "vae"           "${NEO_DIR}/models/VAE"
    # Fully mirrored since Jul 20 2026: the upscaler + detector weights too, so a
    # provision succeeds with ONLY R2 up (the wget fallbacks below no-op via -nc).
    sync_pair "esrgan"        "${NEO_DIR}/models/ESRGAN"
    sync_pair "adetailer"     "${NEO_DIR}/models/adetailer"
    # ControlNet (Jul 25 2026): the control MODELS (models/ControlNet — Forge scans
    # this at startup, so it MUST be synced before launch.py) + the preprocessor
    # ANNOTATOR weights (models/ControlNetPreprocessor — depth_anything_v2 + DWPose
    # yolox/dw-ll, ~975MB) so depth/openpose work with NO HuggingFace dependency at
    # render time. Union model handles both depth+openpose on the SDXL checkpoints.
    sync_pair "controlnet"              "${NEO_DIR}/models/ControlNet"
    sync_pair "controlnet_preprocessor" "${NEO_DIR}/models/ControlNetPreprocessor"
}

function sync_pair() {
    local src="r2:${R2_BUCKET}/$1" dst="$2"
    mkdir -p "$dst"
    echo "[provision] syncing ${src} → ${dst}"
    rclone copy "$src" "$dst" \
        --transfers 4 --multi-thread-streams 8 --multi-thread-cutoff 64M \
        --retries 5 --low-level-retries 20 --stats-one-line --stats 15s -v
}

# ── Config: infotext OFF + deterministic couple faces (same as old fleet) ────
function provisioning_patch_forge_config() {
    python3 - "$NEO_DIR/config.json" <<'EOF'
import json, os, sys
p = sys.argv[1]
cfg = {}
if os.path.exists(p):
    try:
        cfg = json.load(open(p))
    except Exception:
        cfg = {}
cfg["enable_pnginfo"] = False
cfg["stealth_pnginfo"] = False
# ADetailer-Neo reads the SAME key + value strings via opts.data.get() —
# verified against its lib_adetailer/args.py BBOX_SORTBY on the golden box.
cfg["ad_bbox_sortby"] = "Position (left to right)"
json.dump(cfg, open(p, "w"), indent=2)
print(f"[provision] patched {p}")
EOF
}

# ── Warm boot: install torch+deps NOW (visible in provisioning log) ──────────
# First launch.py run downloads torch cu130 etc. Doing it here keeps the
# supervisor start fast and keeps failures loud. Killed once the API answers.
function provisioning_warm_boot() {
    cd "$NEO_DIR"
    echo "[provision] warm boot (installs torch + deps, then killed)..."
    PATH="$HOME/.local/bin:$PATH" setsid .venv/bin/python launch.py --api --port "$NEO_PORT" \
        > /tmp/neo-warmboot.log 2>&1 &
    local pid=$!
    for i in $(seq 1 240); do   # up to 20 min for torch download on slow pipes
        if curl -s -m 3 "127.0.0.1:${NEO_PORT}/sdapi/v1/sd-models" | grep -q model_name; then
            echo "[provision] warm boot API up after ~$((i*5))s — killing"
            kill -TERM -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null
            sleep 3
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[provision] FATAL: warm boot process died — /tmp/neo-warmboot.log tail:"
            tail -40 /tmp/neo-warmboot.log
            exit 1
        fi
        sleep 5
    done
    echo "[provision] FATAL: warm boot never became ready — /tmp/neo-warmboot.log tail:"
    tail -40 /tmp/neo-warmboot.log
    exit 1
}

# ── Supervisor service for Neo (base-image starts no webui by itself) ────────
function provisioning_install_forge_service() {
    cat > /opt/supervisor-scripts/forge-neo.sh <<NEOSH
#!/bin/bash
set -a; . /etc/environment 2>/dev/null; set +a
while [ -f "/.provisioning" ]; do
    echo "forge-neo startup paused (provisioning)..."
    sleep 5
done
# Fresh-sync ALL model folders from R2 on EVERY boot (not just provisioning):
# the owner drops new checkpoints/LoRAs/etc into the mirror and every worker
# picks them up at its next start — no manual per-worker syncs, no reprovision.
# rclone skips unchanged files, so a no-change boot costs only listings (~5s).
if command -v rclone >/dev/null && [ -n "\$R2_BUCKET" ]; then
    rclone copy "r2:\$R2_BUCKET/checkpoints"   "$NEO_DIR/models/Stable-diffusion" --transfers 4 -q || true
    rclone copy "r2:\$R2_BUCKET/lora"          "$NEO_DIR/models/Lora"             --transfers 4 -q || true
    rclone copy "r2:\$R2_BUCKET/embeddings"    "$NEO_DIR/models/embeddings"       --transfers 4 -q || true
    rclone copy "r2:\$R2_BUCKET/text_encoders" "$NEO_DIR/models/text_encoder"     --transfers 4 -q || true
    rclone copy "r2:\$R2_BUCKET/vae"           "$NEO_DIR/models/VAE"              --transfers 4 -q || true
    rclone copy "r2:\$R2_BUCKET/esrgan"        "$NEO_DIR/models/ESRGAN"           --transfers 4 -q || true
    rclone copy "r2:\$R2_BUCKET/adetailer"     "$NEO_DIR/models/adetailer"        --transfers 4 -q || true
    # ControlNet: control model (scanned at launch, so synced here BEFORE it) + preprocessor annotators
    rclone copy "r2:\$R2_BUCKET/controlnet"              "$NEO_DIR/models/ControlNet"              --transfers 4 -q || true
    rclone copy "r2:\$R2_BUCKET/controlnet_preprocessor" "$NEO_DIR/models/ControlNetPreprocessor" --transfers 4 -q || true
fi
cd "$NEO_DIR"
export PATH="\$HOME/.local/bin:\$PATH"
exec .venv/bin/python launch.py --api --port ${NEO_PORT}
NEOSH
    chmod +x /opt/supervisor-scripts/forge-neo.sh

    cat > /etc/supervisor/conf.d/forge-neo.conf <<'NEOC'
[program:forge-neo]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/forge-neo.sh
autostart=true
autorestart=true
startsecs=10
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=15
stdout_logfile=/var/log/portal/forge.log
redirect_stderr=true
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=1
NEOC
    supervisorctl reread && supervisorctl update
    echo "[provision] forge-neo supervisor service installed (port ${NEO_PORT} → /var/log/portal/forge.log)"
}

# ── Verify: a worker missing pieces must DIE here, loudly ────────────────────
function provisioning_verify() {
    local ok=1
    local ckpt_count
    ckpt_count=$(find "${NEO_DIR}/models/Stable-diffusion" -name '*.safetensors' 2>/dev/null | wc -l)
    echo "[provision] verify: ${ckpt_count} checkpoint(s)"
    [[ $ckpt_count -ge 2 ]] || { echo "[provision] FATAL: expected >=2 checkpoints"; ok=0; }
    [[ -f "${NEO_DIR}/models/text_encoder/qwen_3_06b_base.safetensors" ]] || { echo "[provision] FATAL: qwen text encoder missing"; ok=0; }
    [[ -f "${NEO_DIR}/models/VAE/qwen_image_vae.safetensors" ]] || { echo "[provision] FATAL: qwen VAE missing"; ok=0; }
    [[ -f "${NEO_DIR}/models/ESRGAN/RealESRGAN_x4plus_anime_6B.pth" ]] || { echo "[provision] FATAL: hires upscaler missing"; ok=0; }
    [[ -f "${NEO_DIR}/models/adetailer/face_yolov8n.pt" ]] || { echo "[provision] FATAL: adetailer face model missing"; ok=0; }
    [[ -d "${NEO_DIR}/extensions/ADetailer-Neo" ]] || { echo "[provision] FATAL: ADetailer-Neo missing"; ok=0; }
    [[ -d "${NEO_DIR}/extensions/sd-forge-couple" ]] || { echo "[provision] FATAL: sd-forge-couple missing"; ok=0; }
    [[ -d "${NEO_DIR}/tmp" ]] || { echo "[provision] FATAL: tmp dir missing"; ok=0; }
    .venv/bin/python -c "import ultralytics" 2>/dev/null || { echo "[provision] FATAL: ultralytics not importable"; ok=0; }
    if [[ $ok -eq 0 ]]; then
        echo "[provision] VERIFICATION FAILED — worker is not usable"
        exit 1
    fi
    echo "[provision] verification passed"
}

# ── Serverless pyworker (unchanged from old script, PYWORKER_REF=neo) ────────
function provisioning_install_pyworker() {
    if [[ -z "$PYWORKER_REPO" ]]; then
        echo "[provision] PYWORKER_REPO not set — skipping pyworker service"
        return 0
    fi
    # ALWAYS install OUR bootstrap, even when the image ships a pyworker service:
    # the stock one exits unless SERVERLESS=true is in the template env ("Skipping
    # pyworker startup (not serverless)") — fleet workers then never report to the
    # autoscaler and get destroyed at the loading deadline ("worker not found in
    # webserver response", seen live Jul 19 2026, workers 45317341/45317342).
    # Ours overwrites the same script path the stock conf runs, so whichever conf
    # exists ends up running OUR bootstrap.

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

function provisioning_get_files() {
    if [[ -z $2 ]]; then return 1; fi
    dir="$1"
    mkdir -p "$dir"
    shift
    arr=("$@")
    printf "Downloading %s file(s) to %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        printf "Downloading: %s\n" "${url}"
        provisioning_download "${url}" "${dir}"
        printf "\n"
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n#      Peepy NEO worker provisioning         #\n#         This will take some time           #\n##############################################\n\n"
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Application will start now\n\n"
}

function provisioning_download() {
    auth_token=""
    if [[ -n $HF_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
        auth_token="$HF_TOKEN"
    elif
        [[ -n $CIVITAI_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
        auth_token="$CIVITAI_TOKEN"
    fi
    if [[ -n $auth_token ]];then
        wget --header="Authorization: Bearer $auth_token" -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    else
        wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    fi
}

if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
