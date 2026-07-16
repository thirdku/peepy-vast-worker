#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Peepy Vast.ai provisioning script — see VAST_SERVERLESS_PLAN.md §5.2.
# Adapted from the official vastai/sd-forge template PROVISIONING_SCRIPT.
#
# Host at a raw URL and set as the template's PROVISIONING_SCRIPT env var.
# Runs on FIRST BOOT only (disk persists across stop/start, so resumed
# workers skip it). Idempotent: safe to re-run — rclone only transfers
# missing/changed files, git clones are guarded, config patch is a merge.
#
# Model delivery = rclone MIRROR of the R2 bucket (no per-file URL lists):
# upload from the golden box with `rclone copy`, workers pull the same
# folders back. Adding a LoRA later = upload to R2, done — no script edit.
#
# Template env vars REQUIRED (rclone reads its config straight from env;
# use a READ-ONLY R2 API token scoped to this bucket):
#   RCLONE_CONFIG_R2_TYPE=s3
#   RCLONE_CONFIG_R2_PROVIDER=Cloudflare
#   RCLONE_CONFIG_R2_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
#   RCLONE_CONFIG_R2_ACCESS_KEY_ID=<read-only key>
#   RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=<read-only secret>
#   R2_BUCKET=peepy-models
#
# Bucket layout (created by the golden-box upload, §"where to upload"):
#   peepy-models/checkpoints/  → models/Stable-diffusion/
#   peepy-models/lora/         → models/Lora/
#   peepy-models/embeddings/   → embeddings/
#
# Peepy-specific steps beyond model sync:
#   • ADetailer extension + face_yolov8n.pt pre-fetch (first render never stalls)
#   • config.json patched with enable_pnginfo=false — the raw Forge PNG is the
#     premium download master; infotext ON would leak the hidden recipe
#     (belt-and-suspenders: buildForgePayload also overrides it per-request)
#   • hard verification at the end — a worker with missing models must DIE
#     loudly in the provisioning log, not come up and render garbage
# ─────────────────────────────────────────────────────────────────────────────

source /venv/main/bin/activate
FORGE_DIR=${WORKSPACE}/stable-diffusion-webui-forge

APT_PACKAGES=(
)

PIP_PACKAGES=(
    # ADetailer's main dependency, pre-installed so Forge's first real start
    # doesn't spend minutes in the extension installer (the old warm-run
    # approach re-booted Forge just for this — too slow against the
    # autoscaler's ~31-min loading timeout).
    "ultralytics"
)

EXTENSIONS=(
    "https://github.com/Bing-su/adetailer"
    # Forge Couple — regional attention for two-person shots (couple mode in
    # buildForgePayload). An unknown alwayson_scripts key 500s a Forge without
    # the extension, so every worker MUST have it (server guard: FORGE_COUPLE=0).
    "https://github.com/Haoming02/sd-forge-couple"
)

# One-off extras that aren't in the R2 mirror (public URLs, wget'd as-is).
# ADetailer looks for detection models in ${FORGE_DIR}/models/adetailer.
ADETAILER_MODELS=(
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

function provisioning_start() {
    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_get_extensions
    provisioning_get_pip_packages
    provisioning_sync_models
    provisioning_get_files \
        "${FORGE_DIR}/models/adetailer" \
        "${ADETAILER_MODELS[@]}"

    provisioning_patch_forge_config

    # Avoid git errors because we run as root but files are owned by 'user'
    export GIT_CONFIG_GLOBAL=/tmp/temporary-git-config
    git config --file $GIT_CONFIG_GLOBAL --add safe.directory '*'

    # NO warm run (removed Jul 11): the serverless boot chain is strictly
    # serial (docker pull → this script → Forge start → pyworker → benchmark)
    # against a ~31-min autoscaler loading timeout — a redundant full Forge
    # boot here cost minutes. ultralytics is pre-installed via PIP_PACKAGES
    # instead; ADetailer's installer finds it satisfied at first real start.

    provisioning_verify
    provisioning_install_pyworker
    provisioning_print_end
}

# ── Serverless pyworker fallback ─────────────────────────────────────────────
# CONFIRMED needed (Jul 11): the sd-forge image build c055f2d predates the
# base-image serverless support — no pyworker.conf, SERVERLESS never set, so
# endpoint workers provision fine but never report ready and time out after
# ~31 min. This installs the official bootstrap as a supervisor service,
# mirroring ROOT/opt/supervisor-scripts/pyworker.sh from current vast-ai/
# base-image. Gated on PYWORKER_REPO so a manually-created instance from the
# same template without it stays a plain Forge box. No-ops on newer images
# that already ship the service.
function provisioning_install_pyworker() {
    if [[ -z "$PYWORKER_REPO" ]]; then
        echo "[provision] PYWORKER_REPO not set — skipping pyworker service"
        return 0
    fi
    if [[ -f /etc/supervisor/conf.d/pyworker.conf ]]; then
        echo "[provision] image already ships a pyworker service — skipping"
        return 0
    fi

    cat > /opt/supervisor-scripts/pyworker.sh <<'PYW'
#!/bin/bash
set -a; . /etc/environment 2>/dev/null; set +a
# Exit 0 on TERM/INT: during instance STOP supervisor kills worker.py, and
# without this the wrapper respawned mid-shutdown, rm -rf'd the clone, and
# the bootstrap reported "Failed to cd into SERVER_DIR" to the autoscaler —
# which then destroyed a perfectly good stopped worker (seen live Jul 11).
trap 'exit 0' TERM INT
# Wait for provisioning to finish (mirrors the official base-image wrapper)
while [ -f "/.provisioning" ]; do
    echo "pyworker startup paused (provisioning)..."
    sleep 5
done
# Wait for outbound network: a cold-start RESUME launches us seconds after
# boot, before DNS is up — git clone failed instantly and error-reported the
# worker to death (also seen live Jul 11).
for i in $(seq 1 60); do
    getent hosts github.com >/dev/null 2>&1 && break
    echo "pyworker waiting for network ($i)..."
    sleep 2
done
# The official bootstrap needs uv; older images don't ship it
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
# Clone ONCE and keep it (official-image behavior). Re-cloning every boot made
# resumes fragile: one throttled/early GitHub clone and start_server.sh reported
# "Failed to cd into SERVER_DIR" — and the autoscaler destroyed a good worker.
# start_server.sh's own clone fails harmlessly when the dir already exists.
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
    echo "[provision] pyworker supervisor service installed"
}

# Mirror the model stack from R2. rclone is configured entirely from the
# RCLONE_CONFIG_R2_* template env vars — no config file, no interactive setup.
function provisioning_sync_models() {
    if [[ -z "$R2_BUCKET" || -z "$RCLONE_CONFIG_R2_ENDPOINT" ]]; then
        echo "[provision] FATAL: R2_BUCKET / RCLONE_CONFIG_R2_* env vars not set"
        exit 1
    fi
    if ! command -v rclone >/dev/null; then
        echo "[provision] installing rclone..."
        curl -fsSL https://rclone.org/install.sh | bash
    fi
    sync_pair "checkpoints" "${FORGE_DIR}/models/Stable-diffusion"
    sync_pair "lora"        "${FORGE_DIR}/models/Lora"
    sync_pair "embeddings"  "${FORGE_DIR}/embeddings"
}

function sync_pair() {
    local src="r2:${R2_BUCKET}/$1" dst="$2"
    mkdir -p "$dst"
    echo "[provision] syncing ${src} → ${dst}"
    rclone copy "$src" "$dst" \
        --transfers 4 --multi-thread-streams 8 --multi-thread-cutoff 64M \
        --retries 5 --low-level-retries 20 --stats-one-line --stats 15s -v
}

# Force infotext OFF (merge-patch, creates config.json if absent).
# See VAST_SERVERLESS_PLAN.md §5.2a — verify the key name against the golden
# box's config.json ("Write infotext to metadata of the generated image").
function provisioning_patch_forge_config() {
    python3 - "$FORGE_DIR/config.json" <<'EOF'
import json, os, sys
p = sys.argv[1]
cfg = {}
if os.path.exists(p):
    try:
        cfg = json.load(open(p))
    except Exception:
        cfg = {}
cfg["enable_pnginfo"] = False
# Forge's built-in STEALTH pnginfo hides gen info in the alpha channel even
# when enable_pnginfo is off — kill it too (golden box has the option key
# present, so the feature exists in this build).
cfg["stealth_pnginfo"] = False
# ADetailer multi-face determinism: default bbox sort is "None" (arbitrary model
# output order) — couple mode's [SEP] per-face prompts NEED left-to-right so the
# segment order matches the Forge Couple region order. buildForgePayload also
# sends this per-request via override_settings; this is the config-level default.
cfg["ad_bbox_sortby"] = "Position (left to right)"
json.dump(cfg, open(p, "w"), indent=2)
print(f"[provision] patched {p}: enable_pnginfo=false stealth_pnginfo=false")
EOF
}

# A worker missing models must fail HERE, visibly, not benchmark garbage.
function provisioning_verify() {
    local ok=1
    local ckpt_count lora_count emb_count
    ckpt_count=$(find "${FORGE_DIR}/models/Stable-diffusion" -name '*.safetensors' 2>/dev/null | wc -l)
    lora_count=$(find "${FORGE_DIR}/models/Lora" \( -name '*.safetensors' -o -name '*.pt' \) 2>/dev/null | wc -l)
    emb_count=$(find "${FORGE_DIR}/embeddings" \( -name '*.safetensors' -o -name '*.pt' \) 2>/dev/null | wc -l)
    echo "[provision] verify: ${ckpt_count} checkpoint(s), ${lora_count} lora(s), ${emb_count} embedding(s)"
    [[ $ckpt_count -ge 2 ]] || { echo "[provision] FATAL: expected >=2 checkpoints"; ok=0; }
    [[ $lora_count -ge 3 ]] || { echo "[provision] FATAL: expected >=3 loras"; ok=0; }
    [[ $emb_count  -ge 2 ]] || { echo "[provision] FATAL: expected >=2 embeddings"; ok=0; }
    [[ -f "${FORGE_DIR}/models/adetailer/face_yolov8n.pt" ]] || { echo "[provision] FATAL: adetailer face model missing"; ok=0; }
    [[ -d "${FORGE_DIR}/extensions/adetailer" ]] || { echo "[provision] FATAL: adetailer extension missing"; ok=0; }
    [[ -d "${FORGE_DIR}/extensions/sd-forge-couple" ]] || { echo "[provision] FATAL: sd-forge-couple extension missing"; ok=0; }
    if [[ $ok -eq 0 ]]; then
        echo "[provision] VERIFICATION FAILED — worker is not usable"
        exit 1
    fi
    echo "[provision] verification passed"
}

function provisioning_get_apt_packages() {
    if [[ -n $APT_PACKAGES ]]; then
            sudo $APT_INSTALL ${APT_PACKAGES[@]}
    fi
}

function provisioning_get_pip_packages() {
    if [[ -n $PIP_PACKAGES ]]; then
            pip install --no-cache-dir ${PIP_PACKAGES[@]}
    fi
}

function provisioning_get_extensions() {
    for repo in "${EXTENSIONS[@]}"; do
        dir="${repo##*/}"
        path="${FORGE_DIR}/extensions/${dir}"
        if [[ ! -d $path ]]; then
            printf "Downloading extension: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
        fi
    done
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
    printf "\n##############################################\n#          Peepy worker provisioning         #\n#         This will take some time           #\n##############################################\n\n"
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Application will start now\n\n"
}

# Download from $1 URL to $2 dir (HF_TOKEN / CIVITAI_TOKEN honored per-URL).
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

# Allow user to disable provisioning if they started with a script they didn't want
if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
