#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Peepy Vast.ai provisioning — HYBRID worker: video + photos on one GPU (Oct 2026).
# Branch `hybrid` only. Built from the two proven scripts, function for function:
#   • the photo stack of vast-provisioning-neo.sh (branch `neo`): Forge Neo + its
#     extensions + the R2 model mirror + the ComfyUI sidecar with its node packs;
#   • the video stack of vast-provisioning-video.sh (branch `video`): MiniMax H3 + LTX 2.3
#     (Sulphur 2) and their LoRAs, into ComfyUI's own models/ folders.
# ONE ComfyUI serves both: the video worker and the photo sidecar already run the same
# pin (b16023b0). Its venv rides the Forge venv's torch (cu130) like the photo sidecar,
# plus torchaudio for the clips' sound. worker.py on this branch keeps the GPU to one
# job at a time — videos first — and frees the other side's models before a switch.
# Disk: ~190 GB (photo models from R2 + ~92 GB of video weights).
#
# Template env: everything the neo template sets (the R2 rclone block, R2_BUCKET,
# FORGE_MODEL) with PYWORKER_REPO=https://github.com/thirdku/peepy-vast-worker and
# PYWORKER_REF=hybrid; optionally HF_TOKEN / CIVITAI_TOKEN (the Civitai-only video
# LoRAs come from the R2 mirror peepy-models/video/loras/).
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

# ── ComfyUI sidecar (Sep 22 2026) ────────────────────────────────────────────
# The anima worker's PRIMARY render backend now: every Anima-arch style renders
# through ComfyUI (peepy's /comfy/render remote-dispatch handler in worker.py),
# Forge keeps SDXL + the couple-mode fallthrough. Models are SHARED — ComfyUI
# reads Forge's model dirs via extra_model_paths.yaml, so the R2 sync above is
# the only model sync. Everything pinned to the SHAs proven live on worker
# 51615352 (Sep 22 2026); roll forward only by canarying, same rule as Neo.
COMFY_DIR=${WORKSPACE:-/workspace}/ComfyUI
COMFY_VENV=${WORKSPACE:-/workspace}/comfy-venv
COMFY_PORT=${COMFY_INTERNAL_PORT:-8188}
COMFY_PIN=b16023b004d3b1bfbbd6463414dc20da1b36cc4d    # ComfyUI v0.37.0-era, proven Sep 22 2026

# `repo@sha` (cloned under the repo's basename) or `repo@sha|dirname`.
COMFY_NODES=(
    # AnimaBoosterLoader + AnimaTeaCache (the site styles' loader + step cache)
    "https://github.com/BlackSnowSkill/ANIMA_BOOSTER@0cb1f45f4c6b8a726150faf9c08988bc10baf36c"
    # AnimaArtistPack/CrossAttn/Options — dir renamed to the Comfy-registry id
    "https://github.com/An1X3R/Anima-Artist-Mixer@0cc9ab7e8db906ee7cc200e771143c546e61f3af|anima-artist-mixer"
    # FaceDetailer (the ADetailer-parity face pass) + its Ultralytics detector
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack@429d0159ad429e64d2b3916e6e7be9c22d025c3c"
    "https://github.com/ltdrdata/ComfyUI-Impact-Subpack@50c7b71a6a224734cc9b21963c6d1926816a97f1"
    # ClownsharKSampler (HunkyMix v2) + registers beta57 into the core schedulers
    "https://github.com/ClownsharkBatwing/RES4LYF@056bc24a0450f5d46053535988cea4c719554806"
    # DepthAnythingV2 preprocessor (Perfect Copy's depth_anything_v2 module)
    "https://github.com/Fannovel16/comfyui_controlnet_aux@59b1fc411ede8623b2997855b8018f0b3b6cf49f"
    # kohya's Anima ControlNet-LLLite apply node (reads Forge's ControlNet dir)
    "https://github.com/kohya-ss/ComfyUI-Anima-LLLite@b7495bd8eb876e334509976896702484ed19cdbb"
    # Attention regional conditioning for Anima's DiT (couple mode) - the true
    # forge-couple analog (credits it); the archived laksjdjf attention-couple is UNet-only
    "https://github.com/Sen-sou/Comfyui-Anima-Regional-Conditioning@099cf1fa052721394963418455d49f7087efaf6c"
    # prompt-control: PCTextEncode COUPLE syntax + PCAnimaAttnCouplePatch
    # (pamparamm's Attention Couple ported to Anima) — the DEFAULT couple method
    "https://github.com/asagi4/comfyui-prompt-control@88af041dce376c97441d2efe5479e27de4b52cd8"
)

# ── video models (from vast-provisioning-video.sh) ──────────────────────────
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
    # The audio VAE: every clip with a Sound pick decodes its track with it (VAEDecodeAudio).
    # It was copied onto the first worker by hand (Oct 4 2026) and missing from this list, so
    # every replacement worker failed all clips with sound ("value_not_in_list") — Oct 8 2026.
    "vae|minimax_h3_audio_vae_fp32.safetensors|$H3_REPO/vae/minimax_h3_audio_vae_fp32.safetensors|605254808|"
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
#   Epic Cumshots (H3 ALPHA) + HMCumshot (v0.1) — the app adds both when a clip's action (or a
#   section of a split clip) is Cum (Oct 6 2026; clip-graph.ts MOTION_LORAS), same fallback
VIDEO_EXTRAS=(
    "loras|minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors|https://civitai.com/api/download/models/3297123?fileId=3181960|1956193000|C8168EBC17BBACC4296103DDA2FEC1BA85B24392FA08CF2BFBCEF0CFF0DC3CC8"
    "loras|NSFW_ANIME_V7_H3-step00019500.safetensors|https://civitai.com/api/download/models/3286171?fileId=3170500|596450480|C69A8E719B6784A8E475004CD47D34D1DDEFBB5DAA2D7670632CD3B459490B8D"
    "loras|H3_Motion_Booster_anime.safetensors|https://civitai.com/api/download/models/3299686?fileId=3184584|155110280|CF23F3F8AC3D663DD3EA49482F90DD4321EFF6A052AD6DFC7518321C01B78B9B"
    "loras|epic_cumshots-MiniMaxH3-ALPHA-CUMSH0T.safetensors|https://civitai.com/api/download/models/3202064?fileId=3083352|155110392|9D8D1C90AD8C875FD4FBEBCBB0D189E77E874C2C2A64077EA6D1937DEE27C100"
    "loras|HMCumshot_v1_e120.safetensors|https://civitai.com/api/download/models/3227331?fileId=3109563|626294968|87376EAB88511C76DE00BF591D3565ADE1AAE4E5C9D6811A85DB0668467C6A71"
    # ── LTX 2.3 (Sulphur 2), the app's second engine (admin-only trial, Oct 7 2026; clip-graph.ts
    # buildLtxClipGraph). Hand-copied onto the first worker, so the Oct 8 replacement had none of
    # it and every LTX clip failed. Optional: a worker without them still serves H3; LTX clips
    # fail + refund until they land. ~45 GB. Hugging Face files pinned by revision + sha256.
    "checkpoints|sulphur_dev_fp8mixed.safetensors|$HF/SulphurAI/Sulphur-2-base/resolve/65587bbb38c700622f9434a16e732c8b280c115e/sulphur_dev_fp8mixed.safetensors|29161842846|41c999575859c528ff108022246a5524960a778c18742696971c9b0aadb4f70f"
    "text_encoders|gemma_3_12B_it_fp4_mixed.safetensors|$HF/Comfy-Org/ltx-2/resolve/ccde4ba417d7900669fd56dd292a883cee11ff37/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors|9447702218|aaca463d11e6d8d2a4bdb0d6299214c15ef78a3f73e0ef8113d5a9d0219b3f6d"
    "loras|ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors|$HF/SulphurAI/Sulphur-2-base/resolve/65587bbb38c700622f9434a16e732c8b280c115e/distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors|662072824|e970f64a2ce5469491fb1714a3fa72c8b606fa82affff0531e836dbc91d31f34"
    "latent_upscale_models|ltx-2.3-spatial-upscaler-x2-1.1.safetensors|$HF/Lightricks/LTX-2.3/resolve/3c6a4e66e5d0a684231950b9c74dd4ded7b6fadc/ltx-2.3-spatial-upscaler-x2-1.1.safetensors|995743560|5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed"
    # 2D NSFW Motion Enhancer (1.0, every explicit LTX clip) and creampie (not wired yet) — Hugging
    # Face mirrors of the Civitai files (same sha256), saved under the names the app's graph uses
    "loras|LTX23_2d_nsfw_motion_enhancer.safetensors|$HF/Muapi/ltx-2.3-2d-nsfw-motion-enhancer/resolve/17dd5bf5845ced4a24da52bfcda39e226f5462ae/ltx-2.3-2d-nsfw-motion-enhancer.safetensors|1348027400|a7a13202745900078d7b53edf8975c93fcb1f7815f53f97ae338dea077173cfc"
    "loras|LTX23_creampie_cum_animation_early.safetensors|$HF/Muapi/ltx-2.3-i2v-creampie-cum-animation-early/resolve/ffdd7ade77038e209457e8b6b9ecddf7864d5446/ltx-2.3-i2v-creampie-cum-animation-early.safetensors|674249632|33f611042ec5529d4042693653f53aa7e286517697677866f94d1858848110db"
    # Civitai-only (sign-in): Better NSFW motion v2 (0.7) · VBVR reasoning v4 (0.5) · DR34ML4Y v2
    # (0.3) — mirrored in R2 peepy-models/video/loras/ (Oct 9 2026), else CIVITAI_TOKEN
    "loras|Sulphur_LTX 2.3_better _NSFW_motion.safetensors|https://civitai.com/api/download/models/2986751?fileId=2866267|654446000|A0ED4CD01CF0AC88CF324A4FC432D02ABE18F0A0CB9DFB45F04A26A08AA31987"
    "loras|LTX2.3_reasoning_Sulphur-2_I2V_V4.safetensors|https://civitai.com/api/download/models/3025398?fileId=2904187|805412760|1F7C87052D44087E17630B7075D8DFD8205C17CE5E7D3BCBEAE11AF2102C88DB"
    "loras|DR34ML4Y_LTXXX_V2.safetensors|https://civitai.com/api/download/models/2950842?fileId=2830123|1940553800|D91D2E783879601BC7A638C978F8257E5ED340C012E7D668B7CD32DE443FF2E1"
)

function provisioning_start() {
    provisioning_print_header
    # The pyworker service goes in FIRST (the video script's order): it waits while
    # /.provisioning exists, and a failed provision leaves /.provisioning_failed for
    # worker.py to report instead of a silent worker.
    provisioning_install_pyworker
    # ── the photo stack (neo's order) ──
    provisioning_install_neo
    provisioning_get_extensions
    provisioning_sync_models
    provisioning_get_files "${NEO_DIR}/models/adetailer" "${ADETAILER_MODELS[@]}"
    provisioning_get_files "${NEO_DIR}/models/ESRGAN"    "${ESRGAN_MODELS[@]}"
    provisioning_get_files "${NEO_DIR}/models/text_encoder"     "${ANIMA_TE_MODELS[@]}"
    provisioning_get_files "${NEO_DIR}/models/VAE"              "${ANIMA_VAE_MODELS[@]}"
    provisioning_get_files "${NEO_DIR}/models/Stable-diffusion" "${ANIMA_CKPT_MODELS[@]}"
    provisioning_warm_boot
    provisioning_patch_forge_config

    export GIT_CONFIG_GLOBAL=/tmp/temporary-git-config
    git config --file $GIT_CONFIG_GLOBAL --add safe.directory '*'

    provisioning_install_forge_service
    provisioning_install_model_sync
    provisioning_install_comfyui
    # ── the video stack ──
    provisioning_get_video_models
    provisioning_install_comfy_service
    provisioning_verify_image
    provisioning_verify_video
    provisioning_print_end
}

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
    sync_pair "lora"          "${NEO_DIR}/models/Lora"            mirror
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
    local src="r2:${R2_BUCKET}/$1" dst="$2" mode="${3:-copy}"
    mkdir -p "$dst"
    # mode=mirror → rclone sync: adds/updates AND deletes local files no longer in
    # R2 (a true mirror). --max-delete 50 fuses a transient/empty R2 listing so it
    # can never wipe the whole folder. Default copy mode never deletes — keep it for
    # dirs that also hold non-R2 files (controlnet_preprocessor's HF annotators).
    local verb=copy; local extra=()
    if [ "$mode" = mirror ]; then verb=sync; extra=(--max-delete 50); fi
    echo "[provision] ${verb}ing ${src} → ${dst}"
    rclone "$verb" "$src" "$dst" \
        --transfers 4 --multi-thread-streams 8 --multi-thread-cutoff 64M \
        --retries 5 --low-level-retries 20 --stats-one-line --stats 15s -v "${extra[@]}"
}

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
    rclone sync "r2:\$R2_BUCKET/lora"          "$NEO_DIR/models/Lora"             --transfers 4 --max-delete 50 -q || true
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

function provisioning_install_model_sync() {
    cat > /opt/supervisor-scripts/model-sync.sh <<'MSYNC'
#!/bin/bash
set -a; . /etc/environment 2>/dev/null; set +a
NEO_DIR="${WORKSPACE:-/workspace}/sd-webui-forge-neo"
PORT="${FORGE_INTERNAL_PORT:-17860}"
INTERVAL_MIN="${MODEL_SYNC_INTERVAL_MIN:-0}"
if ! [[ "$INTERVAL_MIN" =~ ^[0-9]+$ ]] || [ "$INTERVAL_MIN" -le 0 ]; then
    echo "[model-sync] disabled (MODEL_SYNC_INTERVAL_MIN='${INTERVAL_MIN}') — idling"
    exec sleep infinity
fi
if ! command -v rclone >/dev/null || [ -z "$R2_BUCKET" ]; then
    echo "[model-sync] rclone or R2_BUCKET missing — idling"; exec sleep infinity
fi
sync_one() {  # $1 r2 subdir, $2 local dir, $3 mode(copy|mirror) -> 'changed' when files moved
    local out mode="${3:-copy}" verb=copy extra=()
    if [ "$mode" = mirror ]; then verb=sync; extra=(--max-delete 50); fi
    out="$(rclone "$verb" "r2:$R2_BUCKET/$1" "$2" --transfers 4 "${extra[@]}" --log-level INFO 2>&1)"
    echo "$out" | grep -qE 'Copied|Updated|Deleted' && echo changed
}
while [ -f /.provisioning ]; do sleep 5; done
echo "[model-sync] active — every ${INTERVAL_MIN}min from r2:${R2_BUCKET} (Forge :${PORT})"
while true; do
    sleep $((INTERVAL_MIN*60))
    ck=""; lo=""
    [ -n "$(sync_one checkpoints "$NEO_DIR/models/Stable-diffusion")" ] && ck=1
    [ -n "$(sync_one lora        "$NEO_DIR/models/Lora" mirror)" ]     && lo=1
    sync_one embeddings    "$NEO_DIR/models/embeddings"   >/dev/null
    sync_one text_encoders "$NEO_DIR/models/text_encoder" >/dev/null
    sync_one vae           "$NEO_DIR/models/VAE"          >/dev/null
    sync_one esrgan        "$NEO_DIR/models/ESRGAN"       >/dev/null
    sync_one adetailer     "$NEO_DIR/models/adetailer"    >/dev/null
    if [ -n "$ck" ]; then
        echo "[model-sync] checkpoint change -> refresh-checkpoints"
        curl -s -m 30 -X POST "http://127.0.0.1:${PORT}/sdapi/v1/refresh-checkpoints" -o /dev/null || true
    fi
    if [ -n "$lo" ]; then
        echo "[model-sync] LoRA change -> refresh-loras"
        curl -s -m 30 -X POST "http://127.0.0.1:${PORT}/sdapi/v1/refresh-loras" -o /dev/null || true
    fi
done
MSYNC
    chmod +x /opt/supervisor-scripts/model-sync.sh
    cat > /etc/supervisor/conf.d/model-sync.conf <<'MCONF'
[program:model-sync]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/model-sync.sh
autostart=true
autorestart=true
startsecs=5
stdout_logfile=/var/log/portal/model-sync.log
redirect_stderr=true
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=1
MCONF
    supervisorctl reread && supervisorctl update
    echo "[provision] model-sync supervisor service installed (gated on MODEL_SYNC_INTERVAL_MIN)"
}

function provisioning_install_comfyui() {
    if [[ ! -d "$COMFY_DIR" ]]; then
        git clone https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
    fi
    git -C "$COMFY_DIR" checkout --quiet "$COMFY_PIN" \
        || echo "[provision] WARN: ComfyUI pin checkout FAILED — running HEAD (unvalidated!)"

    # Own venv layered on the Forge venv's python (--system-site-packages): pip
    # resolves ComfyUI's deps venv-locally (incl. its own torch cu130) and the
    # Forge env is NEVER written to. torch/vision/audio filtered from the
    # requirements install so pip can't be tempted to touch the base pins.
    if [[ ! -d "$COMFY_VENV" ]]; then
        "$NEO_DIR/.venv/bin/python" -m venv --system-site-packages "$COMFY_VENV"
    fi
    grep -viE '^(torch|torchvision|torchaudio)([=<>~ ]|$)' "$COMFY_DIR/requirements.txt" > /tmp/comfy-req.txt
    "$COMFY_VENV/bin/pip" install -q -r /tmp/comfy-req.txt
    # hybrid: torchaudio for the video graphs' audio nodes (H3 VAEDecodeAudio, LTX audio latent);
    # torch itself comes from the Forge venv (system site packages), so this pulls no new torch
    "$COMFY_VENV/bin/pip" install -q "torchaudio==2.11.0" --index-url https://download.pytorch.org/whl/cu130 \
        || echo "[provision] WARN: torchaudio not installed (clips with sound will fail)"

    mkdir -p "$COMFY_DIR/custom_nodes"
    local entry spec dir repo sha path
    for entry in "${COMFY_NODES[@]}"; do
        spec="${entry%%|*}"
        repo="${spec%@*}"
        sha="${spec##*@}"
        if [[ "$entry" == *"|"* ]]; then dir="${entry##*|}"; else dir="${repo##*/}"; fi
        path="$COMFY_DIR/custom_nodes/$dir"
        if [[ ! -d "$path" ]]; then
            printf "Downloading ComfyUI node pack: %s @ %s...\n" "$repo" "$sha"
            git clone "$repo" "$path"
        fi
        git -C "$path" checkout --quiet "$sha" \
            || echo "[provision] WARN: node pin ${dir}@${sha} FAILED — running HEAD (unvalidated!)"
        if [[ -f "$path/requirements.txt" ]]; then
            "$COMFY_VENV/bin/pip" install -q -r "$path/requirements.txt" || true
        fi
    done
    "$COMFY_VENV/bin/pip" cache purge >/dev/null 2>&1 || true

    # comfyui-prompt-control fixup: upstream's Anima Attention Couple predates
    # ComfyUI's Jul 17 2026 Cosmos attn2_patch calling convention (pe= kwarg +
    # dict return) — the SDXL-style kv-inject patches crash if they run inside
    # the couple wrapper, and the wrapper already does the whole coupling, so
    # they are stripped there. Verified live Sep 23 2026 (workers c3686b7f/
    # d990221f); drop this once upstream moves past 88af041d with its own fix.
    local pcfile="$COMFY_DIR/custom_nodes/comfyui-prompt-control/prompt_control/anima_couple.py"
    if [[ -f "$pcfile" ]] && ! grep -q "peepy patch" "$pcfile"; then
        python3 - "$pcfile" <<'PCPATCH' || echo "[provision] WARN: prompt-control anima patch FAILED (couple renders will error)"
import sys
p = sys.argv[1]
src = open(p).read()
old = "    out = _forward(xs, cs, rope_emb, transformer_options)"
new = (
    "    # peepy patch: strip SDXL-style attn patches inside the couple wrapper\n"
    "    to_pc = dict(transformer_options)\n"
    '    to_pc["patches"] = {k_: v_ for k_, v_ in to_pc.get("patches", {}).items()\n'
    '                       if k_ not in ("attn2_patch", "attn2_output_patch")}\n'
    "    out = _forward(xs, cs, rope_emb, to_pc)"
)
assert src.count(old) == 1, "anima_couple.py target line not unique"
open(p, "w").write(src.replace(old, new))
print("[provision] comfyui-prompt-control anima_couple patched")
PCPATCH
    fi

    # Model sharing — ComfyUI reads Forge's model dirs directly: one copy on
    # disk, one R2 sync (forge-neo.sh's boot sync feeds both backends).
    # ultralytics_bbox → models/adetailer gives FaceDetailer the SAME
    # face_yolov8n.pt weights Forge's ADetailer uses.
    cat > "$COMFY_DIR/extra_model_paths.yaml" <<EOF
forge:
  base_path: ${NEO_DIR}/models
  checkpoints: Stable-diffusion
  diffusion_models: Stable-diffusion
  unet: Stable-diffusion
  loras: Lora
  vae: VAE
  text_encoders: text_encoder
  clip: text_encoder
  embeddings: embeddings
  upscale_models: ESRGAN
  controlnet: ControlNet
  ultralytics_bbox: adetailer
  ultralytics: adetailer
EOF

    # Depth-Anything vitb prefetch (Perfect Copy's preprocessor) — best-effort;
    # comfyui_controlnet_aux auto-downloads it on first use if this fails.
    local da_dir="$COMFY_DIR/custom_nodes/comfyui_controlnet_aux/ckpts/depth-anything/Depth-Anything-V2-Base"
    mkdir -p "$da_dir"
    if [[ ! -f "$da_dir/depth_anything_v2_vitb.pth" ]]; then
        wget -q -O "$da_dir/depth_anything_v2_vitb.pth" \
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth" \
            || rm -f "$da_dir/depth_anything_v2_vitb.pth"
    fi
    # Lineart annotator prefetch (/create's upload hero): netG.pth = AnimeLineArtPreprocessor (the
    # default, Sep 30 2026), sk_model / sk_model2 = LineArtPreprocessor (the "Realistic" retry, Oct 1 2026;
    # it loads both at init). Best-effort too: without them the first hero render on a worker spends
    # ~30 s downloading.
    local la_dir="$COMFY_DIR/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators"
    mkdir -p "$la_dir"
    local la_f
    for la_f in netG.pth sk_model.pth sk_model2.pth; do
        if [[ ! -f "$la_dir/$la_f" ]]; then
            wget -q -O "$la_dir/$la_f" \
                "https://huggingface.co/lllyasviel/Annotators/resolve/main/$la_f" \
                || rm -f "$la_dir/$la_f"
        fi
    done
    echo "[provision] ComfyUI installed at ${COMFY_DIR} (pin ${COMFY_PIN:0:8}, $(ls "$COMFY_DIR/custom_nodes" | wc -l) node dirs)"
}

function provisioning_install_comfy_service() {
    cat > /etc/supervisor/conf.d/comfyui.conf <<COMFYC
[program:comfyui]
environment=PROC_NAME="%(program_name)s"
command=${COMFY_VENV}/bin/python ${COMFY_DIR}/main.py --listen 127.0.0.1 --port ${COMFY_PORT} --disable-auto-launch --disable-metadata
directory=${COMFY_DIR}
autostart=true
autorestart=true
startsecs=10
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

function file_ok() {
    local f="$1" size="$2" sha="$3"
    [[ -f "$f" ]] && (( $(stat -c %s "$f") == size )) || return 1
    [[ -z "$sha" ]] && return 0
    [[ "$(sha256sum "$f" | cut -d' ' -f1)" == "${sha,,}" ]]
}

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

function provisioning_get_video_models() {
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

function provisioning_verify_image() {
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
    "${NEO_DIR}/.venv/bin/python" -c "import ultralytics" 2>/dev/null || { echo "[provision] FATAL: ultralytics not importable"; ok=0; }
    # ComfyUI sidecar (the anima styles' primary backend — a worker without it
    # would black-hole every anima render into /comfy/render errors)
    [[ -f "${COMFY_DIR}/main.py" ]] || { echo "[provision] FATAL: ComfyUI missing"; ok=0; }
    "${COMFY_VENV}/bin/python" -c "import torch, aiohttp" 2>/dev/null || { echo "[provision] FATAL: comfy venv broken"; ok=0; }
    local nd
    for nd in ANIMA_BOOSTER anima-artist-mixer ComfyUI-Impact-Pack ComfyUI-Impact-Subpack RES4LYF comfyui_controlnet_aux ComfyUI-Anima-LLLite Comfyui-Anima-Regional-Conditioning comfyui-prompt-control; do
        [[ -d "${COMFY_DIR}/custom_nodes/${nd}" ]] || { echo "[provision] FATAL: ComfyUI node pack ${nd} missing"; ok=0; }
    done
    if [[ $ok -eq 0 ]]; then
        echo "[provision] VERIFICATION FAILED — worker is not usable"
        exit 1
    fi
    echo "[provision] verification passed"
}

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

function loader_for() {
    case "$1" in
        diffusion_models) echo "UNETLoader unet_name" ;;
        text_encoders)    echo "CLIPLoader clip_name" ;;
        vae)              echo "VAELoader vae_name" ;;
        loras)            echo "LoraLoaderModelOnly lora_name" ;;
    esac
}

function provisioning_verify_video() {
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
    printf "
##############################################
#   Peepy HYBRID worker provisioning (video + photos)   #
##############################################

"
}

function provisioning_print_end() {
    printf "
Provisioning complete:  Application will start now

"
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
