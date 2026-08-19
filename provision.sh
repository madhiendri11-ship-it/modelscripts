#!/bin/bash
# ============================================================
#  provision.sh — ComfyUI model + node provisioner for RunPod
# ============================================================
#  - Auto-detects ComfyUI wherever the template installed it
#  - FORCES models onto /workspace (network volume) via symlink
#    so nothing is lost when the pod stops
#  - Uses HuggingFace hf_transfer for faster downloads
#  - Resumes partial downloads, skips completed ones
#  - Self-heals required custom nodes (Impact-Pack, Impact-Subpack)
#    every boot, in case the template rebuilds custom_nodes/ fresh
#
# ------------------------------------------------------------
#  QUICK START
# ------------------------------------------------------------
#  Manual (run this first to confirm it works):
#      bash /workspace/provision.sh
#
#  Auto on every pod boot -- RunPod Template -> Container Start Command:
#      bash -c "curl -sL YOUR_RAW_GITHUB_URL -o /workspace/provision.sh && \
#               bash /workspace/provision.sh > /workspace/provision.log 2>&1 & \
#               exec /start.sh"
#
#  Watch progress:
#      tail -f /workspace/provision.log
# ------------------------------------------------------------
#  WHICH PIPELINE TO DOWNLOAD (set as RunPod env vars)
# ------------------------------------------------------------
#  ZIMAGE=true    L0urta pipeline (endri.json, two-stage)
#                 z_image_bf16 + z_image_turbo_bf16 + qwen_3_4b + ae
#                 + RealESRGAN_x4plus + face_yolov8m
#                 Requires: ComfyUI-Impact-Pack + ComfyUI-Impact-Subpack
#                 (installed automatically below)
#
#  KREA2=true     Mara pipeline (Krea 2)
#                 krea2_turbo_fp8_scaled + qwen3vl_4b_fp8_scaled + wan_2.1_vae
#                 NOTE: if your RunPod template already provisions Krea2
#                 models itself (check the boot log for "Provisioning
#                 models HF"), leave this false to avoid downloading
#                 the same ~17GB twice.
#
#  Both default to what's set below. Override per-pod in the
#  RunPod template's Environment Variables, e.g.:
#      ZIMAGE=true
#      KREA2=false
# ------------------------------------------------------------
#  HF_TOKEN   HuggingFace token - recommended, speeds up downloads
#             and avoids rate limits. Get one at
#             huggingface.co/settings/tokens (a "read" token is enough)
#             Set as a RunPod env var, do NOT hardcode it here.
# ------------------------------------------------------------

ZIMAGE="${ZIMAGE:-true}"
KREA2="${KREA2:-false}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
HF_TOKEN="${HF_TOKEN:-}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[..]${NC} $1"; }
err()  { echo -e "${RED}[!!]${NC} $1"; }
hdr()  { echo -e "\n${CYAN}--- $1 ---${NC}"; }

START_TS=$(date +%s)

echo ""
echo "  ==========================================="
echo "   ComfyUI Provisioner"
echo "   ZIMAGE=$ZIMAGE   KREA2=$KREA2"
echo "  ==========================================="

# ------------------------------------------------------------
# 1. Locate ComfyUI
# ------------------------------------------------------------
hdr "Locating ComfyUI"

COMFY="${COMFY:-}"
if [ -z "$COMFY" ]; then
    for c in /workspace/ComfyUI /ComfyUI /opt/ComfyUI /root/ComfyUI \
             /workspace/comfyui /comfyui ; do
        [ -f "$c/main.py" ] && COMFY="$c" && break
    done
fi
if [ -z "$COMFY" ]; then
    F=$(find /workspace / -maxdepth 4 -name main.py -path "*omfy*" 2>/dev/null | head -1)
    [ -n "$F" ] && COMFY=$(dirname "$F")
fi
if [ -z "$COMFY" ]; then
    err "ComfyUI not found anywhere."
    err "Run with the path explicitly:  COMFY=/path/to/ComfyUI bash \$0"
    exit 1
fi
ok "ComfyUI: $COMFY"

# ------------------------------------------------------------
# 2. Force models onto the persistent volume
# ------------------------------------------------------------
hdr "Persistent storage"

VOLUME_MODELS="/workspace/models"

if [ -L "$COMFY/models" ]; then
    ok "models/ already symlinked -> $(readlink "$COMFY/models")"
elif [[ "$COMFY" == /workspace/* ]]; then
    ok "ComfyUI already on the volume"
    VOLUME_MODELS="$COMFY/models"
else
    warn "ComfyUI is on the container disk - relinking models to the volume"
    mkdir -p "$VOLUME_MODELS"
    if [ -d "$COMFY/models" ]; then
        cp -rn "$COMFY/models/." "$VOLUME_MODELS/" 2>/dev/null
        rm -rf "$COMFY/models"
    fi
    ln -s "$VOLUME_MODELS" "$COMFY/models"
    ok "Symlinked $COMFY/models -> $VOLUME_MODELS"
fi

MODELS="$VOLUME_MODELS"
mkdir -p "$MODELS"/unet "$MODELS"/clip "$MODELS"/vae \
         "$MODELS"/upscale_models "$MODELS"/ultralytics/bbox

touch "$MODELS/.persist_check" 2>/dev/null
if [ -f "/workspace/models/.persist_check" ]; then
    ok "Verified: models are on the persistent volume"
    rm -f "$MODELS/.persist_check"
else
    err "WARNING: models may NOT be persistent - check the symlink above"
fi

# ------------------------------------------------------------
# 3. Fast download tooling
# ------------------------------------------------------------
hdr "Download acceleration"

if [ -n "$HF_TOKEN" ]; then
    ok "HF_TOKEN set"
else
    warn "No HF_TOKEN set - downloads will be slower / rate-limited"
    warn "Add HF_TOKEN as a RunPod env var: huggingface.co/settings/tokens"
fi

HAS_HF_CLI=false
if pip install -q "huggingface_hub[hf_transfer]" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    [ -n "$HF_TOKEN" ] && export HF_TOKEN
    if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
        HAS_HF_CLI=true
        ok "hf_transfer enabled"
    fi
fi
[ "$HAS_HF_CLI" = false ] && warn "hf CLI unavailable - using wget fallback"

STATUS_DIR=$(mktemp -d)
NODE_STATUS_DIR=$(mktemp -d)
trap 'rm -rf "$STATUS_DIR" "$NODE_STATUS_DIR"' EXIT

# ------------------------------------------------------------
# 4. Model fetch helper
# ------------------------------------------------------------
fetch_bg() {
    local name="$1" target="$2" url="$3" min_gb="$4"
    local hf_repo="$5" hf_file="$6"
    local slug; slug=$(echo "$name" | tr -c 'a-zA-Z0-9' '_')
    local logf="$STATUS_DIR/${slug}.log"

    (
        if [ -f "$target" ]; then
            b=$(stat -c%s "$target" 2>/dev/null || echo 0)
            min=$(( min_gb * 900000000 ))
            if [ "$b" -gt "$min" ] || [ "$min_gb" -eq 0 ]; then
                echo "SKIP|$name|$(du -h "$target" | cut -f1)" > "$logf"
                exit 0
            fi
            rm -f "$target"
        fi

        DONE_OK=false

        if [ "$HAS_HF_CLI" = true ] && [ -n "$hf_repo" ] && [ -n "$hf_file" ]; then
            TMPD=$(mktemp -d)
            CLI="huggingface-cli"; command -v hf >/dev/null 2>&1 && CLI="hf"
            if [ "$CLI" = "hf" ]; then
                hf download "$hf_repo" "$hf_file" --local-dir "$TMPD" >/dev/null 2>&1
            else
                huggingface-cli download "$hf_repo" "$hf_file" --local-dir "$TMPD" >/dev/null 2>&1
            fi
            SRC=$(find "$TMPD" -name "$(basename "$hf_file")" -type f 2>/dev/null | head -1)
            if [ -n "$SRC" ] && [ -s "$SRC" ]; then
                mv "$SRC" "$target" && DONE_OK=true
            fi
            rm -rf "$TMPD"
        fi

        if [ "$DONE_OK" = false ]; then
            HDR=()
            [ -n "$HF_TOKEN" ] && HDR=(--header="Authorization: Bearer $HF_TOKEN")
            wget --continue --tries=5 --timeout=60 -q "${HDR[@]}" -O "$target" "$url" 2>/dev/null
        fi

        if [ -f "$target" ] && [ -s "$target" ]; then
            if head -c 200 "$target" | grep -qi "<!doctype\|<html"; then
                echo "FAIL|$name|got HTML - bad URL or missing token" > "$logf"
                rm -f "$target"
            else
                b=$(stat -c%s "$target")
                min=$(( min_gb * 900000000 ))
                if [ "$b" -gt "$min" ] || [ "$min_gb" -eq 0 ]; then
                    echo "DONE|$name|$(du -h "$target" | cut -f1)" > "$logf"
                else
                    echo "FAIL|$name|truncated ($(du -h "$target" | cut -f1))" > "$logf"
                fi
            fi
        else
            echo "FAIL|$name|download error" > "$logf"
        fi
    ) &

    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 2; done
}

# ------------------------------------------------------------
# 5. Custom node self-heal
#    Runs as ONE background job in parallel with model downloads
#    (node installs are pip/git-bound, downloads are network-bound --
#    they don't compete for the same resource).
#    Nodes install sequentially relative to EACH OTHER since
#    Impact-Subpack assumes Impact-Pack's shared deps are set up first.
# ------------------------------------------------------------
NODES_DIR="$COMFY/custom_nodes"
mkdir -p "$NODES_DIR"

install_nodes_bg() {
    (
        install_node() {
            local repo_url="$1" name="$2" run_installer="$3"
            local dest="$NODES_DIR/$name"
            local logf="$NODE_STATUS_DIR/${name}.log"

            # Already present with real content (not an empty/failed clone)?
            if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
                echo "SKIP|$name|already present" > "$logf"
                return 0
            fi
            rm -rf "$dest"

            local attempt ok_clone=false
            for attempt in 1 2 3; do
                if git clone --depth 1 -q "$repo_url" "$dest" 2>/dev/null; then
                    ok_clone=true
                    break
                fi
                rm -rf "$dest"
                sleep 3
            done

            if [ "$ok_clone" = false ]; then
                echo "FAIL|$name|clone failed after 3 attempts" > "$logf"
                return 1
            fi

            if [ -f "$dest/requirements.txt" ]; then
                pip install -q -r "$dest/requirements.txt" 2>/dev/null
            fi

            if [ "$run_installer" = "true" ] && [ -f "$dest/install.py" ]; then
                ( cd "$dest" && python install.py ) >/dev/null 2>&1
            fi

            echo "DONE|$name|installed" > "$logf"
        }

        # Impact-Pack first -- its install.py sets up shared deps
        # that Subpack and the FaceDetailer/UltralyticsDetectorProvider
        # nodes rely on.
        install_node "https://github.com/ltdrdata/ComfyUI-Impact-Pack" \
                      "ComfyUI-Impact-Pack" "true"

        install_node "https://github.com/ltdrdata/ComfyUI-Impact-Subpack" \
                      "ComfyUI-Impact-Subpack" "false"
    ) &
}

hdr "Custom nodes (self-healing)"
if [ "$ZIMAGE" = "true" ]; then
    echo "  Installing/verifying: ComfyUI-Impact-Pack, ComfyUI-Impact-Subpack"
    install_nodes_bg
    NODES_STARTED=1
    ok "Node install running in background (parallel with downloads)"
else
    NODES_STARTED=0
    warn "ZIMAGE=false -- skipping Impact-Pack/Subpack (only needed for endri.json)"
fi

# ------------------------------------------------------------
# 6. Queue model downloads
# ------------------------------------------------------------
hdr "Queuing downloads"
QUEUED=0

if [ "$ZIMAGE" = "true" ]; then
    echo "  L0urta pipeline (Z-Image, two-stage)"

    fetch_bg "ae.safetensors" "$MODELS/vae/ae.safetensors" \
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" 0 \
        "Comfy-Org/z_image_turbo" "split_files/vae/ae.safetensors"

    fetch_bg "qwen_3_4b.safetensors" "$MODELS/clip/qwen_3_4b.safetensors" \
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" 7 \
        "Comfy-Org/z_image_turbo" "split_files/text_encoders/qwen_3_4b.safetensors"

    # z_image_bf16 lives in the Comfy-Org/z_image repo -- NOT z_image_turbo.
    # (Common mistake: the turbo variant of this file does not exist there.)
    fetch_bg "z_image_bf16.safetensors" "$MODELS/unet/z_image_bf16.safetensors" \
        "https://huggingface.co/Comfy-Org/z_image/resolve/main/split_files/diffusion_models/z_image_bf16.safetensors" 7 \
        "Comfy-Org/z_image" "split_files/diffusion_models/z_image_bf16.safetensors"

    fetch_bg "z_image_turbo_bf16.safetensors" "$MODELS/unet/z_image_turbo_bf16.safetensors" \
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" 7 \
        "Comfy-Org/z_image_turbo" "split_files/diffusion_models/z_image_turbo_bf16.safetensors"

    fetch_bg "RealESRGAN_x4plus.safetensors" "$MODELS/upscale_models/RealESRGAN_x4plus.safetensors" \
        "https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged/resolve/main/RealESRGAN_x4plus.safetensors" 0 \
        "Comfy-Org/Real-ESRGAN_repackaged" "RealESRGAN_x4plus.safetensors"

    fetch_bg "face_yolov8m.pt" "$MODELS/ultralytics/bbox/face_yolov8m.pt" \
        "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" 0 \
        "Bingsu/adetailer" "face_yolov8m.pt"

    QUEUED=$((QUEUED+6))
fi

if [ "$KREA2" = "true" ]; then
    echo "  Mara pipeline (Krea 2)"

    fetch_bg "wan_2.1_vae.safetensors" "$MODELS/vae/wan_2.1_vae.safetensors" \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" 0 \
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/vae/wan_2.1_vae.safetensors"

    fetch_bg "qwen3vl_4b_fp8_scaled.safetensors" "$MODELS/clip/qwen3vl_4b_fp8_scaled.safetensors" \
        "https://huggingface.co/AlperKTS/Krea2_FP8/resolve/main/qwen3vl_4b_fp8_scaled.safetensors" 4 \
        "AlperKTS/Krea2_FP8" "qwen3vl_4b_fp8_scaled.safetensors"

    fetch_bg "krea2_turbo_fp8_scaled.safetensors" "$MODELS/unet/krea2_turbo_fp8_scaled.safetensors" \
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors" 12 \
        "Comfy-Org/Krea-2" "diffusion_models/krea2_turbo_fp8_scaled.safetensors"

    QUEUED=$((QUEUED+3))
fi

if [ "$QUEUED" -eq 0 ] && [ "$NODES_STARTED" -eq 0 ]; then
    err "Nothing queued - both ZIMAGE and KREA2 are false. Set at least one to true."
    exit 1
fi

ok "$QUEUED download(s) queued - $MAX_PARALLEL at a time"

# ------------------------------------------------------------
# 7. Progress
# ------------------------------------------------------------
hdr "Working"
LAST=""
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
    LINE=""
    for f in "$MODELS/unet"/*.safetensors "$MODELS/clip"/*.safetensors; do
        [ -f "$f" ] || continue
        LINE="$LINE  $(basename "$f" .safetensors)=$(du -h "$f" 2>/dev/null | cut -f1)"
    done
    if [ -n "$LINE" ] && [ "$LINE" != "$LAST" ]; then
        echo "   $(date +%H:%M:%S)$LINE"
        LAST="$LINE"
    fi
    sleep 20
done
wait

# ------------------------------------------------------------
# 8. Summary
# ------------------------------------------------------------
E=$(( $(date +%s) - START_TS ))
echo ""
echo "  ==========================================="
echo "   Done in $(( E / 60 ))m $(( E % 60 ))s"
echo "  ==========================================="

FAILED=0

echo ""
echo "  Models:"
for f in "$STATUS_DIR"/*.log; do
    [ -f "$f" ] || continue
    IFS='|' read -r st nm dt < "$f"
    case "$st" in
        DONE) ok  "$nm  $dt" ;;
        SKIP) ok  "$nm  $dt (already present)" ;;
        FAIL) err "$nm  $dt"; FAILED=$((FAILED+1)) ;;
    esac
done

echo ""
echo "  Custom nodes:"
ANY_NODE=0
for f in "$NODE_STATUS_DIR"/*.log; do
    [ -f "$f" ] || continue
    ANY_NODE=1
    IFS='|' read -r st nm dt < "$f"
    case "$st" in
        DONE) ok  "$nm  ($dt)" ;;
        SKIP) ok  "$nm  ($dt)" ;;
        FAIL) err "$nm  $dt -- install manually via ComfyUI Manager"; FAILED=$((FAILED+1)) ;;
    esac
done
[ "$ANY_NODE" -eq 0 ] && warn "none installed this run (ZIMAGE=false)"

echo ""
ok "Storage: $MODELS  ($(du -sh "$MODELS" 2>/dev/null | cut -f1) used)"

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo -e "  ${GREEN}All good.${NC} Restart ComfyUI to load new models/nodes."
else
    echo -e "  ${RED}$FAILED failed${NC} - re-run this script; downloads and node"
    echo -e "  installs both resume/skip what's already done."
fi
echo ""
