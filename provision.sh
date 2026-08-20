#!/bin/bash
# ============================================================
#  provision.sh — ComfyUI model + node provisioner for RunPod
# ============================================================
#  - Auto-detects ComfyUI wherever the template installed it
#  - FORCES models onto /workspace (network volume) via symlink
#    so nothing is lost when the pod stops
#  - Uses HuggingFace hf_transfer for faster downloads
#  - Resumes partial downloads, skips completed ones
#  - Installs ComfyUI-Impact-Pack + ComfyUI-Impact-Subpack
#    AFTER all model downloads finish (sequential, on purpose --
#    avoids any race with the pod's own first-boot move-to-volume
#    step, which has previously collided with concurrent writes)
#  - VALIDATES the install for real: does a throwaway supervised
#    ComfyUI boot on an alternate port and checks its own
#    "(IMPORT FAILED)" marker for each target node, instead of
#    just checking the folder exists (a folder can be present
#    while the node still fails to import -- we hit exactly this
#    with the missing 'ultralytics' module).
#
# ------------------------------------------------------------
#  QUICK START
# ------------------------------------------------------------
#  Manual (run this first to confirm it works):
#      bash /workspace/provision.sh
#
#  Auto on every pod boot -- RunPod Template -> Container Start Command:
#      bash -c "sleep 30 && curl -sL YOUR_RAW_GITHUB_URL -o /workspace/provision.sh && \
#               bash /workspace/provision.sh > /workspace/provision.log 2>&1 & \
#               exec /start.sh"
#  (the 'sleep 30' lets the template's own first-boot move-to-volume
#   step finish before this script touches the same directories)
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
#                 (installed + validated automatically below)
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
VALIDATE_PORT="${VALIDATE_PORT:-8189}"

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
# 5. Queue model downloads (parallel, as before)
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

if [ "$QUEUED" -gt 0 ]; then
    ok "$QUEUED download(s) queued - $MAX_PARALLEL at a time"

    hdr "Downloading"
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
    ok "All model downloads finished"
else
    warn "No models queued (ZIMAGE=false, KREA2=false)"
fi

# ------------------------------------------------------------
# 6. Custom nodes -- runs ONLY AFTER all downloads above are done.
#    Sequential by design: no race with concurrent volume writes,
#    easy to reason about, and node installs are fast (~30-90s)
#    next to multi-GB model downloads so the cost is negligible.
# ------------------------------------------------------------
NODES_DIR="$COMFY/custom_nodes"
mkdir -p "$NODES_DIR"

install_node() {
    local repo_url="$1" name="$2"
    local dest="$NODES_DIR/$name"

    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        ok "$name folder already present"
    else
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
            err "$name clone failed after 3 attempts"
            return 1
        fi
        ok "$name cloned"
    fi

    if [ -f "$dest/requirements.txt" ]; then
        pip install -q -r "$dest/requirements.txt" 2>/dev/null
    fi
    if [ -f "$dest/install.py" ]; then
        ( cd "$dest" && python install.py ) >/dev/null 2>&1
    fi
}

if [ "$ZIMAGE" = "true" ]; then
    hdr "Custom nodes (Impact-Pack + Impact-Subpack)"

    # Impact-Pack's own documented manual dependency list (from its README,
    # "Package Dependencies (If you need to manual setup.)" section) --
    # installed explicitly since requirements.txt alone has been unreliable
    # in this environment. numpy is intentionally NOT pinned here: forcing
    # numpy<2 blind can downgrade numpy under other packages (torch/
    # transformers builds in this template expect numpy 2.x) and break the
    # whole environment. Let Impact-Pack's own requirements.txt negotiate
    # that pin with pip's resolver instead of overriding it ourselves.
    pip install -q segment-anything scikit-image piexif opencv-python-headless \
                   scipy dill matplotlib onnxruntime 2>/dev/null

    # ultralytics (YOLO) is required by Impact-Subpack's detectors
    # (UltralyticsDetectorProvider, BboxDetectorCombined_v2) but is not
    # reliably pulled in by requirements.txt alone -- install explicitly.
    # This is the exact fix for the ModuleNotFoundError we hit earlier.
    pip install -q ultralytics 2>/dev/null

    install_node "https://github.com/ltdrdata/ComfyUI-Impact-Pack" "ComfyUI-Impact-Pack"
    install_node "https://github.com/ltdrdata/ComfyUI-Impact-Subpack" "ComfyUI-Impact-Subpack"

    # ----------------------------------------------------------
    # 6b. VALIDATE -- actually boot ComfyUI on a throwaway port
    #     and check its own "(IMPORT FAILED)" marker, rather than
    #     trusting that the folder existing means the node loaded.
    # ----------------------------------------------------------
    hdr "Validating node install (throwaway boot on port $VALIDATE_PORT)"

    VALIDATE_LOG=$(mktemp)
    (
        cd "$COMFY"
        python main.py --listen 127.0.0.1 --port "$VALIDATE_PORT" --cpu \
            > "$VALIDATE_LOG" 2>&1
    ) &
    VALIDATE_PID=$!

    WAITED=0
    while [ "$WAITED" -lt 90 ]; do
        if grep -q "Import times for custom nodes" "$VALIDATE_LOG" 2>/dev/null; then
            sleep 2
            break
        fi
        if ! kill -0 "$VALIDATE_PID" 2>/dev/null; then
            break
        fi
        sleep 2
        WAITED=$((WAITED+2))
    done

    kill "$VALIDATE_PID" 2>/dev/null
    wait "$VALIDATE_PID" 2>/dev/null

    for node in "ComfyUI-Impact-Pack" "ComfyUI-Impact-Subpack"; do
        line=$(grep "$node" "$VALIDATE_LOG" | grep "seconds" | tail -1)
        if [ -z "$line" ]; then
            echo "UNKNOWN|$node|not found in import log (boot may have timed out)" > "$NODE_STATUS_DIR/${node}.log"
        elif echo "$line" | grep -qi "IMPORT FAILED"; then
            reason=$(grep -A 3 "$node" "$VALIDATE_LOG" | grep -iE "Error|Traceback" | tail -1)
            echo "FAIL|$node|$reason" > "$NODE_STATUS_DIR/${node}.log"
        else
            echo "DONE|$node|loaded cleanly" > "$NODE_STATUS_DIR/${node}.log"
        fi
    done

    rm -f "$VALIDATE_LOG"
else
    warn "ZIMAGE=false -- skipping Impact-Pack/Subpack (only needed for endri.json)"
fi

# ------------------------------------------------------------
# 7. Summary
# ------------------------------------------------------------
E=$(( $(date +%s) - START_TS ))
echo ""
echo "  ==========================================="
echo "   Done in $(( E / 60 ))m $(( E % 60 ))s"
echo "  ==========================================="

FAILED=0

echo ""
echo "  Models:"
if [ -d "$STATUS_DIR" ] && [ -n "$(ls -A "$STATUS_DIR" 2>/dev/null)" ]; then
    for f in "$STATUS_DIR"/*.log; do
        [ -f "$f" ] || continue
        IFS='|' read -r st nm dt < "$f"
        case "$st" in
            DONE) ok  "$nm  $dt" ;;
            SKIP) ok  "$nm  $dt (already present)" ;;
            FAIL) err "$nm  $dt"; FAILED=$((FAILED+1)) ;;
        esac
    done
else
    warn "none queued"
fi

echo ""
echo "  Custom nodes (validated by actual boot, not just folder check):"
if [ -d "$NODE_STATUS_DIR" ] && [ -n "$(ls -A "$NODE_STATUS_DIR" 2>/dev/null)" ]; then
    for f in "$NODE_STATUS_DIR"/*.log; do
        [ -f "$f" ] || continue
        IFS='|' read -r st nm dt < "$f"
        case "$st" in
            DONE)    ok   "$nm  -- $dt" ;;
            FAIL)    err  "$nm  -- FAILED TO IMPORT: $dt"; FAILED=$((FAILED+1)) ;;
            UNKNOWN) warn "$nm  -- $dt"; FAILED=$((FAILED+1)) ;;
        esac
    done
else
    warn "none checked (ZIMAGE=false)"
fi

echo ""
ok "Storage: $MODELS  ($(du -sh "$MODELS" 2>/dev/null | cut -f1) used)"

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo -e "  ${GREEN}All good.${NC} Restart the main ComfyUI process to pick up new models/nodes:"
    echo "    pkill -f \"main.py --listen 0.0.0.0\""
else
    echo -e "  ${RED}$FAILED item(s) failed${NC} - re-run this script; downloads and"
    echo -e "  node clones both resume/skip what's already done. If a node keeps"
    echo -e "  failing to import, check the reason above and install its missing"
    echo -e "  dependency manually, e.g.:  pip install <package>"
fi
echo ""
