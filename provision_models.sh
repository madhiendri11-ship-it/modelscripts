#!/bin/bash
# ============================================================
# provision_models.sh — Krea 2 model downloader
# ============================================================
# Works with ANY ComfyUI template (auto-detects the install path).
#
# Downloads to the network volume so models persist across pods.
# Skips anything already present. Resumes partial downloads.
#
# ── USAGE ───────────────────────────────────────────────────
#
# Option A — run once manually (simplest):
#     cd /workspace
#     wget -O provision_models.sh <YOUR_URL>
#     bash provision_models.sh
#
# Option B — auto-run on every pod start:
#     Put this file at /workspace/provision_models.sh
#     In your RunPod template, set Container Start Command to:
#       bash -c "bash /workspace/provision_models.sh & exec /start.sh"
#
# ── OPTIONS (environment variables) ─────────────────────────
#   KREA2=true|false        default true   (13.1GB + 5.2GB + 254MB)
#   ZIMAGE=true|false       default false  (Z-Image pipeline)
#   UPSCALERS=true|false    default true   (4x-UltraSharp, face detect)
#   HF_TOKEN=hf_xxx         optional, for gated repos
# ============================================================

KREA2="${KREA2:-true}"
ZIMAGE="${ZIMAGE:-false}"
UPSCALERS="${UPSCALERS:-true}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[..]${NC} $1"; }
err()  { echo -e "${RED}[!!]${NC} $1"; }

echo ""
echo "  ==========================================="
echo "   Krea 2 — Model Provisioning"
echo "  ==========================================="
echo ""

# ---- 1. Find ComfyUI ---------------------------------------------------------
COMFY=""
for candidate in \
    /workspace/ComfyUI \
    /ComfyUI \
    /opt/ComfyUI \
    /root/ComfyUI \
    /workspace/comfyui \
    /comfyui ; do
    if [ -f "$candidate/main.py" ]; then
        COMFY="$candidate"
        break
    fi
done

# Fall back to a filesystem search
if [ -z "$COMFY" ]; then
    warn "Searching filesystem for ComfyUI..."
    FOUND=$(find /workspace / -maxdepth 4 -name "main.py" -path "*omfy*" 2>/dev/null | head -1)
    [ -n "$FOUND" ] && COMFY=$(dirname "$FOUND")
fi

if [ -z "$COMFY" ]; then
    err "ComfyUI not found. Set it manually:"
    echo "     COMFY=/path/to/ComfyUI bash $0"
    exit 1
fi

ok "ComfyUI found: $COMFY"

MODELS="$COMFY/models"

# Warn if models dir isn't on the persistent volume
case "$MODELS" in
    /workspace/*) ok "Models on network volume (persistent)" ;;
    *) warn "Models NOT on /workspace — they will be LOST when the pod stops"
       warn "Consider symlinking: ln -s /workspace/models $MODELS" ;;
esac

mkdir -p "$MODELS"/unet "$MODELS"/clip "$MODELS"/vae "$MODELS"/loras \
         "$MODELS"/upscale_models "$MODELS"/ultralytics/bbox

# ---- 2. Download helper ------------------------------------------------------
fetch() {
    local name="$1" target="$2" url="$3" min_gb="$4"

    if [ -f "$target" ]; then
        local bytes min_bytes
        bytes=$(stat -c%s "$target" 2>/dev/null || echo 0)
        min_bytes=$(( min_gb * 900000000 ))
        if [ "$bytes" -gt "$min_bytes" ] || [ "$min_gb" -eq 0 ]; then
            ok "$name present ($(du -h "$target" | cut -f1))"
            return 0
        fi
        warn "$name incomplete ($(du -h "$target" | cut -f1)) — resuming"
    fi

    warn "Downloading $name ..."
    if [ -n "$HF_TOKEN" ]; then
        wget --continue --tries=5 --timeout=60 --progress=bar:force \
             --header="Authorization: Bearer $HF_TOKEN" -O "$target" "$url"
    else
        wget --continue --tries=5 --timeout=60 --progress=bar:force \
             -O "$target" "$url"
    fi

    if [ -f "$target" ] && [ -s "$target" ]; then
        local bytes min_bytes
        bytes=$(stat -c%s "$target")
        min_bytes=$(( min_gb * 900000000 ))
        if [ "$bytes" -gt "$min_bytes" ] || [ "$min_gb" -eq 0 ]; then
            ok "$name done ($(du -h "$target" | cut -f1))"
        else
            err "$name truncated — will retry next run"
        fi
    else
        err "$name FAILED"
    fi
}

# ---- 3. Krea 2 ---------------------------------------------------------------
if [ "$KREA2" = "true" ]; then
    echo ""
    echo "  -- Krea 2 --"
    fetch "wan_2.1_vae.safetensors" \
          "$MODELS/vae/wan_2.1_vae.safetensors" \
          "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" 0

    fetch "qwen3vl_4b_fp8_scaled.safetensors" \
          "$MODELS/clip/qwen3vl_4b_fp8_scaled.safetensors" \
          "https://huggingface.co/AlperKTS/Krea2_FP8/resolve/main/qwen3vl_4b_fp8_scaled.safetensors" 4

    fetch "krea2_turbo_fp8_scaled.safetensors" \
          "$MODELS/unet/krea2_turbo_fp8_scaled.safetensors" \
          "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors" 12
fi

# ---- 4. Z-Image (optional) ---------------------------------------------------
if [ "$ZIMAGE" = "true" ]; then
    echo ""
    echo "  -- Z-Image --"
    fetch "ae.safetensors" \
          "$MODELS/vae/ae.safetensors" \
          "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" 0

    fetch "qwen_3_4b.safetensors" \
          "$MODELS/clip/qwen_3_4b.safetensors" \
          "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" 7

    fetch "z_image_turbo_bf16.safetensors" \
          "$MODELS/unet/z_image_turbo_bf16.safetensors" \
          "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" 7
fi

# ---- 5. Upscalers ------------------------------------------------------------
if [ "$UPSCALERS" = "true" ]; then
    echo ""
    echo "  -- Upscalers / detectors --"
    fetch "4x-UltraSharp.pth" \
          "$MODELS/upscale_models/4x-UltraSharp.pth" \
          "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth" 0

    fetch "face_yolov8m.pt" \
          "$MODELS/ultralytics/bbox/face_yolov8m.pt" \
          "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" 0
fi

# ---- 6. Custom nodes (only if missing) ---------------------------------------
NODES="$COMFY/custom_nodes"
mkdir -p "$NODES"
echo ""
echo "  -- Custom nodes --"
for repo in \
    "https://github.com/rgthree/rgthree-comfy" \
    "https://github.com/ltdrdata/ComfyUI-Manager" ; do
    name=$(basename "$repo")
    if [ -d "$NODES/$name" ]; then
        ok "$name present"
    else
        warn "Installing $name ..."
        git clone --depth 1 "$repo" "$NODES/$name" 2>/dev/null && ok "$name installed" \
            || err "$name clone failed"
        [ -f "$NODES/$name/requirements.txt" ] && \
            pip install -q -r "$NODES/$name/requirements.txt" 2>/dev/null
    fi
done

# ---- 7. Summary --------------------------------------------------------------
echo ""
echo "  ==========================================="
echo "   Summary"
echo "  ==========================================="
for m in "unet/krea2_turbo_fp8_scaled.safetensors" \
         "clip/qwen3vl_4b_fp8_scaled.safetensors" \
         "vae/wan_2.1_vae.safetensors"; do
    if [ -f "$MODELS/$m" ]; then
        ok "$(basename "$m")  $(du -h "$MODELS/$m" | cut -f1)"
    else
        err "$(basename "$m")  MISSING"
    fi
done

LORAS=$(ls "$MODELS/loras/"*.safetensors 2>/dev/null | wc -l)
echo ""
if [ "$LORAS" -gt 0 ]; then
    ok "LoRAs: $LORAS"
else
    warn "No LoRAs — upload to $MODELS/loras/"
fi

echo ""
echo "  Restart ComfyUI (or refresh the browser) to see new models."
echo ""
