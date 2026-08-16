#!/bin/bash
# ============================================================
#  provision.sh — ComfyUI model + LoRA provisioner for RunPod
# ============================================================
#  - Auto-detects ComfyUI wherever the template installed it
#  - FORCES models onto /workspace (network volume) via symlink
#    so nothing is lost when the pod stops
#  - Uses HuggingFace hf_transfer for 5-10x faster downloads
#  - Downloads models + LoRAs in parallel, resumes partials
#  - Safe to re-run: skips anything already complete
#
# ------------------------------------------------------------
#  QUICK START
# ------------------------------------------------------------
#  Manual:
#      bash /workspace/provision.sh
#
#  Auto on every pod boot — RunPod Template → Container Start Command:
#      bash -c "curl -sL YOUR_RAW_GITHUB_URL -o /workspace/provision.sh && \
#               bash /workspace/provision.sh > /workspace/provision.log 2>&1 & \
#               exec /start.sh"
#
#  Watch progress:
#      tail -f /workspace/provision.log
#
# ------------------------------------------------------------
#  TOKENS  (set as RunPod template Environment Variables)
# ------------------------------------------------------------
#  HF_TOKEN        HuggingFace token — STRONGLY RECOMMENDED
#                  Speeds up downloads and lifts rate limits.
#                  Get one: huggingface.co/settings/tokens
#                  (a "read" token is enough)
#
#  CIVITAI_TOKEN   Only if pulling LoRAs from Civitai
#                  Get one: civitai.com/user/account
#
#  You can also hardcode them just below, but env vars are safer.
# ------------------------------------------------------------

# ┌──────────────────────────────────────────────────────────┐
# │  TOKENS — leave blank to use RunPod env vars instead      │
# └──────────────────────────────────────────────────────────┘
HF_TOKEN="${HF_TOKEN:-}"
CIVITAI_TOKEN="${CIVITAI_TOKEN:-}"

# ┌──────────────────────────────────────────────────────────┐
# │  YOUR LORAS                                              │
# │  Format: "filename.safetensors|url"                      │
# │                                                          │
# │  Can also be listed in /workspace/loras.txt (same        │
# │  format) which is easier to change per-pod.              │
# └──────────────────────────────────────────────────────────┘
LORAS=(
    # "a0lorta.safetensors|https://huggingface.co/USER/REPO/resolve/main/a0lorta.safetensors"
    # "L0urta.safetensors|https://huggingface.co/USER/REPO/resolve/main/L0urta.safetensors"
)

# ┌──────────────────────────────────────────────────────────┐
# │  WHAT TO DOWNLOAD                                        │
# └──────────────────────────────────────────────────────────┘
KREA2="${KREA2:-true}"            # krea2 turbo + qwen3vl clip + wan vae
ZIMAGE="${ZIMAGE:-false}"         # z-image turbo pipeline
UPSCALERS="${UPSCALERS:-true}"    # 4x-UltraSharp + face detector
DOWNLOAD_LORAS="${DOWNLOAD_LORAS:-true}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

# ============================================================

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[..]${NC} $1"; }
err()  { echo -e "${RED}[!!]${NC} $1"; }
hdr()  { echo -e "\n${CYAN}--- $1 ---${NC}"; }

START_TS=$(date +%s)

echo ""
echo "  ==========================================="
echo "   ComfyUI Provisioner"
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
    err "ComfyUI not found. Run with:  COMFY=/path/to/ComfyUI bash \$0"
    exit 1
fi
ok "ComfyUI: $COMFY"

# ------------------------------------------------------------
# 2. Force models onto the persistent volume
#    Many templates install ComfyUI to the container disk (/ComfyUI).
#    Anything written there is DESTROYED when the pod stops.
#    We symlink models/ to /workspace so downloads survive.
# ------------------------------------------------------------
hdr "Persistent storage"

VOLUME_MODELS="/workspace/models"

if [ -L "$COMFY/models" ]; then
    ok "models/ already symlinked -> $(readlink "$COMFY/models")"
elif [[ "$COMFY" == /workspace/* ]]; then
    ok "ComfyUI already on the volume"
    VOLUME_MODELS="$COMFY/models"
else
    warn "ComfyUI is on the container disk — relinking models to the volume"
    mkdir -p "$VOLUME_MODELS"
    if [ -d "$COMFY/models" ]; then
        cp -rn "$COMFY/models/." "$VOLUME_MODELS/" 2>/dev/null
        rm -rf "$COMFY/models"
    fi
    ln -s "$VOLUME_MODELS" "$COMFY/models"
    ok "Symlinked $COMFY/models -> $VOLUME_MODELS"
fi

MODELS="$VOLUME_MODELS"
mkdir -p "$MODELS"/unet "$MODELS"/clip "$MODELS"/vae "$MODELS"/loras \
         "$MODELS"/upscale_models "$MODELS"/checkpoints \
         "$MODELS"/ultralytics/bbox

# Sanity check — write a file and confirm it lands on the volume
touch "$MODELS/.persist_check" 2>/dev/null
if [ -f "/workspace/models/.persist_check" ]; then
    ok "Verified: models are on the persistent volume"
    rm -f "$MODELS/.persist_check"
else
    err "WARNING: models may NOT be persistent — check the symlink"
fi

# ------------------------------------------------------------
# 3. Fast download tooling
# ------------------------------------------------------------
hdr "Download acceleration"

if [ -n "$HF_TOKEN" ]; then
    ok "HF_TOKEN set — authenticated downloads enabled"
else
    warn "No HF_TOKEN — downloads will be slower and rate-limited"
    warn "Add HF_TOKEN as a RunPod env var (huggingface.co/settings/tokens)"
fi

HAS_HF_CLI=false
if pip install -q "huggingface_hub[hf_transfer]" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    [ -n "$HF_TOKEN" ] && export HF_TOKEN
    if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
        HAS_HF_CLI=true
        ok "hf_transfer enabled (multi-threaded, ~5-10x faster)"
    fi
fi
[ "$HAS_HF_CLI" = false ] && warn "hf CLI unavailable — falling back to wget"

STATUS_DIR=$(mktemp -d)
trap 'rm -rf "$STATUS_DIR"' EXIT

# ------------------------------------------------------------
# 4. Fetch helper
#    HuggingFace repos use the CLI (fast, chunked, resumable).
#    Everything else uses wget --continue.
# ------------------------------------------------------------
fetch_bg() {
    local name="$1" target="$2" url="$3" min_gb="$4" kind="${5:-model}"
    local hf_repo="$6" hf_file="$7"     # optional, enables the fast path
    local slug; slug=$(echo "$name" | tr -c 'a-zA-Z0-9' '_')
    local logf="$STATUS_DIR/${slug}.log"

    (
        # Already complete?
        if [ -f "$target" ]; then
            b=$(stat -c%s "$target" 2>/dev/null || echo 0)
            min=$(( min_gb * 900000000 ))
            if [ "$b" -gt "$min" ] || [ "$min_gb" -eq 0 ]; then
                echo "SKIP|$name|$(du -h "$target" | cut -f1)|$kind" > "$logf"
                exit 0
            fi
        fi

        DONE_OK=false

        # --- Fast path: HuggingFace CLI ---
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

        # --- Fallback: wget ---
        if [ "$DONE_OK" = false ]; then
            HDR=()
            case "$url" in
                *huggingface.co*|*hf.co*) [ -n "$HF_TOKEN" ] && HDR=(--header="Authorization: Bearer $HF_TOKEN") ;;
                *civitai.com*)            [ -n "$CIVITAI_TOKEN" ] && HDR=(--header="Authorization: Bearer $CIVITAI_TOKEN") ;;
            esac
            wget --continue --tries=5 --timeout=60 -q "${HDR[@]}" -O "$target" "$url" 2>/dev/null
        fi

        # --- Validate ---
        if [ -f "$target" ] && [ -s "$target" ]; then
            if head -c 200 "$target" | grep -qi "<!doctype\|<html\|Invalid username"; then
                echo "FAIL|$name|got HTML — bad URL or missing token|$kind" > "$logf"
                rm -f "$target"
            else
                b=$(stat -c%s "$target")
                min=$(( min_gb * 900000000 ))
                if [ "$b" -gt "$min" ] || [ "$min_gb" -eq 0 ]; then
                    echo "DONE|$name|$(du -h "$target" | cut -f1)|$kind" > "$logf"
                else
                    echo "FAIL|$name|truncated ($(du -h "$target" | cut -f1))|$kind" > "$logf"
                fi
            fi
        else
            echo "FAIL|$name|download error|$kind" > "$logf"
        fi
    ) &

    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 2; done
}

# ------------------------------------------------------------
# 5. Queue downloads
# ------------------------------------------------------------
hdr "Queuing downloads"
QUEUED=0

if [ "$KREA2" = "true" ]; then
    echo "  Krea 2 pipeline"
    fetch_bg "wan_2.1_vae.safetensors" "$MODELS/vae/wan_2.1_vae.safetensors" \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" 0 model \
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/vae/wan_2.1_vae.safetensors"

    fetch_bg "qwen3vl_4b_fp8_scaled.safetensors" "$MODELS/clip/qwen3vl_4b_fp8_scaled.safetensors" \
        "https://huggingface.co/AlperKTS/Krea2_FP8/resolve/main/qwen3vl_4b_fp8_scaled.safetensors" 4 model \
        "AlperKTS/Krea2_FP8" "qwen3vl_4b_fp8_scaled.safetensors"

    fetch_bg "krea2_turbo_fp8_scaled.safetensors" "$MODELS/unet/krea2_turbo_fp8_scaled.safetensors" \
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors" 12 model \
        "Comfy-Org/Krea-2" "diffusion_models/krea2_turbo_fp8_scaled.safetensors"
    QUEUED=$((QUEUED+3))
fi

if [ "$ZIMAGE" = "true" ]; then
    echo "  Z-Image pipeline"
    fetch_bg "ae.safetensors" "$MODELS/vae/ae.safetensors" \
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" 0 model \
        "Comfy-Org/z_image_turbo" "split_files/vae/ae.safetensors"

    fetch_bg "qwen_3_4b.safetensors" "$MODELS/clip/qwen_3_4b.safetensors" \
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" 7 model \
        "Comfy-Org/z_image_turbo" "split_files/text_encoders/qwen_3_4b.safetensors"

    fetch_bg "z_image_turbo_bf16.safetensors" "$MODELS/unet/z_image_turbo_bf16.safetensors" \
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" 7 model \
        "Comfy-Org/z_image_turbo" "split_files/diffusion_models/z_image_turbo_bf16.safetensors"
    QUEUED=$((QUEUED+3))
fi

if [ "$UPSCALERS" = "true" ]; then
    echo "  Upscalers / detectors"
    fetch_bg "4x-UltraSharp.pth" "$MODELS/upscale_models/4x-UltraSharp.pth" \
        "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth" 0 model \
        "lokCX/4x-Ultrasharp" "4x-UltraSharp.pth"

    fetch_bg "face_yolov8m.pt" "$MODELS/ultralytics/bbox/face_yolov8m.pt" \
        "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" 0 model \
        "Bingsu/adetailer" "face_yolov8m.pt"
    QUEUED=$((QUEUED+2))
fi

# LoRAs
if [ "$DOWNLOAD_LORAS" = "true" ]; then
    LORA_ENTRIES=()
    if [ -f "/workspace/loras.txt" ]; then
        while IFS= read -r line; do
            line="${line%%#*}"; line="$(echo "$line" | xargs)"
            [ -n "$line" ] && LORA_ENTRIES+=("$line")
        done < /workspace/loras.txt
        ok "Loaded $(( ${#LORA_ENTRIES[@]} )) entr(ies) from /workspace/loras.txt"
    fi
    for e in "${LORAS[@]}"; do
        e="$(echo "$e" | xargs)"; [ -n "$e" ] && LORA_ENTRIES+=("$e")
    done

    if [ "${#LORA_ENTRIES[@]}" -gt 0 ]; then
        echo "  LoRAs"
        for entry in "${LORA_ENTRIES[@]}"; do
            fname="${entry%%|*}"; furl="${entry#*|}"
            if [ -z "$fname" ] || [ -z "$furl" ] || [ "$fname" = "$furl" ]; then
                err "Bad line (need filename|url): $entry"; continue
            fi
            # Derive repo/file for the HF fast path when possible
            hrepo=""; hfile=""
            if [[ "$furl" == *huggingface.co* ]]; then
                stripped="${furl#*huggingface.co/}"
                hrepo="$(echo "$stripped" | cut -d/ -f1-2)"
                hfile="${stripped#*/resolve/main/}"
                [ "$hfile" = "$stripped" ] && { hrepo=""; hfile=""; }
            fi
            fetch_bg "$fname" "$MODELS/loras/$fname" "$furl" 0 lora "$hrepo" "$hfile"
            QUEUED=$((QUEUED+1))
        done
    else
        warn "No LoRAs configured (edit /workspace/loras.txt or the LORAS array)"
    fi
fi

echo ""
ok "$QUEUED download(s) queued — $MAX_PARALLEL at a time"

# ------------------------------------------------------------
# 6. Custom nodes (runs while downloads continue)
# ------------------------------------------------------------
hdr "Custom nodes"
NODES="$COMFY/custom_nodes"
mkdir -p "$NODES"
for repo in "https://github.com/rgthree/rgthree-comfy" \
            "https://github.com/ltdrdata/ComfyUI-Manager" ; do
    name=$(basename "$repo")
    if [ -d "$NODES/$name" ]; then
        ok "$name present"
    elif git clone --depth 1 -q "$repo" "$NODES/$name" 2>/dev/null; then
        ok "$name installed"
        [ -f "$NODES/$name/requirements.txt" ] && \
            pip install -q -r "$NODES/$name/requirements.txt" 2>/dev/null
    else
        err "$name clone failed — install via ComfyUI Manager"
    fi
done

# ------------------------------------------------------------
# 7. Progress
# ------------------------------------------------------------
hdr "Downloading"
LAST=""
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
    LINE=""
    for f in "$MODELS/unet"/*.safetensors "$MODELS/clip"/*.safetensors "$MODELS/loras"/*.safetensors; do
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
    IFS='|' read -r st nm dt kd < "$f"
    [ "$kd" = "lora" ] && continue
    case "$st" in
        DONE) ok  "$nm  $dt" ;;
        SKIP) ok  "$nm  $dt (already present)" ;;
        FAIL) err "$nm  $dt"; FAILED=$((FAILED+1)) ;;
    esac
done

echo ""
echo "  LoRAs:"
ANY=0
for f in "$STATUS_DIR"/*.log; do
    [ -f "$f" ] || continue
    IFS='|' read -r st nm dt kd < "$f"
    [ "$kd" != "lora" ] && continue
    ANY=1
    case "$st" in
        DONE) ok  "$nm  $dt" ;;
        SKIP) ok  "$nm  $dt (already present)" ;;
        FAIL) err "$nm  $dt"; FAILED=$((FAILED+1)) ;;
    esac
done
[ "$ANY" -eq 0 ] && warn "none configured"

echo ""
ok "Storage: $MODELS  ($(du -sh "$MODELS" 2>/dev/null | cut -f1) used)"
ok "LoRA files on volume: $(ls "$MODELS/loras/"*.safetensors 2>/dev/null | wc -l)"

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo -e "  ${GREEN}All good.${NC} Refresh ComfyUI to load the new models."
else
    echo -e "  ${RED}$FAILED failed${NC} — re-run this script; partial files resume."
fi
echo ""
