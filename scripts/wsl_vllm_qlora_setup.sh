#!/usr/bin/env bash
# ==============================================================================
# TopoAgent-Bench: WSL2 Local Training & Serving Orchestrator
# Target Platform: WSL2 Ubuntu 22.04 + NVIDIA GeForce RTX 5070 Ti Laptop (12GB VRAM)
# Target Model:    Qwen/Qwen2.5-3B-Instruct (4-Bit NF4 QLoRA + vLLM LoRA Serving)
# ==============================================================================
#
# VRAM Budget Breakdown (12GB Physical VRAM):
# ------------------------------------------------------------------------------
# 1. Fine-Tuning Stage (4-bit QLoRA):
#    - 4-bit Base Weights:   ~1.85 GB
#    - LoRA Parameters (r=16): ~0.15 GB
#    - Activations & Cache:    ~2.80 GB (batch_size=2, grad_accum=4, seq_len=2048)
#    - CUDA Context Overhead:  ~1.20 GB
#    -> Peak Training VRAM:    ~6.00 GB (< 7.5 GB ceiling, 6.0 GB headroom)
#
# 2. Serving Stage (vLLM Engine):
#    - Base Weights (bf16/fp16): ~6.20 GB (or AWQ 4-bit ~2.2 GB)
#    - Dynamic LoRA Adapter:     ~0.15 GB
#    - KV Cache (utilization 0.72): ~2.30 GB
#    -> Peak Serving VRAM:       ~8.64 GB (12GB * 0.72 = 8.64 GB budget)
#    -> Free Margin for Windows DWM: ~3.36 GB
# ==============================================================================

set -euo pipefail

# Color Codes for Terminal UI
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

VENV_PATH="${HOME}/.venvs/topoagent-bench"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo -e "${BOLD}${BLUE}======================================================================${NC}"
echo -e "${BOLD}${BLUE} TopoAgent-Bench: WSL2 Ubuntu22 vLLM & 4-Bit QLoRA Setup Pipeline${NC}"
echo -e "${BOLD}${BLUE}======================================================================${NC}"

# ------------------------------------------------------------------------------
# Step 0: Pre-flight Diagnostic Checks
# ------------------------------------------------------------------------------
check_gpu() {
    echo -e "\n${BOLD}[1/4] Checking NVIDIA GPU and CUDA support in WSL2...${NC}"
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}[ERROR] 'nvidia-smi' not found! Make sure NVIDIA driver is installed on Windows Host.${NC}"
        exit 1
    fi

    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo -e "${GREEN}[+] NVIDIA GPU detected and accessible via WSL2.${NC}"
}

# ------------------------------------------------------------------------------
# Step 1: Virtual Environment and Dependencies Setup
# ------------------------------------------------------------------------------
setup_environment() {
    echo -e "\n${BOLD}[2/4] Setting up Python virtual environment at '${VENV_PATH}'...${NC}"
    
    if [ ! -d "${VENV_PATH}" ]; then
        echo "[*] Creating Python3 virtual environment..."
        python3 -m venv "${VENV_PATH}"
    fi

    # Activate virtual environment
    source "${VENV_PATH}/bin/activate"

    echo "[*] Upgrading pip, setuptools, wheel..."
    pip install --upgrade pip setuptools wheel --quiet

    echo "[*] Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet

    echo "[*] Installing LLM fine-tuning and serving stack (vLLM, PEFT, TRL, bitsandbytes)..."
    pip install \
        "vllm>=0.6.0" \
        "peft>=0.11.0" \
        "trl>=0.9.0" \
        "bitsandbytes>=0.43.0" \
        "transformers>=4.43.0" \
        "accelerate>=0.32.0" \
        "datasets>=2.20.0" \
        --quiet

    echo -e "${GREEN}[+] Python environment successfully initialized.${NC}"
}

# ------------------------------------------------------------------------------
# Step 2: 4-Bit QLoRA Fine-Tuning Execution
# ------------------------------------------------------------------------------
run_training() {
    echo -e "\n${BOLD}[3/4] Launching 4-Bit QLoRA Fine-Tuning on Distillation Trajectories...${NC}"
    source "${VENV_PATH}/bin/activate"
    cd "${PROJECT_ROOT}"

    # Verify dataset presence; bootstrap sample trajectories if missing
    if [ ! -f "results/distill_trajectories.jsonl" ]; then
        echo -e "${YELLOW}[!] 'results/distill_trajectories.jsonl' not found. Bootstrapping initial trajectories...${NC}"
        python3 scripts/train_qlora_distill.py --dry-run-verify
    fi

    echo "[*] Executing QLoRA training script (target VRAM < 7.5GB)..."
    python3 scripts/train_qlora_distill.py \
        --model-id "Qwen/Qwen2.5-3B-Instruct" \
        --dataset-path "results/distill_trajectories.jsonl" \
        --output-dir "results/qlora-adapter" \
        --log-path "results/qlora_training_log.json" \
        --epochs 3 \
        --batch-size 2 \
        --grad-accum 4 \
        --lr 2e-4 \
        --lora-r 16 \
        --lora-alpha 32 \
        --max-seq-len 2048

    echo -e "${GREEN}[+] Fine-tuning completed. Adapter saved to 'results/qlora-adapter'.${NC}"
}

# ------------------------------------------------------------------------------
# Step 3: Local vLLM Inference Serving with LoRA
# ------------------------------------------------------------------------------
run_serving() {
    echo -e "\n${BOLD}[4/4] Launching Local vLLM Inference Server with LoRA Adapter...${NC}"
    source "${VENV_PATH}/bin/activate"
    cd "${PROJECT_ROOT}"

    ADAPTER_DIR="${PROJECT_ROOT}/results/qlora-adapter"
    if [ ! -d "${ADAPTER_DIR}" ]; then
        echo -e "${YELLOW}[Notice] Adapter directory '${ADAPTER_DIR}' not found. Serving base model.${NC}"
        LORA_ARGS=""
    else
        echo -e "${GREEN}[+] Attaching LoRA module 'topo-planner' from '${ADAPTER_DIR}'${NC}"
        LORA_ARGS="--enable-lora --lora-modules topo-planner=${ADAPTER_DIR}"
    fi

    echo -e "${BLUE}[*] Server Configuration:${NC}"
    echo "    - Base Model:             Qwen/Qwen2.5-3B-Instruct"
    echo "    - GPU Memory Utilization: 0.72 (~8.64 GB limit on 12GB RTX 5070 Ti)"
    echo "    - Max Context Length:     4096 tokens"
    echo "    - HTTP Port:              8000"
    echo "    - LoRA Module:            topo-planner"
    echo ""
    echo -e "${YELLOW}To test the serving API from another terminal, run:${NC}"
    echo "curl http://localhost:8000/v1/chat/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{\"model\": \"topo-planner\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the critical path length in hours?\"}]}'"
    echo ""

    # Launch vLLM OpenAI-compatible server
    vllm serve Qwen/Qwen2.5-3B-Instruct \
        ${LORA_ARGS} \
        --gpu-memory-utilization 0.72 \
        --max-model-len 4096 \
        --port 8000 \
        --trust-remote-code
}

# ------------------------------------------------------------------------------
# Dispatcher / Command Router
# ------------------------------------------------------------------------------
ACTION="${1:---all}"

case "${ACTION}" in
    --check-gpu)
        check_gpu
        ;;
    --install)
        check_gpu
        setup_environment
        ;;
    --dry-run)
        check_gpu
        setup_environment
        source "${VENV_PATH}/bin/activate"
        cd "${PROJECT_ROOT}"
        python3 scripts/train_qlora_distill.py --dry-run-verify
        ;;
    --train)
        check_gpu
        setup_environment
        run_training
        ;;
    --serve)
        check_gpu
        setup_environment
        run_serving
        ;;
    --all)
        check_gpu
        setup_environment
        run_training
        run_serving
        ;;
    *)
        echo "Usage: $0 [--check-gpu | --install | --dry-run | --train | --serve | --all]"
        exit 1
        ;;
esac
