#!/usr/bin/env python3
"""OpenAI-Compatible Local LLM Serving Engine for TopoAgent-Bench.

Engineered for local inference on NVIDIA RTX 5070 Ti Laptop GPU (12GB VRAM, sm_120)
using PyTorch native SDPA (Scaled Dot-Product Attention) and LoRA adapter integration.

Provides:
- GET  /health: Health check and GPU telemetry.
- GET  /v1/models: OpenAI models list.
- POST /v1/chat/completions: OpenAI chat completions endpoint compatible with HybridLLMRouter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def resolve_project_path(p: str | Path) -> Path:
    path_obj = Path(p)
    return path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_obj)


# =====================================================================
# Request / Response Schemas (OpenAI Spec)
# =====================================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "topo-planner"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    enable_thinking: Optional[bool] = False


# Global server state
model = None
tokenizer = None
gpu_device_name = "CUDA"
serving_stats = {
    "requests_served": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_generation_time_ms": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, gpu_device_name
    model_id = app.state.model_id
    adapter_path = resolve_project_path(app.state.adapter_path)

    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    print("\n" + "=" * 70)
    print(" TopoAgent-Bench: Local OpenAI Serving Engine Startup")
    print("=" * 70)
    print(f"[*] Base Model: {model_id}")
    print(f"[*] Adapter Path: {adapter_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading Base Model on cuda:0 (bfloat16 SDPA sm_120)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        print(f"[+] Attaching Fine-Tuned LoRA Adapter from '{adapter_path}'...")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    else:
        print("[!] No LoRA adapter found; serving base model directly.")
        model = base_model

    model.eval()
    gpu_device_name = torch.cuda.get_device_name(0)
    alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"[+] Server Ready on GPU: {gpu_device_name} (Allocated: {alloc_mb:.1f} MB VRAM)")
    print("=" * 70 + "\n")

    yield

    print("[*] Shutting down Local Serving Engine...")
    del model
    del base_model
    torch.cuda.empty_cache()


app = FastAPI(title="TopoAgent Local Inference Server", lifespan=lifespan)


@app.get("/health")
async def health_check():
    alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    return {
        "status": "healthy",
        "gpu": gpu_device_name,
        "vram_allocated_mb": round(alloc_mb, 2),
        "vram_peak_mb": round(peak_mb, 2),
        "serving_stats": serving_stats,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "topo-planner", "object": "model", "owned_by": "topoagent"},
            {"id": "default", "object": "model", "owned_by": "topoagent"},
            {"id": "local_vllm", "object": "model", "owned_by": "topoagent"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global model, tokenizer, serving_stats
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    t0 = time.perf_counter()
    conv_messages = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                conv_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = "\n".join(f"<|im_start|>{m.role}\n{m.content}<|im_end|>" for m in req.messages) + "\n<|im_start|>assistant\n"
    except Exception:
        prompt_text = "\n".join(f"{m.role.capitalize()}: {m.content}" for m in req.messages) + "\nAssistant: "

    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
    prompt_tokens = inputs["input_ids"].shape[1]

    max_new = min(req.max_tokens or 2048, 4096)
    temp = max(req.temperature or 0.7, 0.01)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=(temp > 0.1),
            temperature=temp,
            top_p=req.top_p or 0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_ids = generated_ids[0, prompt_tokens:]
    completion_tokens = completion_ids.shape[0]
    output_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    latency_ms = (time.perf_counter() - t0) * 1000.0
    throughput = (completion_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

    serving_stats["requests_served"] += 1
    serving_stats["total_prompt_tokens"] += prompt_tokens
    serving_stats["total_completion_tokens"] += completion_tokens
    serving_stats["total_generation_time_ms"] += latency_ms

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "telemetry": {
            "latency_ms": round(latency_ms, 2),
            "tokens_per_sec": round(throughput, 2),
            "gpu": gpu_device_name,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Local OpenAI Server for TopoAgent-Bench")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="results/qlora-adapter")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app.state.model_id = args.model_id
    app.state.adapter_path = args.adapter_path

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
