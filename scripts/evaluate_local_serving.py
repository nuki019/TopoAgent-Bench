#!/usr/bin/env python3
"""Evaluator for Local LLM / vLLM Inference Serving on TopoAgent-Bench.

Evaluates HybridLLMRouter with model_tier="local_vllm" on:
1. math_01: DAG Critical Path & Slack Scheduling
2. code_01: Longest Path in Directed Acyclic Graph (DAG)
3. duohire_01: Multi-Constraint Bilateral Recruitment Intelligence

Records:
- Model Name, GPU Hardware Name
- VRAM Allocated / Peak VRAM (MB)
- Prompt Tokens, Completion Tokens, Total Tokens
- Latency (ms), Throughput (tokens/sec)
- Deterministic Evaluation Accuracy (0.0 - 1.0)
Exports full benchmark telemetry to 'results/local_vllm_serving_log.json'.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from topoagent.llm_router import HybridLLMRouter, LLMResponse
from benchmarks.tasks import (
    MATH_PLANNING_TASKS,
    CODE_SYNTHESIS_TASKS,
    DUOHIRE_A2A_TASKS,
    evaluate_task,
)


def resolve_project_path(p: str | Path) -> Path:
    path_obj = Path(p)
    return path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_obj)


def query_server_health(base_url: str = "http://127.0.0.1:8000") -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TopoAgent-Bench"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def main():
    print("\n" + "=" * 70)
    print(" TopoAgent-Bench: Local Model Serving (vLLM / PyTorch SDPA) Benchmark")
    print("=" * 70)

    server_url = os.getenv("LOCAL_VLLM_URL", "http://127.0.0.1:8000/v1")
    os.environ["LOCAL_VLLM_URL"] = server_url
    os.environ["LOCAL_VLLM_MODEL"] = "topo-planner"

    base_health_url = server_url.replace("/v1", "")
    health_before = query_server_health(base_health_url)
    print(f"[*] Serving Base URL: {server_url}")
    print(f"[*] Server Health Status: {health_before.get('status', 'unknown')}")
    print(f"[*] Detected GPU: {health_before.get('gpu', 'Unknown')}")
    print(f"[*] VRAM Allocated: {health_before.get('vram_allocated_mb', 0)} MB")
    print(f"[*] VRAM Peak: {health_before.get('vram_peak_mb', 0)} MB\n")

    if health_before.get("status") != "healthy":
        print(f"[ERROR] Local inference server at '{server_url}' is not responding or unhealthy.")
        sys.exit(1)

    # 3 Benchmark Target Tasks
    tasks = [
        MATH_PLANNING_TASKS[0],      # math_01
        CODE_SYNTHESIS_TASKS[0],     # code_01
        DUOHIRE_A2A_TASKS[0],        # duohire_01
    ]

    router = HybridLLMRouter(mock_mode=False, timeout=60.0)
    task_results: List[Dict[str, Any]] = []

    print("-" * 70)
    print(f" Running 3 Benchmark Tasks with HybridLLMRouter(model_tier='local_vllm')")
    print("-" * 70)

    for idx, task in enumerate(tasks, 1):
        print(f"\n[{idx}/3] Task: {task.task_id} - '{task.title}' ({task.category})")
        t_start = time.perf_counter()

        response: LLMResponse = await router.generate(
            prompt=task.formatted_prompt,
            system_prompt="You are TopoPlanner, an expert autonomous multi-agent reasoning and algorithmic system.",
            model_tier="local_vllm",
            temperature=0.1,  # Low temperature for deterministic accuracy
            max_tokens=2048,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        acc_score = evaluate_task(task, response.content)
        throughput_tok_sec = (response.completion_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

        print(f"    - Latency:          {response.latency_ms:.2f} ms")
        print(f"    - Prompt Tokens:    {response.prompt_tokens}")
        print(f"    - Completion Tokens:{response.completion_tokens}")
        print(f"    - Throughput:       {throughput_tok_sec:.2f} tokens/sec")
        print(f"    - Accuracy Score:   {acc_score * 100:.1f}%")
        print(f"    - Content Preview:  {response.content[:100].replace(chr(10), ' ')}...")

        task_results.append({
            "task_id": task.task_id,
            "category": task.category,
            "title": task.title,
            "accuracy_score": acc_score,
            "latency_ms": round(response.latency_ms, 2),
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.prompt_tokens + response.completion_tokens,
            "throughput_tokens_sec": round(throughput_tok_sec, 2),
            "model_used": response.model,
            "tier": response.tier,
            "ground_truth": task.ground_truth,
            "output_preview": response.content[:300],
        })

    health_after = query_server_health(base_health_url)

    # Aggregate Telemetry
    total_latency_ms = sum(r["latency_ms"] for r in task_results)
    avg_latency_ms = total_latency_ms / len(task_results)
    total_completion_tokens = sum(r["completion_tokens"] for r in task_results)
    avg_throughput = (total_completion_tokens / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0
    overall_accuracy = (sum(r["accuracy_score"] for r in task_results) / len(task_results)) * 100.0

    print("\n" + "=" * 70)
    print(" Local Serving Benchmark Summary (NVIDIA RTX 5070 Ti sm_120)")
    print("=" * 70)
    print(f"[*] Tasks Evaluated:        {len(task_results)}")
    print(f"[*] Overall Accuracy:       {overall_accuracy:.1f}%")
    print(f"[*] Average Latency:        {avg_latency_ms:.2f} ms")
    print(f"[*] Aggregate Throughput:   {avg_throughput:.2f} tokens/sec")
    print(f"[*] Total Completion Tokens:{total_completion_tokens}")
    print(f"[*] Peak VRAM Allocated:    {health_after.get('vram_allocated_mb', 0)} MB")
    print(f"[*] Peak VRAM Reserved:     {health_after.get('vram_peak_mb', 0)} MB")
    print("=" * 70 + "\n")

    serving_log = {
        "status": "completed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_name": health_after.get("gpu", "NVIDIA GeForce RTX 5070 Ti Laptop GPU"),
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_model": "results/qlora-adapter (TopoAgent Distillation)",
        "serving_engine": "PyTorch 2.11.0+cu128 SDPA Native / FastAPI OpenAI Spec",
        "endpoint_url": server_url,
        "vram_allocated_mb": health_after.get("vram_allocated_mb", 0),
        "vram_peak_mb": health_after.get("vram_peak_mb", 0),
        "total_tasks_evaluated": len(task_results),
        "overall_accuracy_percent": round(overall_accuracy, 2),
        "average_latency_ms": round(avg_latency_ms, 2),
        "average_throughput_tokens_sec": round(avg_throughput, 2),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in task_results),
        "total_completion_tokens": total_completion_tokens,
        "task_results": task_results,
    }

    log_path = resolve_project_path("results/local_vllm_serving_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(serving_log, f, indent=2, ensure_ascii=False)
    print(f"[+] Local inference serving log exported to '{log_path}'.")


if __name__ == "__main__":
    asyncio.run(main())
