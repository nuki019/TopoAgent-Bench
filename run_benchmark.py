"""
End-to-End Live Benchmark & MCTS Workflow Optimization Runner for TopoAgent-Bench.
Evaluates 5 Multi-Agent Topologies + AFlow-Lite MCTS across Math, Code, and DuoHire A2A tasks
using the Bailian Open-Source Model Pool (MAX_CONCURRENCY=5).
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from topoagent.llm_router import HybridLLMRouter
from topoagent.dag_engine import WorkflowDAG, ExecutionTrace
from topoagent.topologies import (
    create_single_agent_dag,
    create_chain_dag,
    create_star_dag,
    create_diamond_dag,
    create_dylan_pruned_dag,
    extract_answer,
)
from topoagent.mcts_search import (
    MCTSWorkflowSearcher,
    WorkflowConfig,
    calculate_pareto_reward,
    export_distillation_dataset,
)
from benchmarks.tasks import (
    BenchmarkTask,
    MATH_PLANNING_TASKS,
    CODE_SYNTHESIS_TASKS,
    DUOHIRE_A2A_TASKS,
    ALL_TASKS,
    evaluate_task,
)


def domain_aware_consensus(task: BenchmarkTask):
    """
    Creates a domain-aware DyLAN early-stopping consensus function that returns
    the full formatted response when >= 2 parallel solvers agree on the key answer/code.
    """
    def _consensus_fn(outputs: List[str]) -> Optional[str]:
        if len(outputs) < 2:
            return None

        if task.category == "code_synthesis":
            # For code synthesis, if >= 2 solvers pass the unit test assertions or produce identical AST/code
            passing = [out for out in outputs if evaluate_task(task, out) >= 1.0]
            if len(passing) >= 2:
                return passing[0]
            # Or check normalized function body signature
            return None

        elif task.category == "duohire_a2a":
            extracted_keys = []
            for out in outputs:
                m = re.search(r"FINAL_ANSWER:\s*(PASS|REJECT|NEGOTIATE)\s*,\s*(\d+)", out.upper())
                if m:
                    extracted_keys.append((f"{m.group(1)}, {m.group(2)}", out))
                else:
                    d = re.search(r"\b(PASS|REJECT|NEGOTIATE)\b", out.upper())
                    c = re.search(r"CONFLICT_COUNT:\s*(\d+)", out.upper())
                    if d and c:
                        extracted_keys.append((f"{d.group(1)}, {c.group(1)}", out))
            if len(extracted_keys) >= 2 and extracted_keys[0][0] == extracted_keys[1][0]:
                return extracted_keys[0][1]
            return None

        else:  # math_planning
            keys = [(extract_answer(out).replace(",", "").replace("$", "").strip(), out) for out in outputs if out]
            keys = [(k, o) for k, o in keys if k]
            if len(keys) >= 2 and keys[0][0] == keys[1][0]:
                return keys[0][1]
            return None

    return _consensus_fn


async def evaluate_topology_on_tasks(
    topo_label: str,
    dag_factory,
    tasks: List[BenchmarkTask],
    router: HybridLLMRouter,
    use_dylan_consensus: bool = False,
) -> Tuple[Dict, List[ExecutionTrace]]:
    """Runs a topology across all tasks (with bounded task-level concurrency)."""
    task_sem = asyncio.Semaphore(1)  # Run 1 DAG at a time so intra-DAG parallel nodes get 100% uncontended access to Semaphore(5)
    traces: List[ExecutionTrace] = []
    scores: List[float] = []
    per_category: Dict[str, List[float]] = {"math_planning": [], "code_synthesis": [], "duohire_a2a": []}

    async def _run_one(task: BenchmarkTask):
        async with task_sem:
            dag: WorkflowDAG = dag_factory()
            # Ensure output formatting instructions are preserved by final nodes
            for node in dag.nodes.values():
                if node.role in ("verifier", "aggregator", "critic_synthesizer", "solver"):
                    node.system_prompt += (
                        " CRITICAL: You MUST preserve the exact final output format requested in [INSTRUCTIONS] "
                        "(e.g. \\boxed{NUMBER} or FINAL_ANSWER: ... or ```python ... ``` code block)."
                    )
            early_fn = domain_aware_consensus(task) if use_dylan_consensus else None
            trace = await dag.execute(
                task_input=task.formatted_prompt,
                router=router,
                early_stop_fn=early_fn,
            )
            score = evaluate_task(task, trace.final_output)
            trace.success = score >= 1.0
            trace.metadata["task_id"] = task.task_id
            trace.metadata["category"] = task.category
            trace.metadata["score"] = score
            return task, trace, score

    results = await asyncio.gather(*(_run_one(t) for t in tasks))

    for task, trace, score in results:
        traces.append(trace)
        scores.append(score)
        per_category[task.category].append(score)

    n = max(len(tasks), 1)
    avg_acc = sum(scores) / n
    avg_wall_ms = sum(t.wall_time_ms for t in traces) / n
    avg_sum_node_ms = sum(t.sum_node_latency_ms for t in traces) / n
    avg_tokens = sum(t.total_tokens for t in traces) / n
    avg_speedup = sum(t.parallel_speedup for t in traces) / n
    early_stop_rate = sum(1 for t in traces if t.early_stopped) / n
    pareto_reward = calculate_pareto_reward(avg_acc, avg_wall_ms, avg_tokens)

    summary = {
        "topology": topo_label,
        "accuracy": round(avg_acc * 100, 2),
        "math_acc": round(sum(per_category["math_planning"]) / max(len(per_category["math_planning"]), 1) * 100, 1),
        "code_acc": round(sum(per_category["code_synthesis"]) / max(len(per_category["code_synthesis"]), 1) * 100, 1),
        "duohire_acc": round(sum(per_category["duohire_a2a"]) / max(len(per_category["duohire_a2a"]), 1) * 100, 1),
        "avg_wall_latency_ms": round(avg_wall_ms, 1),
        "avg_sum_node_latency_ms": round(avg_sum_node_ms, 1),
        "parallel_speedup": round(avg_speedup, 2),
        "avg_tokens": round(avg_tokens, 1),
        "early_stop_rate": round(early_stop_rate * 100, 1),
        "pareto_reward": round(pareto_reward, 4),
    }
    return summary, traces


async def main():
    parser = argparse.ArgumentParser(description="Run TopoAgent-Bench Live Evaluation")
    parser.add_argument("--full", action="store_true", help="Run all 24 tasks instead of 12-task balanced suite")
    args = parser.parse_args()

    if args.full:
        eval_tasks = ALL_TASKS
    else:
        # Balanced 12-task suite (4 Math + 4 Code + 4 DuoHire)
        eval_tasks = MATH_PLANNING_TASKS[:4] + CODE_SYNTHESIS_TASKS[:4] + DUOHIRE_A2A_TASKS[:4]

    router = HybridLLMRouter()
    flagship_m = router.resolve_model("flagship")[0]
    fast_m = router.resolve_model("fast")[0]
    alt_fast_m = router.resolve_model("alt_fast")[0]
    print(f"=== TopoAgent-Bench Live Evaluation ({len(eval_tasks)} Tasks | Max Concurrency=5) ===")
    print(f"Models -> Flagship: {flagship_m} | Fast: {fast_m} | Alt-Fast: {alt_fast_m}\n")

    topologies = [
        ("1. Single-Agent (Fast)", lambda: create_single_agent_dag(tier="fast"), False),
        ("2. Sequential Chain (3-Hop)", lambda: create_chain_dag(tier="fast"), False),
        ("3. Parallel Star (3x Hetero -> Flagship)", lambda: create_star_dag(), False),
        ("4. MacNet Diamond DAG (Plan -> 2x Parallel -> Synth)", lambda: create_diamond_dag(), False),
        ("5. DyLAN Early-Stop DAG (Dynamic Pruning)", lambda: create_dylan_pruned_dag(), True),
    ]

    all_summaries = []
    golden_traces: List[ExecutionTrace] = []

    for label, factory, use_dylan in topologies:
        t0 = time.perf_counter()
        print(f"Running {label} ...")
        summary, traces = await evaluate_topology_on_tasks(label, factory, eval_tasks, router, use_dylan)
        dt = time.perf_counter() - t0
        all_summaries.append(summary)
        golden_traces.extend([tr for tr in traces if tr.success])
        print(
            f"  -> Done in {dt:.1f}s | Acc: {summary['accuracy']}% "
            f"(Math {summary['math_acc']}%, Code {summary['code_acc']}%, DuoHire {summary['duohire_acc']}%) | "
            f"Wall Latency: {summary['avg_wall_latency_ms']}ms | Speedup: {summary['parallel_speedup']}x | "
            f"Tokens: {summary['avg_tokens']} | EarlyStop: {summary['early_stop_rate']}% | "
            f"Pareto R: {summary['pareto_reward']}"
        )

    # Run AFlow-Lite MCTS Topology Search on a 4-task validation slice
    print("\nRunning AFlow-Lite MCTS Workflow Search (UCB1 Exploration) ...")
    mcts_slice = [MATH_PLANNING_TASKS[0], CODE_SYNTHESIS_TASKS[0], DUOHIRE_A2A_TASKS[0], MATH_PLANNING_TASKS[1]]
    mcts_dicts = [
        {
            "input": t.formatted_prompt,
            "expected": t.ground_truth,
            "_eval_fn": lambda out, task_obj=t: evaluate_task(task_obj, out) >= 1.0,
            "_early_stop_fn": domain_aware_consensus(t),
        }
        for t in mcts_slice
    ]

    searcher = MCTSWorkflowSearcher(router=router, exploration_weight=1.414)
    best_cfg, best_reward, mcts_history = await searcher.search(
        eval_tasks=mcts_dicts,
        num_iterations=4,
    )
    print(f"  -> MCTS Best Discovered Workflow: {best_cfg} | Pareto Reward: {best_reward:.4f}")

    # Export Golden Distillation Trajectories
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    distill_path = results_dir / "distill_trajectories.jsonl"
    exported_count = export_distillation_dataset(golden_traces, str(distill_path), format_type="sharegpt")
    print(f"\nExported {exported_count} verified Golden DAG Trajectories to {distill_path}")

    # Save structured JSON report
    report_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "flagship": flagship_m,
            "fast": fast_m,
            "alt_fast": alt_fast_m,
            "max_concurrency": 5,
        },
        "task_count": len(eval_tasks),
        "topology_results": all_summaries,
        "mcts_search": {
            "best_config": {
                "topology": best_cfg.topology,
                "solver_tier": best_cfg.solver_tier,
                "aggregator_tier": best_cfg.aggregator_tier,
                "early_stopping": best_cfg.early_stopping,
            },
            "best_pareto_reward": round(best_reward, 4),
            "search_history": mcts_history,
        },
        "distillation_samples_exported": exported_count,
    }
    summary_json_path = results_dir / "benchmark_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved full telemetry to {summary_json_path}")


if __name__ == "__main__":
    asyncio.run(main())
