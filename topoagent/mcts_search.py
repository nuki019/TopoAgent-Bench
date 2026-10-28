"""
AFlow-Lite MCTS Workflow Optimizer and Distillation Dataset Exporter.
"""

import math
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable

from topoagent.llm_router import HybridLLMRouter
from topoagent.dag_engine import WorkflowDAG, ExecutionTrace
from topoagent.topologies import (
    create_single_agent_dag,
    create_chain_dag,
    create_star_dag,
    create_diamond_dag,
    create_dylan_pruned_dag,
    extract_answer,
    majority_vote_consensus,
)


@dataclass
class WorkflowConfig:
    """Hyperparameter and structural configuration for a multi-agent workflow."""
    topology: str = "single"  # single, chain, star, diamond, dylan_pruned
    solver_tier: str = "fast"  # fast, alt_fast, flagship, local_vllm
    aggregator_tier: str = "flagship"  # fast, alt_fast, flagship
    early_stopping: bool = True
    temperature: float = 0.3

    def to_dag(self) -> WorkflowDAG:
        """Instantiates the concrete WorkflowDAG for this configuration."""
        top = self.topology.lower()
        if top == "single":
            dag = create_single_agent_dag(tier=self.solver_tier)
        elif top == "chain":
            dag = create_chain_dag(tier=self.solver_tier)
        elif top == "star":
            dag = create_star_dag(
                fast_tier=self.solver_tier,
                alt_fast_tier="alt_fast" if self.solver_tier != "alt_fast" else "fast",
                flagship_tier=self.aggregator_tier,
            )
        elif top == "diamond":
            dag = create_diamond_dag(
                fast_tier=self.solver_tier,
                alt_fast_tier="alt_fast" if self.solver_tier != "alt_fast" else "fast",
                flagship_tier=self.aggregator_tier,
            )
        elif top == "dylan_pruned":
            dag = create_dylan_pruned_dag(
                fast_tier=self.solver_tier,
                alt_fast_tier="alt_fast" if self.solver_tier != "alt_fast" else "fast",
                flagship_tier=self.aggregator_tier,
            )
            dag.early_stop_enabled = self.early_stopping
        else:
            dag = create_single_agent_dag(tier=self.solver_tier)

        for node in dag.nodes.values():
            node.temperature = self.temperature

        return dag


class MCTSNode:
    """A search tree node in the MCTS workflow configuration space."""

    def __init__(
        self,
        config: WorkflowConfig,
        parent: Optional["MCTSNode"] = None,
        action_taken: str = "root",
    ):
        self.config = config
        self.parent = parent
        self.action_taken = action_taken
        self.children: List["MCTSNode"] = []
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.best_reward: float = -float("inf")
        self.untried_mutations: Optional[List[WorkflowConfig]] = None

    def ucb1(self, c: float = 1.414) -> float:
        """
        Computes Upper Confidence Bound (UCB1):
        Q/N + c * sqrt(ln(N_parent) / N).
        """
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.total_reward / self.visits
        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        """Check if all potential mutations have been spawned as children."""
        if self.untried_mutations is None:
            return False
        return len(self.untried_mutations) == 0


def calculate_pareto_reward(accuracy: float, avg_latency_ms: float, avg_tokens: float) -> float:
    """
    Computes AFlow-Lite Pareto reward balancing accuracy against latency and token budget:
    R = accuracy - 0.05 * (avg_latency_ms / 1000.0) - 0.05 * (avg_tokens / 1000.0)
    """
    latency_penalty = 0.05 * (avg_latency_ms / 1000.0)
    token_penalty = 0.05 * (avg_tokens / 1000.0)
    return accuracy - latency_penalty - token_penalty


def export_distillation_dataset(
    traces: List[ExecutionTrace],
    output_path: str,
    format_type: str = "sharegpt",
    min_tokens: int = 1,
) -> int:
    """
    Saves high-quality DAG planning and execution traces in ShareGPT or Alpaca
    JSONL format for fine-tuning smaller models (e.g. QLoRA).
    Returns the number of exported samples.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(target, "w", encoding="utf-8") as f:
        for trace in traces:
            if not trace.final_output or trace.total_tokens < min_tokens or not trace.success:
                continue

            # Build intermediate rationale from nodes if available
            rationale_steps = []
            for nid, node in trace.node_results.items():
                if not node.skipped and node.output:
                    rationale_steps.append(f"[{node.role}]: {node.output}")
            composite_reasoning = "\n\n".join(rationale_steps)

            if format_type.lower() == "sharegpt":
                entry = {
                    "conversations": [
                        {"from": "human", "value": trace.task_input},
                        {
                            "from": "gpt",
                            "value": f"{composite_reasoning}\n\nFinal Verified Answer: {trace.final_output}"
                            if composite_reasoning else trace.final_output,
                        },
                    ],
                    "metadata": {
                        "dag_name": trace.dag_name,
                        "wall_time_ms": round(trace.wall_time_ms, 2),
                        "total_tokens": trace.total_tokens,
                        "parallel_speedup": round(trace.parallel_speedup, 2),
                        "early_stopped": trace.early_stopped,
                    },
                }
            else:  # alpaca format
                entry = {
                    "instruction": "Solve the reasoning problem thoroughly step-by-step and produce the final verified answer.",
                    "input": trace.task_input,
                    "output": f"{composite_reasoning}\n\nFinal Verified Answer: {trace.final_output}"
                    if composite_reasoning else trace.final_output,
                    "metadata": {
                        "dag_name": trace.dag_name,
                        "wall_time_ms": round(trace.wall_time_ms, 2),
                        "total_tokens": trace.total_tokens,
                    },
                }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    return count


class MCTSWorkflowSearcher:
    """
    AFlow-Lite MCTS Optimizer searching over multi-agent workflow topologies,
    model tier allocations, and DyLAN early-stopping thresholds.
    """

    TOPOLOGIES = ["single", "chain", "star", "diamond", "dylan_pruned"]
    SOLVER_TIERS = ["fast", "alt_fast", "flagship"]
    AGGREGATOR_TIERS = ["flagship", "fast"]

    def __init__(
        self,
        router: Optional[HybridLLMRouter] = None,
        exploration_weight: float = 1.414,
    ):
        self.router = router or HybridLLMRouter()
        self.c = exploration_weight
        self.root = MCTSNode(config=WorkflowConfig(topology="single", solver_tier="fast"))
        self.all_evaluated_traces: List[ExecutionTrace] = []
        self.history: List[Dict[str, Any]] = []

    def get_mutations(self, config: WorkflowConfig) -> List[WorkflowConfig]:
        """Generates valid mutated successor configurations."""
        mutations = []
        # Mutate topology
        for top in self.TOPOLOGIES:
            if top != config.topology:
                mutations.append(
                    WorkflowConfig(
                        topology=top,
                        solver_tier=config.solver_tier,
                        aggregator_tier=config.aggregator_tier,
                        early_stopping=config.early_stopping,
                        temperature=config.temperature,
                    )
                )

        # Mutate solver tier
        for tier in self.SOLVER_TIERS:
            if tier != config.solver_tier:
                mutations.append(
                    WorkflowConfig(
                        topology=config.topology,
                        solver_tier=tier,
                        aggregator_tier=config.aggregator_tier,
                        early_stopping=config.early_stopping,
                        temperature=config.temperature,
                    )
                )

        # Mutate early stopping toggle for dylan
        if config.topology == "dylan_pruned":
            mutations.append(
                WorkflowConfig(
                    topology=config.topology,
                    solver_tier=config.solver_tier,
                    aggregator_tier=config.aggregator_tier,
                    early_stopping=not config.early_stopping,
                    temperature=config.temperature,
                )
            )

        return mutations

    async def evaluate_config(
        self,
        config: WorkflowConfig,
        eval_tasks: List[Dict[str, Any]],
        eval_fn: Optional[Callable[[str, str], bool]] = None,
    ) -> Tuple[float, Dict[str, Any], List[ExecutionTrace]]:
        """
        Evaluates a workflow configuration over a mini-batch of tasks,
        computing Pareto reward = accuracy - 0.05*(latency/1000) - 0.05*(tokens/1000).
        """
        dag = config.to_dag()
        traces: List[ExecutionTrace] = []
        correct_count = 0

        for task in eval_tasks:
            task_input = task["input"]
            expected = task.get("expected", "")
            early_stop_fn = (
                task.get("_early_stop_fn", majority_vote_consensus)
                if config.early_stopping
                else None
            )
            trace = await dag.execute(
                task_input=task_input,
                router=self.router,
                early_stop_fn=early_stop_fn,
            )
            traces.append(trace)

            # Evaluate correctness
            if "_eval_fn" in task and callable(task["_eval_fn"]):
                is_correct = bool(task["_eval_fn"](trace.final_output))
            else:
                pred_answer = extract_answer(trace.final_output)
                exp_answer = extract_answer(str(expected))
                if eval_fn:
                    is_correct = eval_fn(pred_answer, exp_answer)
                else:
                    is_correct = (pred_answer.lower() == exp_answer.lower()) or (exp_answer in pred_answer)
            if is_correct:
                correct_count += 1

        n_tasks = max(len(eval_tasks), 1)
        accuracy = correct_count / n_tasks
        avg_latency_ms = sum(t.wall_time_ms for t in traces) / n_tasks
        avg_tokens = sum(t.total_tokens for t in traces) / n_tasks

        pareto_reward = calculate_pareto_reward(accuracy, avg_latency_ms, avg_tokens)

        metrics = {
            "config": asdict(config),
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency_ms,
            "avg_tokens": avg_tokens,
            "pareto_reward": pareto_reward,
            "task_count": n_tasks,
        }

        self.all_evaluated_traces.extend(traces)
        return pareto_reward, metrics, traces

    def select(self, node: MCTSNode) -> MCTSNode:
        """Traverse tree using UCB1 until finding an unexpanded node or leaf."""
        current = node
        while current.children:
            if not current.is_fully_expanded():
                return current
            # Choose child with max UCB1
            current = max(current.children, key=lambda c: c.ucb1(self.c))
        return current

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expands a child node using untried mutations."""
        if node.untried_mutations is None:
            node.untried_mutations = self.get_mutations(node.config)

        if node.untried_mutations:
            mutation = node.untried_mutations.pop(0)
            child = MCTSNode(
                config=mutation,
                parent=node,
                action_taken=f"{mutation.topology}_{mutation.solver_tier}",
            )
            node.children.append(child)
            return child
        return node

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Propagates reward up the tree to the root."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            if reward > current.best_reward:
                current.best_reward = reward
            current = current.parent

    async def search(
        self,
        eval_tasks: List[Dict[str, Any]],
        num_iterations: int = 5,
        eval_fn: Optional[Callable[[str, str], bool]] = None,
    ) -> Tuple[WorkflowConfig, float, List[Dict[str, Any]]]:
        """
        Executes MCTS search over workflow configurations.
        Returns (best_config, best_reward, history).
        """
        # Initial evaluation of root
        root_reward, root_metrics, _ = await self.evaluate_config(self.root.config, eval_tasks, eval_fn)
        self.backpropagate(self.root, root_reward)
        self.history.append(root_metrics)

        best_config = self.root.config
        best_reward = root_reward

        for _ in range(num_iterations):
            selected = self.select(self.root)
            expanded = self.expand(selected)
            reward, metrics, _ = await self.evaluate_config(expanded.config, eval_tasks, eval_fn)
            self.backpropagate(expanded, reward)
            self.history.append(metrics)

            if reward > best_reward:
                best_reward = reward
                best_config = expanded.config

        return best_config, best_reward, self.history

    def export_traces(self, output_path: str, format_type: str = "sharegpt") -> int:
        """Exports all evaluated execution traces to disk."""
        return export_distillation_dataset(
            self.all_evaluated_traces,
            output_path=output_path,
            format_type=format_type,
        )
