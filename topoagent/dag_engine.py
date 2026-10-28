"""
Topological Execution Engine with Layering, Parallelism, and DyLAN Consensus Pruning.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict

from topoagent.llm_router import HybridLLMRouter, LLMResponse


@dataclass
class AgentNode:
    """An individual agent execution node in the workflow graph."""
    id: str
    role: str
    system_prompt: str = ""
    task_template: str = "{input}"
    deps: List[str] = field(default_factory=list)
    model_tier: str = "fast"
    temperature: float = 0.7
    max_tokens: int = 2048
    output: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    skipped: bool = False
    error: Optional[str] = None

    def reset(self) -> None:
        """Reset execution state for a new run."""
        self.output = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_ms = 0.0
        self.skipped = False
        self.error = None

    def construct_prompt(self, task_input: str, upstream_outputs: Dict[str, str]) -> str:
        """
        Builds the prompt by substituting upstream inputs and contextual deps.
        """
        fmt_kwargs: Dict[str, Any] = {
            "input": task_input,
            **upstream_outputs,
        }

        # Build human-readable deps summary if multiple deps exist
        deps_summary = "\n\n".join(
            f"--- [Output from {dep_id}] ---\n{upstream_outputs.get(dep_id, '')}"
            for dep_id in self.deps
            if dep_id in upstream_outputs
        )
        fmt_kwargs["deps_summary"] = deps_summary

        prompt_str = self.task_template
        # Attempt safe formatting
        try:
            prompt_str = self.task_template.format(**fmt_kwargs)
        except Exception:
            # Fallback simple replacement
            if "{input}" in prompt_str:
                prompt_str = prompt_str.replace("{input}", task_input)

        # If deps exist and were not explicitly interpolated into template, append them
        has_direct_dep_ref = any(f"{{{dep}}}" in self.task_template for dep in self.deps)
        if self.deps and not has_direct_dep_ref and "{deps_summary}" not in self.task_template:
            if deps_summary:
                prompt_str = f"{prompt_str}\n\nContext from predecessor agents:\n{deps_summary}"

        return prompt_str


@dataclass
class ExecutionTrace:
    """Detailed telemetry and outputs of a DAG execution run."""
    dag_name: str
    task_input: str
    final_output: str
    node_results: Dict[str, AgentNode]
    wall_time_ms: float
    critical_path_ms: float
    sum_node_latency_ms: float
    total_tokens: int
    parallel_speedup: float
    early_stopped: bool = False
    consensus_answer: Optional[str] = None
    layers_executed: int = 0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sum_node_latency(self) -> float:
        """Alias for sum_node_latency_ms."""
        return self.sum_node_latency_ms


class WorkflowDAG:
    """
    Kahn-based DAG Execution Engine supporting topological generation layering,
    concurrent layer execution, and DyLAN dynamic early-stopping pruning.
    """

    def __init__(self, name: str = "WorkflowDAG"):
        self.name = name
        self.nodes: Dict[str, AgentNode] = {}
        self.early_stop_enabled: bool = False

    def add_node(self, node: AgentNode) -> "WorkflowDAG":
        """Add a node to the workflow graph."""
        self.nodes[node.id] = node
        return self

    def validate_acyclic(self) -> None:
        """Validates that the graph is a directed acyclic graph (DAG). Raises ValueError on cycle."""
        self.topological_layers()

    def topological_layers(self) -> List[List[AgentNode]]:
        """
        Uses Kahn's algorithm to partition nodes into concurrent topological layers.
        Nodes within the same layer have all their dependencies satisfied by previous layers.
        Raises ValueError if a cycle is detected or dependencies are missing.
        """
        # Validate dependencies exist in DAG
        for nid, node in self.nodes.items():
            for dep in node.deps:
                if dep not in self.nodes:
                    raise ValueError(f"Node '{nid}' references undefined dependency '{dep}'")

        in_degrees: Dict[str, int] = {nid: len(node.deps) for nid, node in self.nodes.items()}
        dependents: Dict[str, List[str]] = defaultdict(list)
        for nid, node in self.nodes.items():
            for dep in node.deps:
                dependents[dep].append(nid)

        current_layer_ids = [nid for nid, deg in in_degrees.items() if deg == 0]
        layers: List[List[AgentNode]] = []
        visited_count = 0

        while current_layer_ids:
            # Deterministic ordering by ID within layer
            current_layer_ids.sort()
            layer_nodes = [self.nodes[nid] for nid in current_layer_ids]
            layers.append(layer_nodes)
            visited_count += len(current_layer_ids)

            next_layer_ids = []
            for nid in current_layer_ids:
                for child_id in dependents[nid]:
                    in_degrees[child_id] -= 1
                    if in_degrees[child_id] == 0:
                        next_layer_ids.append(child_id)
            current_layer_ids = next_layer_ids

        if visited_count < len(self.nodes):
            raise ValueError(f"Cycle detected in DAG: '{self.name}'. Processed {visited_count}/{len(self.nodes)} nodes.")

        return layers

    async def _execute_single_node(
        self,
        node: AgentNode,
        task_input: str,
        router: HybridLLMRouter,
    ) -> None:
        """Executes one node using upstream outputs and LLM router."""
        if node.skipped:
            return

        upstream_outputs = {
            dep_id: self.nodes[dep_id].output or ""
            for dep_id in node.deps
            if dep_id in self.nodes
        }

        prompt = node.construct_prompt(task_input, upstream_outputs)
        try:
            resp: LLMResponse = await router.generate(
                prompt=prompt,
                system_prompt=node.system_prompt,
                model_tier=node.model_tier,
                temperature=node.temperature,
                max_tokens=node.max_tokens,
            )
            node.output = resp.content
            node.prompt_tokens = resp.prompt_tokens
            node.completion_tokens = resp.completion_tokens
            node.latency_ms = resp.latency_ms
        except Exception as e:
            node.error = str(e)
            node.output = f"[Error in {node.id}: {e}]"

    async def execute(
        self,
        task_input: str,
        router: HybridLLMRouter,
        early_stop_fn: Optional[Callable[[List[str]], Optional[str]]] = None,
    ) -> ExecutionTrace:
        """
        Executes DAG layer by layer, concurrently evaluating nodes within each layer.
        Implements DyLAN early stopping: if early_stop_fn returns a non-None consensus
        on a layer with >= 2 parallel nodes, remaining layers are pruned (skipped).
        """
        # Validate acyclic structure and get topological layers
        layers = self.topological_layers()

        # Reset all nodes
        for node in self.nodes.values():
            node.reset()

        start_wall_time = time.perf_counter()
        early_stopped = False
        consensus_answer: Optional[str] = None
        executed_layers_count = 0
        final_output = ""

        effective_early_stop = early_stop_fn if (early_stop_fn is not None or self.early_stop_enabled) else None

        for layer_idx, layer in enumerate(layers):
            active_nodes = [node for node in layer if not node.skipped]
            if not active_nodes:
                continue

            executed_layers_count += 1

            # Concurrent execution of all ready nodes in current layer
            tasks = [self._execute_single_node(node, task_input, router) for node in active_nodes]
            await asyncio.gather(*tasks)

            # Check DyLAN early-stopping consensus if parallel nodes >= 2
            if effective_early_stop is not None and len(active_nodes) >= 2:
                valid_outputs = [node.output for node in active_nodes if node.output is not None and not node.error]
                if len(valid_outputs) >= 2:
                    agreed_val = effective_early_stop(valid_outputs)
                    if agreed_val is not None:
                        # Consensus reached! Early prune remaining downstream layers
                        early_stopped = True
                        consensus_answer = agreed_val
                        final_output = agreed_val

                        # Mark all nodes in subsequent layers as skipped
                        for downstream_layer in layers[layer_idx + 1:]:
                            for down_node in downstream_layer:
                                down_node.skipped = True
                        break

            # If this is the last executed layer and no early stop, set final output
            if not early_stopped:
                non_empty = [node.output for node in active_nodes if node.output]
                if non_empty:
                    final_output = non_empty[-1]

        total_wall_time_ms = (time.perf_counter() - start_wall_time) * 1000.0
        sum_node_latency_ms = sum(n.latency_ms for n in self.nodes.values() if not n.skipped)
        total_tokens = sum((n.prompt_tokens + n.completion_tokens) for n in self.nodes.values() if not n.skipped)

        speedup = (sum_node_latency_ms / total_wall_time_ms) if total_wall_time_ms > 0 else 1.0

        return ExecutionTrace(
            dag_name=self.name,
            task_input=task_input,
            final_output=final_output,
            node_results=self.nodes.copy(),
            wall_time_ms=total_wall_time_ms,
            critical_path_ms=sum_node_latency_ms,
            sum_node_latency_ms=sum_node_latency_ms,
            total_tokens=total_tokens,
            parallel_speedup=speedup,
            early_stopped=early_stopped,
            consensus_answer=consensus_answer,
            layers_executed=executed_layers_count,
            success=all(n.error is None for n in self.nodes.values() if not n.skipped),
        )
