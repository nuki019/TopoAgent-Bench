"""
TopoAgent-Bench: High-Performance Multi-Agent Workflow Engine & Benchmark.
"""

from topoagent.llm_router import HybridLLMRouter, LLMResponse, load_dotenv
from topoagent.dag_engine import AgentNode, WorkflowDAG, ExecutionTrace
from topoagent.topologies import (
    create_single_agent_dag,
    create_chain_dag,
    create_star_dag,
    create_diamond_dag,
    create_dylan_pruned_dag,
    majority_vote_consensus,
    exact_match_consensus,
)
from topoagent.mcts_search import (
    MCTSWorkflowSearcher,
    WorkflowConfig,
    export_distillation_dataset,
)

__version__ = "0.1.0"
__all__ = [
    "HybridLLMRouter",
    "LLMResponse",
    "load_dotenv",
    "AgentNode",
    "WorkflowDAG",
    "ExecutionTrace",
    "create_single_agent_dag",
    "create_chain_dag",
    "create_star_dag",
    "create_diamond_dag",
    "create_dylan_pruned_dag",
    "majority_vote_consensus",
    "exact_match_consensus",
    "MCTSWorkflowSearcher",
    "WorkflowConfig",
    "export_distillation_dataset",
]
