"""Benchmark Datasets and Deterministic Evaluators for TopoAgent-Bench.

Aligned with Cloud & Distributed Systems Agent R&D requirements:
1. MATH_PLANNING_TASKS (10 tasks): DAG scheduling, network flow, combinatorics, queueing & Markov analysis.
2. CODE_SYNTHESIS_TASKS (8 tasks): Algorithmic Python coding (DAGs, constrained shortest paths, interval partitioning, tree DP).
3. DUOHIRE_A2A_TASKS (6 tasks): Complex multi-constraint bilateral recruitment decision making.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =====================================================================
# Task Data Structures
# =====================================================================

@dataclass
class BenchmarkTask:
    """Represents a single deterministic benchmark evaluation task."""
    task_id: str
    category: str  # "math_planning", "code_synthesis", "duohire_a2a"
    title: str
    prompt: str
    ground_truth: str
    test_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def formatted_prompt(self) -> str:
        """Appends the category-specific formatting instructions to the prompt."""
        if self.category == "math_planning":
            suffix = (
                "\n\n[INSTRUCTIONS]\n"
                "Provide your clear step-by-step reasoning. At the end of your response, "
                "you MUST provide your final numeric answer in either '\\boxed{NUMBER}' or "
                "'FINAL_ANSWER: NUMBER' format (e.g., FINAL_ANSWER: 42 or \\boxed{42})."
            )
        elif self.category == "code_synthesis":
            suffix = (
                "\n\n[INSTRUCTIONS]\n"
                "Write a complete, self-contained Python function matching the exact signature specified. "
                "Enclose your Python solution inside a ```python ... ``` code block. "
                "Do not use external third-party libraries outside the Python standard library."
            )
        elif self.category == "duohire_a2a":
            suffix = (
                "\n\n[INSTRUCTIONS]\n"
                "Analyze the candidate profile against all hard and flexible job constraints. "
                "At the end of your response, output:\n"
                "FINAL_DECISION: PASS|REJECT|NEGOTIATE\n"
                "CONFLICT_COUNT: <integer>\n"
                "FINAL_ANSWER: <DECISION>, <CONFLICT_COUNT>"
            )
        else:
            suffix = "\n\nFINAL_ANSWER: <answer>"
        return f"{self.prompt}{suffix}"


# =====================================================================
# Deterministic Evaluators
# =====================================================================

def evaluate_math(prediction: str, ground_truth: str) -> float:
    """Deterministically evaluates a math reasoning prediction against ground truth.

    Supports \\boxed{...}, FINAL_ANSWER: ..., and trailing numeric statements.
    Tolerates float representations, percentages, and standard formatting differences.
    Returns 1.0 on exact match or within 1e-3 relative/absolute tolerance, else 0.0.
    """
    if not prediction:
        return 0.0

    extracted: Optional[str] = None

    # 1. Search for \boxed{...}
    boxed_match = re.search(r"\\boxed\{([^{}]+)\}", prediction)
    if boxed_match:
        extracted = boxed_match.group(1).strip()

    # 2. Search for FINAL_ANSWER: ...
    if not extracted:
        final_match = re.search(r"FINAL_ANSWER:\s*([^\n\r]+)", prediction, re.IGNORECASE)
        if final_match:
            extracted = final_match.group(1).strip()

    # 3. Fallback: Search for 'The answer is ...' or 'is ...'
    if not extracted:
        answer_is_match = re.search(r"(?:answer is|result is)\s*[:=]?\s*([$]?[-+]?\d+(?:\.\d+)?%?)", prediction, re.IGNORECASE)
        if answer_is_match:
            extracted = answer_is_match.group(1).strip()

    # 4. Fallback: Last number in the text
    if not extracted:
        all_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", prediction)
        if all_numbers:
            extracted = all_numbers[-1]

    if not extracted:
        return 0.0

    # Clean extracted string
    clean_p = extracted.replace("$", "").replace(",", "").strip()
    clean_gt = ground_truth.replace("$", "").replace(",", "").strip()

    # Remove % if present and handle percentage
    if clean_p.endswith("%"):
        clean_p = clean_p[:-1].strip()
    if clean_gt.endswith("%"):
        clean_gt = clean_gt[:-1].strip()

    # Direct string match
    if clean_p == clean_gt:
        return 1.0

    # Numeric comparison
    try:
        val_p = float(clean_p)
        val_gt = float(clean_gt)
        if abs(val_p - val_gt) <= 1e-3 or abs(val_p - val_gt) / (abs(val_gt) + 1e-9) <= 1e-3:
            return 1.0
        # Allow fraction representation (e.g. 0.852 vs 85.2%)
        if abs(val_p * 100 - val_gt) <= 1e-2 or abs(val_p - val_gt * 100) <= 1e-2:
            return 1.0
    except (ValueError, TypeError):
        pass

    return 0.0


def evaluate_code(prediction: str, test_code: Optional[str], timeout: float = 6.0) -> float:
    """Deterministically evaluates code synthesis using isolated subprocess execution.

    Extracts the python code block from prediction, appends test assertions,
    and runs in a sandboxed child python interpreter with timeout.
    Returns 1.0 if all assertions pass cleanly, 0.0 on error or timeout.
    """
    if not prediction or not test_code:
        return 0.0

    # Extract python code block
    code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", prediction, re.IGNORECASE)
    if code_blocks:
        user_code = code_blocks[-1].strip()
    else:
        # If no markdown fences, treat full prediction as raw python code
        user_code = prediction.strip()

    full_script = f"{user_code}\n\n# --- BENCHMARK ASSERTIONS ---\n{test_code}\nprint('BENCHMARK_SUCCESS')"

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and "BENCHMARK_SUCCESS" in result.stdout:
            return 1.0
    except (subprocess.TimeoutExpired, Exception):
        return 0.0

    return 0.0


def evaluate_duohire(prediction: str, ground_truth: str) -> float:
    """Deterministically evaluates bilateral DuoHire recruitment decisions.

    Ground truth format: '<DECISION>, <CONFLICT_COUNT>' e.g. 'PASS, 0' or 'REJECT, 3'.
    Returns 1.0 for perfect match (decision & conflict count),
            0.5 for correct decision with mismatched count,
            0.0 otherwise.
    """
    if not prediction:
        return 0.0

    # Parse ground truth
    gt_parts = [p.strip().upper() for p in ground_truth.split(",")]
    gt_decision = gt_parts[0]
    gt_count = int(gt_parts[1]) if len(gt_parts) > 1 and gt_parts[1].isdigit() else 0

    pred_upper = prediction.upper()

    # Search for structured format: FINAL_ANSWER: DECISION, COUNT
    fa_match = re.search(r"FINAL_ANSWER:\s*(PASS|REJECT|NEGOTIATE)\s*,\s*(\d+)", pred_upper)
    if fa_match:
        pred_decision = fa_match.group(1)
        pred_count = int(fa_match.group(2))
    else:
        # Search individual tags
        dec_match = re.search(r"FINAL_DECISION:\s*(PASS|REJECT|NEGOTIATE)", pred_upper)
        if not dec_match:
            dec_match = re.search(r"\b(PASS|REJECT|NEGOTIATE)\b", pred_upper)
        pred_decision = dec_match.group(1) if dec_match else None

        cnt_match = re.search(r"CONFLICT_COUNT:\s*(\d+)", pred_upper)
        pred_count = int(cnt_match.group(1)) if cnt_match else None

    if not pred_decision:
        return 0.0

    if pred_decision == gt_decision:
        if pred_count is not None and pred_count == gt_count:
            return 1.0
        return 0.5  # Partial credit for correct classification

    return 0.0


def evaluate_task(task: BenchmarkTask, prediction: str) -> float:
    """Unified benchmark evaluation router."""
    if task.category == "math_planning":
        return evaluate_math(prediction, task.ground_truth)
    elif task.category == "code_synthesis":
        return evaluate_code(prediction, task.test_code)
    elif task.category == "duohire_a2a":
        return evaluate_duohire(prediction, task.ground_truth)
    else:
        return 1.0 if task.ground_truth.strip() in prediction else 0.0


# =====================================================================
# Benchmark Suites (24 Tasks Total)
# =====================================================================

MATH_PLANNING_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        task_id="math_01",
        category="math_planning",
        title="DAG Critical Path & Slack Scheduling",
        prompt=(
            "Consider a project with 6 tasks T1, T2, T3, T4, T5, T6 having the following durations and dependencies:\n"
            "- T1 takes 4 hours (no dependencies).\n"
            "- T2 takes 5 hours (depends on T1).\n"
            "- T3 takes 7 hours (depends on T1).\n"
            "- T4 takes 6 hours (depends on T2).\n"
            "- T5 takes 4 hours (depends on both T2 and T3).\n"
            "- T6 takes 3 hours (depends on both T4 and T5).\n\n"
            "Assuming all tasks start as early as their prerequisites finish, what is the total project duration (the critical path length in hours)?"
        ),
        ground_truth="18",
        metadata={"domain": "dag_scheduling", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="math_02",
        category="math_planning",
        title="Network Flow Maximum Throughput",
        prompt=(
            "A distributed data processing pipeline routes data from a single Source S to a single Sink T via 3 worker nodes {W1, W2, W3} "
            "and 3 job aggregation nodes {J1, J2, J3}.\n"
            "The edge capacities are as follows:\n"
            "- From Source S: (S, W1) = 15, (S, W2) = 20, (S, W3) = 25.\n"
            "- From Workers to Jobs: (W1, J1) = 10, (W1, J2) = 10, (W2, J1) = 15, (W2, J3) = 15, (W3, J2) = 10, (W3, J3) = 20.\n"
            "- From Jobs to Sink T: (J1, T) = 18, (J2, T) = 16, (J3, T) = 22.\n\n"
            "All edges not mentioned have zero capacity. What is the maximum data throughput (max flow) from S to T?"
        ),
        ground_truth="56",
        metadata={"domain": "network_flow", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="math_03",
        category="math_planning",
        title="Combinatorial Server Rack Permutation",
        prompt=(
            "A cloud data center has 4 identical compute servers, 3 identical storage servers, and 2 identical high-power GPU accelerators.\n"
            "All 9 servers must be installed in a single rack containing 9 slots in a vertical line (slots 1 through 9).\n"
            "To prevent thermal hotspots, the 2 GPU accelerators must NEVER be placed in adjacent slots.\n\n"
            "How many distinct valid linear arrangements of these 9 servers are possible in the rack?"
        ),
        ground_truth="980",
        metadata={"domain": "combinatorics", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="math_04",
        category="math_planning",
        title="Topological Order Counting with Precedence Constraints",
        prompt=(
            "Consider a directed acyclic graph (DAG) with vertices {A, B, C, D, E, F} and the following directed edges:\n"
            "- A -> C\n"
            "- B -> C\n"
            "- C -> D\n"
            "- C -> E\n"
            "- D -> F\n"
            "- E -> F\n\n"
            "How many distinct valid topological sortings of the 6 vertices exist?"
        ),
        ground_truth="4",
        metadata={"domain": "graph_theory", "difficulty": "medium"},
    ),
    BenchmarkTask(
        task_id="math_05",
        category="math_planning",
        title="Multi-Stage Cloud Storage Tiering Optimization",
        prompt=(
            "A 120 TB dataset is stored over 30 days divided into three 10-day intervals (Days 1-10, Days 11-20, Days 21-30).\n"
            "Three storage tiers are available:\n"
            "- Hot Tier: $0.020 / TB / day storage cost, $0.00 / TB access (read) cost.\n"
            "- Cool Tier: $0.010 / TB / day storage cost, $0.01 / TB access (read) cost.\n"
            "- Archive Tier: $0.002 / TB / day storage cost, $0.05 / TB access (read) cost.\n\n"
            "The data read patterns are:\n"
            "- Days 1 to 10: 40 TB read per day.\n"
            "- Days 11 to 20: 10 TB read per day.\n"
            "- Days 21 to 30: 1 TB read per day.\n\n"
            "At the beginning of each 10-day interval, the data can be placed in any tier (tier transition cost is $0, and data must stay in one chosen tier throughout that 10-day interval).\n"
            "What is the minimum total cost (in USD) to store and access the data across all 30 days?"
        ),
        ground_truth="26.30",
        metadata={"domain": "resource_optimization", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="math_06",
        category="math_planning",
        title="Markov Packet Routing & Absorption Steps",
        prompt=(
            "A network packet moves through router states 0, 1, 2, 3, 4 where state 0 is Source and state 4 is Destination (absorbing state).\n"
            "The transition probabilities are:\n"
            "- From 0: always transitions to 1 (P(0 -> 1) = 1.0).\n"
            "- From 1: P(1 -> 2) = 0.6, P(1 -> 0) = 0.4.\n"
            "- From 2: P(2 -> 3) = 0.5, P(2 -> 1) = 0.5.\n"
            "- From 3: P(3 -> 4) = 0.8, P(3 -> 2) = 0.2.\n\n"
            "What is the expected total number of transition hops (steps) starting from state 0 until first reaching state 4?"
        ),
        ground_truth="10",
        metadata={"domain": "markov_chains", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="math_07",
        category="math_planning",
        title="Knapsack Bandwidth Utility Maximization",
        prompt=(
            "An edge router has an uplink capacity limit of 50 MBps. There are 5 available streaming telemetry flows:\n"
            "- Stream A: requires 12 MBps, provides utility score 24.\n"
            "- Stream B: requires 15 MBps, provides utility score 32.\n"
            "- Stream C: requires 20 MBps, provides utility score 45.\n"
            "- Stream D: requires 25 MBps, provides utility score 56.\n"
            "- Stream E: requires 8 MBps, provides utility score 18.\n\n"
            "Streams cannot be split (0/1 knapsack). What is the maximum total utility score achievable within the 50 MBps capacity?"
        ),
        ground_truth="106",
        metadata={"domain": "knapsack_optimization", "difficulty": "medium"},
    ),
    BenchmarkTask(
        task_id="math_08",
        category="math_planning",
        title="Bipartite Microservice-to-Node Assignment Cost",
        prompt=(
            "Three microservices S1, S2, S3 must be assigned to three cloud virtual machines N1, N2, N3 in a one-to-one matching.\n"
            "The latency/cost matrix C[service, node] is:\n"
            "- S1: N1 = 14, N2 = 5,  N3 = 8\n"
            "- S2: N1 = 9,  N2 = 7,  N3 = 12\n"
            "- S3: N1 = 10, N2 = 6,  N3 = 4\n\n"
            "What is the minimum possible total assignment cost across all 3 services?"
        ),
        ground_truth="18",
        metadata={"domain": "bipartite_matching", "difficulty": "medium"},
    ),
    BenchmarkTask(
        task_id="math_09",
        category="math_planning",
        title="Byzantine Fault Tolerant Quorum Probability",
        prompt=(
            "A distributed consensus cluster consists of N = 7 validator nodes. For a transaction block to commit, "
            "it requires a supermajority of at least 5 honest validator votes in the voting round.\n"
            "Each validator node independently remains honest and online with probability p = 0.8 (and fails or acts maliciously with probability 0.2).\n\n"
            "What is the probability (expressed as a percentage rounded to two decimal places, e.g., 85.20) that the consensus round successfully commits?"
        ),
        ground_truth="85.20",
        metadata={"domain": "distributed_consensus", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="math_10",
        category="math_planning",
        title="Token Bucket Rate Limiter Burst Drain Time",
        prompt=(
            "A cloud API gateway uses a token bucket rate limiter with a maximum bucket capacity of B = 50 tokens "
            "and a constant refill rate of r = 10 tokens per second.\n"
            "At time t = 0 seconds, the bucket is completely full (50 tokens).\n"
            "A sudden spike of 110 requests arrives simultaneously at t = 0. Each request strictly consumes 1 token to be processed. "
            "Requests that do not immediately obtain a token wait in an infinite FIFO queue and are processed instantaneously as soon as new tokens are refilled into the bucket.\n\n"
            "How many seconds from t = 0 does it take for the 110th (final) request to receive its token and finish processing?"
        ),
        ground_truth="6",
        metadata={"domain": "queueing_theory", "difficulty": "medium"},
    ),
]


CODE_SYNTHESIS_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        task_id="code_01",
        category="code_synthesis",
        title="DAG Longest Weighted Path",
        prompt=(
            "Implement a function `longest_dag_path(n: int, edges: list[tuple[int, int, int]]) -> int`:\n"
            "- `n`: number of nodes labeled 0 to n - 1.\n"
            "- `edges`: list of directed weighted edges `(u, v, weight)` representing a Directed Acyclic Graph (DAG).\n"
            "- Return the maximum total path weight between any pair of nodes in the graph.\n"
            "- If there are no edges, return 0. If multiple paths exist, return the weight of the longest path."
        ),
        ground_truth="longest_dag_path",
        test_code=(
            "assert longest_dag_path(1, []) == 0\n"
            "assert longest_dag_path(3, [(0, 1, 5), (1, 2, 7)]) == 12\n"
            "assert longest_dag_path(4, [(0, 1, 3), (0, 2, 2), (1, 3, 4), (2, 3, 8)]) == 10\n"
            "assert longest_dag_path(4, [(0, 1, 10), (1, 2, -2), (2, 3, 5), (0, 3, 1)]) == 13\n"
            "assert longest_dag_path(5, [(0, 1, 1), (1, 2, 1), (3, 4, 10)]) == 10\n"
        ),
        metadata={"algorithm": "topological_sort_dp", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="code_02",
        category="code_synthesis",
        title="Course Schedule Cycle Detection and Lexicographical Order",
        prompt=(
            "Implement `find_safe_schedule(num_courses: int, prerequisites: list[tuple[int, int]]) -> list[int]`:\n"
            "- `num_courses`: total courses labeled 0 to num_courses - 1.\n"
            "- `prerequisites`: list of pairs `(u, v)` meaning course `u` must be taken before course `v` (`u -> v`).\n"
            "- Return a valid sequence to take all courses.\n"
            "- If there are cycles (deadlock), return an empty list `[]`.\n"
            "- If multiple valid orderings exist, return the lexicographically smallest ordering (always take the available course with smallest index first using a min-heap)."
        ),
        ground_truth="find_safe_schedule",
        test_code=(
            "assert find_safe_schedule(2, [(1, 0)]) == [1, 0]\n"
            "assert find_safe_schedule(4, [(1, 0), (2, 0), (3, 1), (3, 2)]) == [3, 1, 2, 0]\n"
            "assert find_safe_schedule(2, [(0, 1), (1, 0)]) == []\n"
            "assert find_safe_schedule(3, []) == [0, 1, 2]\n"
            "assert find_safe_schedule(5, [(0, 2), (1, 2), (3, 4)]) == [0, 1, 2, 3, 4]\n"
        ),
        metadata={"algorithm": "kahns_priority_queue", "difficulty": "medium"},
    ),
    BenchmarkTask(
        task_id="code_03",
        category="code_synthesis",
        title="Constrained Shortest Path with Time Budget",
        prompt=(
            "Implement `min_cost_constrained_path(n: int, roads: list[tuple[int, int, int, int]], src: int, dst: int, max_time: int) -> int`:\n"
            "- A network has `n` nodes labeled 0 to n - 1.\n"
            "- `roads`: list of bidirectional edges `(u, v, cost, time)`.\n"
            "- Find the minimum monetary cost to travel from `src` to `dst` such that the total elapsed time <= `max_time`.\n"
            "- If `dst` cannot be reached within `max_time`, return -1."
        ),
        ground_truth="min_cost_constrained_path",
        test_code=(
            "roads1 = [(0, 1, 10, 5), (1, 2, 10, 5), (0, 2, 100, 2)]\n"
            "assert min_cost_constrained_path(3, roads1, 0, 2, 10) == 20\n"
            "assert min_cost_constrained_path(3, roads1, 0, 2, 4) == 100\n"
            "assert min_cost_constrained_path(3, roads1, 0, 2, 1) == -1\n"
            "assert min_cost_constrained_path(2, [(0, 1, 5, 10)], 0, 1, 9) == -1\n"
            "assert min_cost_constrained_path(3, [(0, 1, 4, 2), (1, 2, 3, 3)], 0, 2, 5) == 7\n"
        ),
        metadata={"algorithm": "dijkstra_2d_state", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="code_04",
        category="code_synthesis",
        title="Interval Partitioning Minimum Server Allocation",
        prompt=(
            "Implement `min_servers_for_tasks(tasks: list[tuple[int, int]]) -> int`:\n"
            "- `tasks`: list of half-open intervals `(start, end)` representing task runtime `[start, end)`.\n"
            "- If a server finishes a task at time `t`, it can immediately start another task that begins at time >= `t`.\n"
            "- Return the minimum number of concurrent servers required to schedule all tasks without any overlap.\n"
            "- If `tasks` is empty, return 0."
        ),
        ground_truth="min_servers_for_tasks",
        test_code=(
            "assert min_servers_for_tasks([]) == 0\n"
            "assert min_servers_for_tasks([(1, 4), (5, 8)]) == 1\n"
            "assert min_servers_for_tasks([(1, 5), (5, 10)]) == 1\n"
            "assert min_servers_for_tasks([(1, 4), (2, 5), (3, 6)]) == 3\n"
            "assert min_servers_for_tasks([(0, 30), (5, 10), (15, 20)]) == 2\n"
            "assert min_servers_for_tasks([(1, 10), (2, 7), (3, 19), (8, 12), (10, 20), (11, 30)]) == 4\n"
        ),
        metadata={"algorithm": "sweep_line_heap", "difficulty": "medium"},
    ),
    BenchmarkTask(
        task_id="code_05",
        category="code_synthesis",
        title="Tree Maximum Weight Independent Set",
        prompt=(
            "Implement `max_weight_independent_set(n: int, edges: list[tuple[int, int]], weights: list[int]) -> int`:\n"
            "- A tree has `n` vertices labeled 0 to n - 1 with non-negative vertex weights.\n"
            "- `edges`: list of `n - 1` undirected edges `(u, v)` forming a connected tree.\n"
            "- Find the maximum total weight of an independent set of vertices (no two chosen vertices are connected by an edge).\n"
            "- If `n == 0`, return 0."
        ),
        ground_truth="max_weight_independent_set",
        test_code=(
            "assert max_weight_independent_set(1, [], [10]) == 10\n"
            "assert max_weight_independent_set(3, [(0, 1), (1, 2)], [5, 12, 6]) == 12\n"
            "assert max_weight_independent_set(4, [(0, 1), (0, 2), (0, 3)], [10, 4, 4, 4]) == 12\n"
            "assert max_weight_independent_set(5, [(0, 1), (1, 2), (2, 3), (3, 4)], [1, 10, 1, 10, 1]) == 20\n"
        ),
        metadata={"algorithm": "tree_dynamic_programming", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="code_06",
        category="code_synthesis",
        title="Parallel Workflow Makespan with Transfer Latencies",
        prompt=(
            "Implement `min_makespan_dag(durations: list[int], dependencies: list[tuple[int, int]], transfer_times: list[tuple[int, int, int]]) -> int`:\n"
            "- `durations`: execution time of each task `i` (0 to n - 1).\n"
            "- `dependencies`: list of pairs `(u, v)` meaning task `u` must complete before task `v` can start.\n"
            "- `transfer_times`: list of tuples `(u, v, t)` meaning transferring data from completed task `u` to task `v` takes `t` units of time (so `start(v) >= finish(u) + t`). If an edge `(u, v)` has no specified transfer time, transfer time defaults to 0.\n"
            "- Assuming unlimited parallel compute resources, return the minimum total makespan (earliest time all tasks are completed)."
        ),
        ground_truth="min_makespan_dag",
        test_code=(
            "assert min_makespan_dag([5, 3], [], []) == 5\n"
            "assert min_makespan_dag([4, 6], [(0, 1)], [(0, 1, 2)]) == 12\n"
            "assert min_makespan_dag([3, 5, 2, 4], [(0, 1), (0, 2), (1, 3), (2, 3)], [(0, 1, 1), (0, 2, 0), (1, 3, 2)]) == 15\n"
        ),
        metadata={"algorithm": "dag_longest_path_makespan", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="code_07",
        category="code_synthesis",
        title="Tree Path Maximum Edge Query (LCA / Tree Traversal)",
        prompt=(
            "Implement `max_edge_on_path(n: int, edges: list[tuple[int, int, int]], queries: list[tuple[int, int]]) -> list[int]`:\n"
            "- `n`: number of vertices in a tree (0 to n - 1).\n"
            "- `edges`: list of `n - 1` undirected edges `(u, v, weight)`.\n"
            "- `queries`: list of pairs `(u, v)`.\n"
            "- For each query `(u, v)`, return the maximum edge weight along the unique simple path between node `u` and node `v`.\n"
            "- If `u == v`, the maximum edge weight on the path is 0."
        ),
        ground_truth="max_edge_on_path",
        test_code=(
            "edges1 = [(0, 1, 4), (1, 2, 7), (1, 3, 2), (3, 4, 9)]\n"
            "queries1 = [(0, 2), (2, 4), (0, 0), (3, 4)]\n"
            "assert max_edge_on_path(5, edges1, queries1) == [7, 9, 0, 9]\n"
            "edges2 = [(0, 1, 10)]\n"
            "assert max_edge_on_path(2, edges2, [(0, 1), (1, 0), (0, 0)]) == [10, 10, 0]\n"
        ),
        metadata={"algorithm": "tree_traversal_lca", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="code_08",
        category="code_synthesis",
        title="Bipartite Graph Maximum Cardinality Matching",
        prompt=(
            "Implement `max_bipartite_matching(n_left: int, n_right: int, edges: list[tuple[int, int]]) -> int`:\n"
            "- A bipartite graph has left partition vertices {0 .. n_left - 1} and right partition vertices {0 .. n_right - 1}.\n"
            "- `edges`: list of directed pairs `(u, v)` where `u` is in left partition and `v` is in right partition.\n"
            "- Return the maximum number of disjoint edges that can be selected (maximum cardinality matching using augmenting paths / Kuhn's algorithm)."
        ),
        ground_truth="max_bipartite_matching",
        test_code=(
            "assert max_bipartite_matching(2, 2, [(0, 0), (1, 1)]) == 2\n"
            "assert max_bipartite_matching(2, 2, [(0, 0), (0, 1), (1, 0), (1, 1)]) == 2\n"
            "assert max_bipartite_matching(3, 2, [(0, 0), (1, 0), (2, 0)]) == 1\n"
            "assert max_bipartite_matching(3, 3, [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]) == 3\n"
            "assert max_bipartite_matching(3, 3, []) == 0\n"
        ),
        metadata={"algorithm": "hopcroft_karp_or_dfs_matching", "difficulty": "hard"},
    ),
]


DUOHIRE_A2A_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        task_id="duohire_01",
        category="duohire_a2a",
        title="DuoHire: Senior Cloud Infrastructure Architect",
        prompt=(
            "Perform multi-constraint recruitment vetting for the following role and candidate:\n\n"
            "[JOB SPECIFICATION: Senior Cloud Infrastructure Architect]\n"
            "- Hard Budget Ceiling: $180,000 / year (strict upper bound).\n"
            "- Location Requirement: Hybrid (minimum 2 days/week on-site at Beijing R&D HQ).\n"
            "- Latest Start Date: Must be able to onboard on or before November 15, 2026.\n"
            "- Mandatory Skill DAG: Go, Kubernetes, and Distributed Systems.\n"
            "  * Prerequisite Rule: Kubernetes mastery requires documented Docker fundamentals.\n\n"
            "[CANDIDATE DOSSIER: Candidate A]\n"
            "- Expected Compensation: $175,000 / year.\n"
            "- Current Location & Preferences: Beijing resident, happy to work 3 days/week on-site.\n"
            "- Availability: Ready to onboard November 1, 2026.\n"
            "- Skill Inventory: Go (5 yrs), Docker (4 yrs), Kubernetes (3 yrs), Distributed Systems (4 yrs).\n\n"
            "Evaluate every condition. Determine whether the candidate passes, is rejected, or requires negotiation."
        ),
        ground_truth="PASS, 0",
        metadata={"domain": "bilateral_recruitment", "difficulty": "medium"},
    ),
    BenchmarkTask(
        task_id="duohire_02",
        category="duohire_a2a",
        title="DuoHire: Distributed Database Storage Kernel Engineer",
        prompt=(
            "Perform multi-constraint recruitment vetting for the following role and candidate:\n\n"
            "[JOB SPECIFICATION: Distributed Database Storage Kernel Engineer]\n"
            "- Hard Budget Ceiling: $150,000 / year (non-negotiable firm cap).\n"
            "- Location Requirement: Strictly 100% On-site in Shenzhen High-Tech Park.\n"
            "- Latest Start Date: Must onboard on or before December 1, 2026.\n"
            "- Mandatory Skills: C++ (>= 3 years), Storage Engine LSM/B-Tree internals, Linux Kernel.\n\n"
            "[CANDIDATE DOSSIER: Candidate B]\n"
            "- Expected Compensation: $190,000 / year.\n"
            "- Current Location & Preferences: Chengdu resident, strictly remote-only, firmly unwilling to relocate.\n"
            "- Availability: January 15, 2027.\n"
            "- Skill Inventory: C++ (4 yrs), Storage Engine (3 yrs), Linux Kernel (2 yrs).\n\n"
            "Evaluate every condition. Count the number of fatal hard conflicts and output the decision."
        ),
        ground_truth="REJECT, 3",
        metadata={"domain": "bilateral_recruitment", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="duohire_03",
        category="duohire_a2a",
        title="DuoHire: LLM Serving & Ascend Compiler Engineer",
        prompt=(
            "Perform multi-constraint recruitment vetting for the following role and candidate:\n\n"
            "[JOB SPECIFICATION: LLM Serving & Ascend Compiler Engineer]\n"
            "- Base Target Budget: $160,000 / year. Negotiable Flex Buffer up to $175,000 / year for candidates with deep Triton/Kernel skills.\n"
            "- Location Requirement: Hybrid in Beijing or Shanghai.\n"
            "- Target Onboarding Date: November 20, 2026 (flexible within 3 weeks if candidate accepts compensation terms).\n"
            "- Mandatory Skill DAG: Python, PyTorch, vLLM / TensorRT-LLM, Triton/CUDA kernels.\n"
            "  * Prerequisite Rule: vLLM serving requires solid PyTorch foundation.\n\n"
            "[CANDIDATE DOSSIER: Candidate C]\n"
            "- Expected Compensation: $170,000 / year.\n"
            "- Current Location: Beijing resident, accepts hybrid policy.\n"
            "- Availability: December 5, 2026 (approximately 2 weeks after target date).\n"
            "- Skill Inventory: Python (6 yrs), PyTorch (4 yrs), vLLM (2 yrs), Triton (1 yr).\n\n"
            "Determine if this profile is an immediate PASS, hard REJECT, or bilateral NEGOTIATE with count of negotiable items."
        ),
        ground_truth="NEGOTIATE, 2",
        metadata={"domain": "bilateral_recruitment", "difficulty": "hard"},
    ),
    BenchmarkTask(
        task_id="duohire_04",
        category="duohire_a2a",
        title="DuoHire: Frontend UI/UX Performance Architect",
        prompt=(
            "Perform multi-constraint recruitment vetting for the following role and candidate:\n\n"
            "[JOB SPECIFICATION: Frontend UI/UX Performance Architect]\n"
            "- Hard Budget Ceiling: $140,000 / year.\n"
            "- Location Requirement: Any location (100% Remote approved).\n"
            "- Latest Start Date: December 1, 2026.\n"
            "- Mandatory Skill DAG: TypeScript, React, TailwindCSS, State Management.\n"
            "  * Prerequisite Rule: TypeScript is an absolute non-negotiable core requirement.\n\n"
            "[CANDIDATE DOSSIER: Candidate D]\n"
            "- Expected Compensation: $135,000 / year (within budget).\n"
            "- Current Location: Hangzhou (remote is fine).\n"
            "- Availability: November 25, 2026.\n"
            "- Skill Inventory: React (4 yrs), TailwindCSS (2 yrs), Redux (3 yrs). Has extensive pure JavaScript background, but 0 months of TypeScript experience.\n\n"
            "Evaluate every condition and determine the final outcome."
        ),
        ground_truth="REJECT, 1",
        metadata={"domain": "bilateral_recruitment", "difficulty": "medium"},
    ),
    BenchmarkTask(
        task_id="duohire_05",
        category="duohire_a2a",
        title="DuoHire: Ascend C Operator Tiling Optimization Engineer",
        prompt=(
            "Perform multi-constraint recruitment vetting for the following role and candidate:\n\n"
            "[JOB SPECIFICATION: Ascend C Operator Tiling Optimization Engineer]\n"
            "- Base Target Budget: $165,000 / year. Negotiable upper bound up to $180,000 / year for candidates with demonstrated Huawei Ascend C / CANN experience.\n"
            "- Location Requirement: On-site Beijing.\n"
            "- Latest Start Date: November 10, 2026.\n"
            "- Mandatory Skill DAG: C++, Ascend C / CANN, Operator Tiling, TVM / Compiler Optimization.\n\n"
            "[CANDIDATE DOSSIER: Candidate E]\n"
            "- Expected Compensation: $178,000 / year (exceeds base target, but within negotiable upper bound due to verified 3 years Ascend C experience).\n"
            "- Location & Work Style: Beijing, fully committed to on-site work.\n"
            "- Availability: November 5, 2026.\n"
            "- Skill Inventory: C++ (5 yrs), Ascend C (3 yrs), Operator Tiling (2 yrs), TVM (1 yr).\n\n"
            "Determine the recruitment status and conflict/negotiation count."
        ),
        ground_truth="NEGOTIATE, 1",
        metadata={"domain": "bilateral_recruitment", "difficulty": "medium"},
    ),
    BenchmarkTask(
        task_id="duohire_06",
        category="duohire_a2a",
        title="DuoHire: DevSecOps & Cloud Security Engineer",
        prompt=(
            "Perform multi-constraint recruitment vetting for the following role and candidate:\n\n"
            "[JOB SPECIFICATION: DevSecOps & Cloud Security Engineer]\n"
            "- Hard Budget Ceiling: $155,000 / year.\n"
            "- Location Requirement: Hybrid in Shenzhen.\n"
            "- Latest Start Date: December 1, 2026.\n"
            "- Mandatory Skill DAG: Kubernetes Security, IAM / OAuth2, CI/CD Hardening, Terraform.\n"
            "  * Prerequisite Rule: Kubernetes Security requires verified Kubernetes core foundation.\n\n"
            "[CANDIDATE DOSSIER: Candidate F]\n"
            "- Expected Compensation: $150,000 / year.\n"
            "- Location & Preferences: Shenzhen resident, accepts hybrid.\n"
            "- Availability: November 20, 2026.\n"
            "- Skill Inventory: Kubernetes (3 yrs), Kubernetes Security (2 yrs), OAuth2 (4 yrs), CI/CD (3 yrs), Terraform (2 yrs).\n\n"
            "Evaluate every condition and determine the final outcome."
        ),
        ground_truth="PASS, 0",
        metadata={"domain": "bilateral_recruitment", "difficulty": "medium"},
    ),
]

ALL_TASKS: List[BenchmarkTask] = (
    MATH_PLANNING_TASKS + CODE_SYNTHESIS_TASKS + DUOHIRE_A2A_TASKS
)
