"""TopoAgent-Bench: Benchmark Suites & Evaluation Harness.

Provides deterministic evaluation benchmarks across:
- MATH_PLANNING_TASKS: Multi-step mathematical reasoning, DAG scheduling, graph theory.
- CODE_SYNTHESIS_TASKS: Algorithmic Python coding with graph algorithms and dynamic programming.
- DUOHIRE_A2A_TASKS: Multi-constraint bilateral recruitment decision making.
"""

from benchmarks.tasks import (
    BenchmarkTask,
    MATH_PLANNING_TASKS,
    CODE_SYNTHESIS_TASKS,
    DUOHIRE_A2A_TASKS,
    ALL_TASKS,
    evaluate_task,
    evaluate_math,
    evaluate_code,
    evaluate_duohire,
)

__all__ = [
    "BenchmarkTask",
    "MATH_PLANNING_TASKS",
    "CODE_SYNTHESIS_TASKS",
    "DUOHIRE_A2A_TASKS",
    "ALL_TASKS",
    "evaluate_task",
    "evaluate_math",
    "evaluate_code",
    "evaluate_duohire",
]
