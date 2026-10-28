"""
Workflow Topologies Factory and DyLAN Consensus Extractors.
"""

import re
from typing import List, Optional, Callable, Dict, Any
from collections import Counter

from topoagent.dag_engine import AgentNode, WorkflowDAG


def extract_answer(text: str) -> str:
    """
    Robustly extracts target answer or boxed expression from LLM outputs.
    Handles LaTeX \\boxed{...} with nested braces, 'The answer is ...',
    option letters (A-E), and numbers.
    """
    if not text:
        return ""

    # Check for \\boxed{...} with balanced brace parsing
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        depth = 0
        start = idx + len(r"\boxed{")
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    return text[start:i].strip()
                depth -= 1

    # Check for explicit 'the answer is:? X'
    ans_match = re.search(r"(?:the\s+)?answer\s+is\s*[:\*\s]*([^\n\.\,]+)", text, re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).strip()

    # Check for explicit option choices e.g. 'choice C' or 'Option [B]'
    opt_match = re.search(r"(?:choice|option)\s*[:\*\s]*[\(\[]?([A-E])[\)\]]?", text, re.IGNORECASE)
    if opt_match:
        return opt_match.group(1).upper().strip()

    bracket_opt = re.search(r"(?:^|[^\w])[\(\[]([A-E])[\)\]]", text, re.IGNORECASE)
    if bracket_opt:
        return bracket_opt.group(1).upper().strip()

    # Check for trailing number
    num_matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if num_matches:
        return num_matches[-1].strip()

    # Fallback to trimmed text
    return text.strip()


def majority_vote_consensus(
    outputs: List[str],
    extract_fn: Optional[Callable[[str], str]] = None,
    min_agreement: int = 2,
) -> Optional[str]:
    """
    Returns consensus answer if at least min_agreement parallel solvers agree.
    Otherwise returns None.
    """
    if not outputs:
        return None

    extractor = extract_fn or extract_answer
    extracted_answers = [extractor(out) for out in outputs if out]
    extracted_answers = [a for a in extracted_answers if a]

    if len(extracted_answers) < min_agreement:
        return None

    counter = Counter(extracted_answers)
    most_common, count = counter.most_common(1)[0]
    if count >= min_agreement:
        return most_common
    return None


def exact_match_consensus(
    outputs: List[str],
    extract_fn: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    """
    Returns consensus answer if all outputs match the exact same extracted value.
    """
    if len(outputs) < 2:
        return None

    extractor = extract_fn or extract_answer
    extracted = [extractor(out) for out in outputs if out]
    if len(extracted) >= 2 and len(set(extracted)) == 1:
        return extracted[0]
    return None


# =====================================================================
# Factory Functions for the 5 Core Topologies
# =====================================================================

def create_single_agent_dag(
    tier: str = "fast",
    system_prompt: str = "You are a concise, precise, and analytical expert reasoning agent. Solve the problem directly.",
    task_template: str = "{input}",
) -> WorkflowDAG:
    """
    Topology 1: Single Agent (Baseline)
    Consists of 1 solver node.
    """
    dag = WorkflowDAG(name="single_agent")
    solver = AgentNode(
        id="solver",
        role="solver",
        system_prompt=system_prompt,
        task_template=task_template,
        deps=[],
        model_tier=tier,
        temperature=0.3,
    )
    dag.add_node(solver)
    return dag


def create_chain_dag(
    tier: str = "fast",
) -> WorkflowDAG:
    """
    Topology 2: Sequential Chain
    3 sequential nodes: planner -> solver -> verifier.
    """
    dag = WorkflowDAG(name="chain")

    planner = AgentNode(
        id="planner",
        role="planner",
        system_prompt="You are an expert strategic planner. Analyze the input problem and decompose it into clear, step-by-step instructions.",
        task_template="Analyze this problem and outline a clear step-by-step solution plan:\n{input}",
        deps=[],
        model_tier=tier,
        temperature=0.3,
    )

    solver = AgentNode(
        id="solver",
        role="solver",
        system_prompt="You are an expert problem solver. Follow the provided plan carefully and compute the comprehensive answer.",
        task_template="Problem:\n{input}\n\nExecution Plan:\n{planner}\n\nPlease compute the full detailed solution step by step.",
        deps=["planner"],
        model_tier=tier,
        temperature=0.3,
    )

    verifier = AgentNode(
        id="verifier",
        role="verifier",
        system_prompt="You are a strict verifier and proofchecker. Check the solution steps, verify edge cases, and output the complete final verified solution strictly following the format requested in the problem's [INSTRUCTIONS].",
        task_template="Original Problem:\n{input}\n\nProposed Solution:\n{solver}\n\nVerify every step and edge case. Correct any errors and output the complete final verified response strictly following the [INSTRUCTIONS] format in the Original Problem.",
        deps=["solver"],
        model_tier=tier,
        temperature=0.1,
    )

    dag.add_node(planner).add_node(solver).add_node(verifier)
    return dag


def create_star_dag(
    fast_tier: str = "fast",
    alt_fast_tier: str = "alt_fast",
    flagship_tier: str = "flagship",
) -> WorkflowDAG:
    """
    Topology 3: Star / Diverse Parallel Ensemble
    3 parallel diverse solvers (solver_qwen, solver_deepseek, solver_cot) -> 1 flagship aggregator.
    """
    dag = WorkflowDAG(name="star_ensemble")

    s_qwen = AgentNode(
        id="solver_qwen",
        role="solver_qwen",
        system_prompt="You are a direct, concise mathematical and algorithmic solver.",
        task_template="Solve the following task directly and clearly:\n{input}",
        deps=[],
        model_tier=fast_tier,
        temperature=0.2,
    )

    s_deepseek = AgentNode(
        id="solver_deepseek",
        role="solver_deepseek",
        system_prompt="You are an analytical reasoning specialist. Look for subtle edge cases.",
        task_template="Analyze and solve the task with deep verification of edge cases:\n{input}",
        deps=[],
        model_tier=alt_fast_tier,
        temperature=0.3,
    )

    s_cot = AgentNode(
        id="solver_cot",
        role="solver_cot",
        system_prompt="You are a chain-of-thought reasoner. Show every intermediate algebraic or logical step.",
        task_template="Break down the problem step-by-step and write out your entire chain of reasoning:\n{input}",
        deps=[],
        model_tier=fast_tier,
        temperature=0.5,
    )

    aggregator = AgentNode(
        id="aggregator",
        role="aggregator",
        system_prompt="You are the lead synthesizer. Compare the diverse parallel candidate answers, identify errors or consensus, and produce the authoritative final response strictly following the format requested in the task's [INSTRUCTIONS].",
        task_template=(
            "Task:\n{input}\n\n"
            "Candidate Solution 1 (Qwen):\n{solver_qwen}\n\n"
            "Candidate Solution 2 (DeepSeek):\n{solver_deepseek}\n\n"
            "Candidate Solution 3 (Chain-of-Thought):\n{solver_cot}\n\n"
            "Synthesize these responses, resolve discrepancies, and output the final authoritative solution strictly following the [INSTRUCTIONS] format in the Task (including full ```python ... ``` code if it is a coding task, or \\boxed{}/FINAL_ANSWER if requested)."
        ),
        deps=["solver_qwen", "solver_deepseek", "solver_cot"],
        model_tier=flagship_tier,
        temperature=0.1,
    )

    dag.add_node(s_qwen).add_node(s_deepseek).add_node(s_cot).add_node(aggregator)
    return dag


def create_diamond_dag(
    fast_tier: str = "fast",
    alt_fast_tier: str = "alt_fast",
    flagship_tier: str = "flagship",
) -> WorkflowDAG:
    """
    Topology 4: Diamond Workflow
    planner (fast) -> [solver_a (fast), solver_b (alt_fast)] -> critic_synthesizer (flagship).
    """
    dag = WorkflowDAG(name="diamond")

    planner = AgentNode(
        id="planner",
        role="planner",
        system_prompt="You are a senior workflow planner. Formulate a structured solution strategy.",
        task_template="Formulate an optimal solution strategy for:\n{input}",
        deps=[],
        model_tier=fast_tier,
        temperature=0.3,
    )

    solver_a = AgentNode(
        id="solver_a",
        role="solver_a",
        system_prompt="You are Solver A. Implement the planner strategy using direct algebraic derivation.",
        task_template="Task:\n{input}\n\nPlan:\n{planner}\n\nExecute the plan using direct derivation.",
        deps=["planner"],
        model_tier=fast_tier,
        temperature=0.3,
    )

    solver_b = AgentNode(
        id="solver_b",
        role="solver_b",
        system_prompt="You are Solver B. Implement the planner strategy using alternative invariant checks.",
        task_template="Task:\n{input}\n\nPlan:\n{planner}\n\nExecute the plan using alternative verification.",
        deps=["planner"],
        model_tier=alt_fast_tier,
        temperature=0.3,
    )

    critic = AgentNode(
        id="critic_synthesizer",
        role="critic_synthesizer",
        system_prompt="You are the lead critic and synthesizer. Review both solutions against the original plan, correct flaws, and produce the final verified response strictly following the format requested in the task's [INSTRUCTIONS].",
        task_template=(
            "Task:\n{input}\n\n"
            "Planner Strategy:\n{planner}\n\n"
            "Solver A Output:\n{solver_a}\n\n"
            "Solver B Output:\n{solver_b}\n\n"
            "Evaluate both solutions, reconcile discrepancies, and output the definitive final solution strictly following the [INSTRUCTIONS] format in the Task (including full ```python ... ``` code if it is a coding task, or \\boxed{}/FINAL_ANSWER if requested)."
        ),
        deps=["solver_a", "solver_b"],
        model_tier=flagship_tier,
        temperature=0.1,
    )

    dag.add_node(planner).add_node(solver_a).add_node(solver_b).add_node(critic)
    return dag


def create_dylan_pruned_dag(
    fast_tier: str = "fast",
    alt_fast_tier: str = "alt_fast",
    flagship_tier: str = "flagship",
) -> WorkflowDAG:
    """
    Topology 5: DyLAN Dynamic Pruned Workflow
    Layer 1: [fast_solver_1 (fast), fast_solver_2 (alt_fast)]
    If Layer 1 solvers extract the same answer, DyLAN consensus triggers early stop,
    and downstream expensive flagship nodes (deep_critic, final_aggregator) are pruned!
    """
    dag = WorkflowDAG(name="dylan_pruned")
    dag.early_stop_enabled = True

    fast_s1 = AgentNode(
        id="fast_solver_1",
        role="fast_solver_1",
        system_prompt="You are a fast precision solver. Solve the problem directly and strictly follow the output format in [INSTRUCTIONS].",
        task_template="Solve the task directly and follow the [INSTRUCTIONS] output format:\n{input}",
        deps=[],
        model_tier=fast_tier,
        temperature=0.2,
    )

    fast_s2 = AgentNode(
        id="fast_solver_2",
        role="fast_solver_2",
        system_prompt="You are an alternative fast reasoning solver. Solve independently and strictly follow the output format in [INSTRUCTIONS].",
        task_template="Solve independently and follow the [INSTRUCTIONS] output format:\n{input}",
        deps=[],
        model_tier=alt_fast_tier,
        temperature=0.3,
    )

    deep_critic = AgentNode(
        id="deep_critic",
        role="deep_critic",
        system_prompt="You are a deep cognitive critic. Dissect differences between solvers and conduct comprehensive verification.",
        task_template=(
            "Task:\n{input}\n\n"
            "Fast Solver 1:\n{fast_solver_1}\n\n"
            "Fast Solver 2:\n{fast_solver_2}\n\n"
            "Deeply analyze conflicting points and explain which approach is valid."
        ),
        deps=["fast_solver_1", "fast_solver_2"],
        model_tier=flagship_tier,
        temperature=0.2,
    )

    final_agg = AgentNode(
        id="final_aggregator",
        role="final_aggregator",
        system_prompt="You are the final flagship arbiter. Output the final definitive solution strictly following the [INSTRUCTIONS] format.",
        task_template=(
            "Task:\n{input}\n\n"
            "Critic Analysis:\n{deep_critic}\n\n"
            "Output the final resolved solution strictly following the [INSTRUCTIONS] format in the Task."
        ),
        deps=["deep_critic", "fast_solver_1", "fast_solver_2"],
        model_tier=flagship_tier,
        temperature=0.1,
    )

    dag.add_node(fast_s1).add_node(fast_s2).add_node(deep_critic).add_node(final_agg)
    return dag
