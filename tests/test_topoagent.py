"""
Comprehensive Unit Tests for TopoAgent-Bench.
Tests:
  1. Kahn's topological sort and layering
  2. Cycle detection & validation
  3. Parallel execution speedup
  4. DyLAN consensus early-stopping & dynamic pruning
  5. AFlow-Lite MCTS UCB1 calculation, search, & distillation export
  6. Live smoke test with Bailian API
"""

import os
import math
import json
import asyncio
import tempfile
import unittest
from pathlib import Path

from topoagent.llm_router import HybridLLMRouter, LLMResponse, load_dotenv
from topoagent.dag_engine import AgentNode, WorkflowDAG, ExecutionTrace
from topoagent.topologies import (
    create_single_agent_dag,
    create_chain_dag,
    create_star_dag,
    create_diamond_dag,
    create_dylan_pruned_dag,
    extract_answer,
    majority_vote_consensus,
    exact_match_consensus,
)
from topoagent.mcts_search import (
    WorkflowConfig,
    MCTSNode,
    MCTSWorkflowSearcher,
    calculate_pareto_reward,
    export_distillation_dataset,
)


class TestTopologicalSortAndCycle(unittest.TestCase):
    """Verifies Kahn's algorithm layering and cycle detection."""

    def test_single_agent_dag_layers(self):
        dag = create_single_agent_dag()
        layers = dag.topological_layers()
        self.assertEqual(len(layers), 1)
        self.assertEqual(len(layers[0]), 1)
        self.assertEqual(layers[0][0].id, "solver")

    def test_chain_dag_layers(self):
        dag = create_chain_dag()
        layers = dag.topological_layers()
        self.assertEqual(len(layers), 3)
        self.assertEqual([n.id for n in layers[0]], ["planner"])
        self.assertEqual([n.id for n in layers[1]], ["solver"])
        self.assertEqual([n.id for n in layers[2]], ["verifier"])

    def test_star_dag_layers(self):
        dag = create_star_dag()
        layers = dag.topological_layers()
        self.assertEqual(len(layers), 2)
        # Layer 0 has 3 parallel solvers
        layer0_ids = sorted([n.id for n in layers[0]])
        self.assertEqual(layer0_ids, ["solver_cot", "solver_deepseek", "solver_qwen"])
        # Layer 1 has aggregator
        self.assertEqual([n.id for n in layers[1]], ["aggregator"])

    def test_diamond_dag_layers(self):
        dag = create_diamond_dag()
        layers = dag.topological_layers()
        self.assertEqual(len(layers), 3)
        self.assertEqual([n.id for n in layers[0]], ["planner"])
        layer1_ids = sorted([n.id for n in layers[1]])
        self.assertEqual(layer1_ids, ["solver_a", "solver_b"])
        self.assertEqual([n.id for n in layers[2]], ["critic_synthesizer"])

    def test_cycle_detection_simple(self):
        dag = WorkflowDAG(name="cycle_dag")
        n1 = AgentNode(id="A", role="a", deps=["B"])
        n2 = AgentNode(id="B", role="b", deps=["A"])
        dag.add_node(n1).add_node(n2)
        with self.assertRaises(ValueError) as ctx:
            dag.validate_acyclic()
        self.assertIn("Cycle detected", str(ctx.exception))

    def test_cycle_detection_triangular(self):
        dag = WorkflowDAG(name="tri_cycle")
        n1 = AgentNode(id="A", role="a", deps=["C"])
        n2 = AgentNode(id="B", role="b", deps=["A"])
        n3 = AgentNode(id="C", role="c", deps=["B"])
        dag.add_node(n1).add_node(n2).add_node(n3)
        with self.assertRaises(ValueError) as ctx:
            dag.topological_layers()
        self.assertIn("Cycle detected", str(ctx.exception))

    def test_undefined_dependency(self):
        dag = WorkflowDAG(name="missing_dep")
        n1 = AgentNode(id="A", role="a", deps=["non_existent_node"])
        dag.add_node(n1)
        with self.assertRaises(ValueError) as ctx:
            dag.topological_layers()
        self.assertIn("undefined dependency", str(ctx.exception))


class TestParallelSpeedup(unittest.IsolatedAsyncioTestCase):
    """Verifies concurrent layer execution and speedup metrics."""

    async def test_parallel_speedup_in_star_dag(self):
        # Create a mock router that simulates network/inference latency
        router = HybridLLMRouter(mock_mode=True)

        async def simulated_generate(prompt, system_prompt="", model_tier="fast", **kwargs):
            # Simulate 40ms of latency per call
            await asyncio.sleep(0.04)
            return LLMResponse(
                content=f"Result for {prompt[:20]}",
                prompt_tokens=30,
                completion_tokens=20,
                latency_ms=40.0,
                model="mock",
                tier=model_tier,
            )

        router.generate = simulated_generate

        dag = create_star_dag()
        trace = await dag.execute(task_input="Solve 15 * 14", router=router)

        # Star DAG has 3 parallel solvers in layer 0, then 1 aggregator in layer 1 (total 4 nodes)
        self.assertEqual(len(trace.node_results), 4)
        # Sum of node latency is 4 * 40 = 160ms
        self.assertGreaterEqual(trace.sum_node_latency_ms, 150.0)
        self.assertEqual(trace.critical_path_ms, trace.sum_node_latency_ms)

        # Wall clock should be ~ 40ms (layer 0) + 40ms (layer 1) = 80ms, well below 160ms
        self.assertLess(trace.wall_time_ms, trace.sum_node_latency_ms * 0.85)
        # Speedup should be > 1.2
        self.assertGreater(trace.parallel_speedup, 1.2)
        self.assertEqual(trace.total_tokens, 4 * (30 + 20))


class TestDyLANEarlyStopping(unittest.IsolatedAsyncioTestCase):
    """Verifies DyLAN consensus dynamic pruning of downstream expensive nodes."""

    async def test_dylan_early_stopping_prunes_downstream_when_solvers_agree(self):
        router = HybridLLMRouter(mock_mode=True)
        # Both Layer 1 fast solvers agree on \boxed{42}
        router.mock_handler = lambda prompt, sys, tier: "Direct calculation shows \\boxed{42}."

        dag = create_dylan_pruned_dag()
        trace = await dag.execute(
            task_input="What is the answer?",
            router=router,
            early_stop_fn=majority_vote_consensus,
        )

        self.assertTrue(trace.early_stopped)
        self.assertEqual(trace.consensus_answer, "42")
        self.assertEqual(trace.final_output, "42")
        self.assertEqual(trace.layers_executed, 1)

        # Layer 1 solvers executed
        self.assertFalse(trace.node_results["fast_solver_1"].skipped)
        self.assertFalse(trace.node_results["fast_solver_2"].skipped)
        # Downstream flagship nodes were dynamically pruned (skipped)
        self.assertTrue(trace.node_results["deep_critic"].skipped)
        self.assertTrue(trace.node_results["final_aggregator"].skipped)

    async def test_dylan_executes_full_dag_when_solvers_disagree(self):
        router = HybridLLMRouter(mock_mode=True)

        # Return different answers depending on node/prompt
        def divergent_mock(prompt, sys, tier):
            if "solver_1" in sys or "precision" in sys:
                return "The answer is \\boxed{42}."
            elif "solver_2" in sys or "independent" in sys:
                return "The answer is \\boxed{99}."
            else:
                return "The synthesized definitive answer is \\boxed{42}."

        router.mock_handler = divergent_mock

        dag = create_dylan_pruned_dag()
        trace = await dag.execute(
            task_input="What is the answer?",
            router=router,
            early_stop_fn=majority_vote_consensus,
        )

        self.assertFalse(trace.early_stopped)
        self.assertEqual(trace.layers_executed, 3)

        # All nodes executed, none skipped
        self.assertFalse(trace.node_results["fast_solver_1"].skipped)
        self.assertFalse(trace.node_results["fast_solver_2"].skipped)
        self.assertFalse(trace.node_results["deep_critic"].skipped)
        self.assertFalse(trace.node_results["final_aggregator"].skipped)
        self.assertIn("42", trace.final_output)


class TestMCTSSearchAndDistillation(unittest.IsolatedAsyncioTestCase):
    """Verifies AFlow-Lite MCTS UCB1 optimization, Pareto scoring, and dataset export."""

    def test_pareto_reward_formula(self):
        # Accuracy 1.0, 500ms latency, 200 tokens
        # R = 1.0 - 0.05*(500/1000) - 0.05*(200/1000) = 1.0 - 0.025 - 0.010 = 0.965
        r = calculate_pareto_reward(1.0, 500.0, 200.0)
        self.assertAlmostEqual(r, 0.965, places=4)

    def test_ucb1_selection(self):
        root = MCTSNode(config=WorkflowConfig(topology="single"))
        child1 = MCTSNode(config=WorkflowConfig(topology="chain"), parent=root)
        child2 = MCTSNode(config=WorkflowConfig(topology="star"), parent=root)
        root.children = [child1, child2]

        # Unvisited child has infinite UCB
        self.assertEqual(child1.ucb1(), float("inf"))

        # After visiting
        root.visits = 10
        child1.visits = 5
        child1.total_reward = 4.0  # Q/N = 0.8
        expected_ucb = (4.0 / 5) + 1.414 * math.sqrt(math.log(10) / 5)
        self.assertAlmostEqual(child1.ucb1(c=1.414), expected_ucb, places=4)

    async def test_mcts_search_mini_loop(self):
        router = HybridLLMRouter(mock_mode=True)
        searcher = MCTSWorkflowSearcher(router=router, exploration_weight=1.414)

        eval_tasks = [
            {"input": "Calculate 10 + 20", "expected": "30"},
            {"input": "Calculate 5 * 6", "expected": "30"},
        ]

        best_config, best_reward, history = await searcher.search(
            eval_tasks=eval_tasks,
            num_iterations=3,
        )

        self.assertIsInstance(best_config, WorkflowConfig)
        self.assertIsInstance(best_reward, float)
        self.assertEqual(len(history), 4)  # 1 root + 3 iterations

    def test_distillation_export_sharegpt_and_alpaca(self):
        node = AgentNode(
            id="solver",
            role="solver",
            output="Step 1: Compute 2+2=4. Final: \\boxed{4}",
            prompt_tokens=20,
            completion_tokens=15,
            latency_ms=30.0,
        )
        trace = ExecutionTrace(
            dag_name="single_agent",
            task_input="Compute 2+2",
            final_output="4",
            node_results={"solver": node},
            wall_time_ms=35.0,
            critical_path_ms=30.0,
            sum_node_latency_ms=30.0,
            total_tokens=35,
            parallel_speedup=0.86,
            success=True,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            sharegpt_file = os.path.join(tmp_dir, "dataset_sharegpt.jsonl")
            exported_count = export_distillation_dataset([trace], sharegpt_file, format_type="sharegpt")
            self.assertEqual(exported_count, 1)

            with open(sharegpt_file, "r", encoding="utf-8") as f:
                line = json.loads(f.readline())
                self.assertIn("conversations", line)
                self.assertEqual(line["conversations"][0]["value"], "Compute 2+2")
                self.assertEqual(line["metadata"]["dag_name"], "single_agent")

            alpaca_file = os.path.join(tmp_dir, "dataset_alpaca.jsonl")
            exported_count_alp = export_distillation_dataset([trace], alpaca_file, format_type="alpaca")
            self.assertEqual(exported_count_alp, 1)

            with open(alpaca_file, "r", encoding="utf-8") as f:
                line_alp = json.loads(f.readline())
                self.assertIn("instruction", line_alp)
                self.assertEqual(line_alp["input"], "Compute 2+2")


class TestLLMRouterAndTiers(unittest.IsolatedAsyncioTestCase):
    """Verifies LLM router features, fallback mechanisms, and concurrency."""

    def test_dotenv_manual_parser(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env") as f:
            f.write("# Sample comment\n")
            f.write("TEST_ENV_KEY_1=value1\n")
            f.write("TEST_ENV_KEY_2='value_quoted'\n")
            temp_path = f.name

        try:
            loaded = load_dotenv(temp_path)
            self.assertEqual(loaded.get("TEST_ENV_KEY_1"), "value1")
            self.assertEqual(loaded.get("TEST_ENV_KEY_2"), "value_quoted")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def test_mock_router_mode(self):
        router = HybridLLMRouter(mock_mode=True)
        res = await router.generate("Hello world", model_tier="fast")
        self.assertIn("Hello world", res.content)
        self.assertEqual(len(res), 4)
        c, pt, ct, lat = res
        self.assertEqual(c, res.content)
        self.assertGreater(pt, 0)
        self.assertGreater(ct, 0)
        self.assertGreaterEqual(lat, 0.0)

    async def test_local_vllm_fallback_to_fast(self):
        # Point LOCAL_VLLM_URL to an unreachable local port to trigger fallback
        old_url = os.environ.get("LOCAL_VLLM_URL")
        os.environ["LOCAL_VLLM_URL"] = "http://127.0.0.1:59999/v1"
        try:
            # We mock _sync_http_request to verify it tries vLLM then falls back
            router = HybridLLMRouter(mock_mode=False)
            attempts = []

            def mock_http(url, api_key, payload, timeout):
                attempts.append(url)
                if "59999" in url:
                    raise ConnectionRefusedError("Connection refused by mock vLLM")
                return {
                    "choices": [{"message": {"content": "Fallback fast response"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 5},
                }

            router._sync_http_request = mock_http
            res = await router.generate("Test prompt", model_tier="local_vllm")
            self.assertEqual(res.content, "Fallback fast response")
            self.assertEqual(len(attempts), 2)
            self.assertIn("59999", attempts[0])
            self.assertIn("chat/completions", attempts[1])
        finally:
            if old_url:
                os.environ["LOCAL_VLLM_URL"] = old_url
            else:
                os.environ.pop("LOCAL_VLLM_URL", None)


class TestAnswerExtraction(unittest.TestCase):
    """Verifies answer extraction patterns."""

    def test_extract_boxed(self):
        self.assertEqual(extract_answer(r"The result is \boxed{3.14}."), "3.14")
        self.assertEqual(extract_answer(r"\boxed{\frac{1}{2}}"), r"\frac{1}{2}")

    def test_extract_the_answer_is(self):
        self.assertEqual(extract_answer("Therefore, the answer is: 42"), "42")
        self.assertEqual(extract_answer("The answer is Beijing."), "Beijing")

    def test_extract_option_choices(self):
        self.assertEqual(extract_answer("The correct choice is (C)"), "C")
        self.assertEqual(extract_answer("Option [B] is correct"), "B")

    def test_extract_trailing_number(self):
        self.assertEqual(extract_answer("Final value equals 1234"), "1234")


class TestLiveBailianSmoke(unittest.IsolatedAsyncioTestCase):
    """
    Live smoke test invoking 1 actual Bailian API call if key is present,
    or verifying offline mock router if running in an air-gapped CI.
    """

    async def test_bailian_api_live_or_mock_smoke(self):
        load_dotenv()
        api_key = os.getenv("BAILIAN_API_KEY", "")
        # If valid API key is present in environment, test 1 real call
        if api_key and not api_key.startswith("dummy"):
            router = HybridLLMRouter(mock_mode=False, timeout=15.0)
            res = await router.generate(
                prompt="Ping! Respond only with: Pong",
                system_prompt="You are a ping-pong bot.",
                model_tier="fast",
                temperature=0.1,
            )
            self.assertIn("Pong", res.content)
            self.assertGreater(res.latency_ms, 0)
            self.assertGreater(res.prompt_tokens, 0)
            self.assertGreater(res.completion_tokens, 0)
            # Ensure tuple unpack works natively
            c, pt, ct, lat = res
            self.assertEqual(c, res.content)
            self.assertEqual(pt, res.prompt_tokens)
        else:
            # Fallback to mock mode for air-gapped CI
            router = HybridLLMRouter(mock_mode=True)
            res = await router.generate("Ping! Respond only with: Pong", model_tier="fast")
            self.assertTrue(len(res.content) > 0)


if __name__ == "__main__":
    unittest.main()
