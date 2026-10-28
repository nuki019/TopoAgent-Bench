# TopoAgent-Bench: Multi-Agent DAG Dynamic Orchestration & MCTS Workflow Optimization

[![CI](https://img.shields.io/badge/tests-22%20passed-emerald)](./tests/test_topoagent.py)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](./pyproject.toml)
[![Papers](https://img.shields.io/badge/ICLR%202025-MacNet%20%2B%20AFlow-indigo)](./topoagent/)

**TopoAgent-Bench** 是一个面向复杂任务拆解、有向无环图（DAG）串并行动态编排、异构开源模型路由与工作流帕累托（Pareto）优化的实验研究框架。本项目复现并融合了 **ICLR 2025 *MacNet***（DAG 规模化拓扑协同）、**ICLR 2025 Oral *AFlow***（基于蒙特卡洛树搜索 MCTS 的工作流自动生成）以及 **COLM 2024 *DyLAN***（推理期动态共识早停与节点剪枝）的核心机制。

---

## 1. 核心特性与工程亮点

1. **Kahn 分层拓扑并发调度器 (`topoagent/dag_engine.py`)**
   - 自动执行有向无环图（DAG）环路检测（Cycle Detection）与 Kahn 算法拓扑分层（`topological_layers()`）。
   - 同一拓扑层内无依赖的 Agent 节点通过 `asyncio.gather` **100% 并发执行**，自动完成跨层上下文依赖组装（`deps_summary`）与关键路径延迟（Critical Path Latency）追踪。
2. **DyLAN 动态共识早停与剪枝 (`topoagent/topologies.py`)**
   - 支持同层多 Solver 并行推理后的实时共识检测（`majority_vote_consensus` / `domain_aware_consensus`）。一旦首层异构轻量模型达成一致，立即动态剪枝（Skip）下游高成本 Critic 与 Aggregator 层。
3. **AFlow-Lite MCTS 拓扑搜索器 (`topoagent/mcts_search.py`)**
   - 将多 Agent 组织方式建模为离散配置空间，基于 **UCB1 公式** 与多目标帕累托奖励函数自动搜索最优拓扑与模型层级配比。
4. **轨迹蒸馏与本地 4-bit QLoRA + vLLM 闭环 (`scripts/`)**
   - 自动将评测中得分满分的 Golden DAG 轨迹导出为 ShareGPT 格式（`results/distill_trajectories.jsonl`），配套提供适配单卡 12GB 显存（RTX 5070 Ti Laptop + WSL2）的 `Qwen2.5-3B-Instruct` 4-bit NF4 QLoRA 微调脚本与 `vLLM` LoRA 服务启动脚本。

---

## 2. 五种组织拓扑实测对比实验（Quality vs. Latency vs. Token Cost）

基于异构开源模型池（Fast Worker: `qwen3.8-27b` & `deepseek-v4.1-flash`，Flagship Arbiter: `qwen3.8-max`，全局并发限制 `Semaphore(5)`），在 **数学与图规划 (`math_planning`)**、**算法代码生成 (`code_synthesis`)** 与 **多约束双边匹配决策 (`duohire_a2a`)** 三大场景上的实测数据如下：

| 组织拓扑 (Topology) | 总体准确率 (Acc) | 数学规划 | 代码合成 | 双边决策 (A2A) | 平均端到端耗时 | 层内并行加速比 | 平均 Token 消耗 | 动态早停率 | 综合帕累托分 (Pareto) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Single-Agent (`qwen3.8-27b`)** | 83.33% | 75.0% | **100.0%** | 75.0% | 20.2s | 1.00x | **1,310** | 0.0% | **71.26** |
| **2. Sequential Chain (3-Hop 串行链)** | 75.00% | **100.0%** | 50.0% | 75.0% | 68.5s | 1.00x | 5,830 | 0.0% | 31.99 |
| **3. Parallel Star (3x 异构并行 -> 裁判)** | **91.67%** | 75.0% | **100.0%** | **100.0%** | 47.1s | **1.52x** | 5,035 | 0.0% | 60.58 |
| **4. MacNet Diamond DAG (拆解->并行->综合)** | **91.67%** | 75.0% | **100.0%** | **100.0%** | 65.5s | 1.14x | 6,417 | 0.0% | 49.31 |
| **5. DyLAN Pruned DAG (动态共识早停)** | 83.33% | 75.0% | **100.0%** | 75.0% | **19.1s** | **1.22x** | 2,145 | **75.0%** | **70.56** |

### 关键实验结论（Ablation & Insights）
1. **质量边界（Star & Diamond 夺冠，91.67%）**：通过异构并行采样（`qwen3.8-27b` + `deepseek-v4.1-flash`）结合旗舰裁判（`qwen3.8-max`）交叉验证，在代码生成与多约束招聘决策上均达到 **100% 准确率**，相比单智能体提升 **+8.34%**，相比纯串行链提升 **+16.67%**。
2. **串行链退化现象（Chain Error Cascading）**：3 跳串行链路虽然在多步数学推导上达到 100%，但在代码生成任务中因 Verifier 过度改写导致准确率跌至 50.0%，且耗时高达 68.5s。
3. **DyLAN 动态早停的成本收益**：在 **75.0%** 的测试任务上，首层两个异构轻量 Solver 直接达成一致共识并提前终止，使得平均 Token 成本相比 Diamond DAG **下降 66.6%**（2,145 vs. 6,417），端到端耗时 **缩短 70.8%**（19.1s vs. 65.5s）。
4. **裁判层思维链截断消融（Aggregator Reasoning Truncation Ablation）**：当汇聚节点使用默认强制开启长思维链的模型（`kimi-k3`）处理上游多路长上下文时，内部思维链占用近 2,000 Tokens 导致输出截断（准确率仅 `8.33%`）；切换为非思考直出模式（`enable_thinking=False`）并透传格式契约后，准确率恢复至 **`91.67%`**。

---

## 3. 快速开始 (Quickstart)

```bash
# 1. 运行全部 22 项单元测试（Kahn 拓扑分层、环路检测、并行加速比、DyLAN 早停、MCTS UCB1）
python -m unittest discover -s tests -v

# 2. 运行端到端多拓扑实测与 AFlow-Lite MCTS 搜索（自动导出黄金轨迹）
python run_benchmark.py

# 3. 验证本地 RTX 5070 Ti (12GB) 4-bit QLoRA 轨迹蒸馏参数与显存预算 (~6.0GB Peak VRAM)
python scripts/train_qlora_distill.py --dry-run-verify

# 4. 在 WSL2 (Ubuntu 22.04) 中一键执行 QLoRA 微调并启动本地 vLLM 推理服务
bash scripts/wsl_vllm_qlora_setup.sh
```
