# omgv2-o1 最终结构简述

> 更新时间：2026-04-18
> 用途：给新对话或新成员快速说明项目最终形态，不展开实现细节，只说明结构与职责。

---

## 1. 一句话

最终项目不是“T5 直接给答案”的管线，而是一个 **LLM-first 的 KBQA Agent 系统**：

- T5 负责缩小搜索空间，提供候选路径和受限子图。
- LLM 负责真正的推理、决策和 tool 调用。
- Tool 是可调用的代码能力。
- Skill 是可读取的领域知识文档。
- Experience KB 是 Skill 的来源和演化基础。
- Execution / Validation 负责每一步的可执行反馈。

---

## 2. 最终数据流

```text
问题 + 起点实体
  -> T5 路径检索
  -> 候选路径 / draft hint / 受限子图
  -> LLM Agent
  -> 调用 Tools（探索、扩展表达式、验证、查经验、查看候选路径）
  -> 读取 Skills（KBQA 规则、错误恢复、方向判断、多跳模式）
  -> 执行反馈（partial / final execution）
  -> 最终 S-expression
  -> SPARQL
  -> 答案 + trace
```

核心原则：

- **T5 只做候选提供者，不做最终决策者。**
- **LLM 在受限子图上做工具化推理。**
- **知识库不直接替代推理，而是给 LLM 提供可复用的策略知识。**

---

## 3. 最终分层

### Layer A: T5 / Subgraph 层

职责：把全图问题压缩成一个小搜索空间，给后续 LLM 推理提供边界。

包含：

- T5 路径检索结果
- top-k candidate paths
- path draft hint
- restricted subgraph

在最终结构里，T5 的作用是：

- 提供候选 relation
- 提供初始 draft expression
- 提供受限子图
- 在 LLM 失败时作为 fallback 参考

不再负责：

- 直接短路出最终答案
- 替代 LLM 做主决策

对应位置：

- `/data/gt/omgv2-o1/reasoning/subgraph.py`
- `/data/gt/omgv2-o1/skills/path_to_lf.py`

### Layer B: LLM Agent 层

职责：作为系统控制器，决定下一步该探索什么、扩展什么表达式、是否需要验证、是否需要查经验知识。

最终会有两类 Agent：

- `Greedy / LLMGuidedAgent`：单路径 step-by-step 推理
- `BeamAgent`：多分支保留与比较

LLM 在这里负责：

- 看当前状态
- 选择下一步 Tool
- 结合 Tool Observation 更新推理状态
- 必要时调用 Skill 或 KB guidance
- 决定何时 Finish

对应位置：

- `/data/gt/omgv2-o1/reasoning/agent.py`
- `/data/gt/omgv2-o1/reasoning/beam_agent.py`
- `/data/gt/omgv2-o1/reasoning/llm_agent.py`

### Layer C: Tools 层

职责：提供 LLM 可以直接调用的、具备 JSON Schema 的原子能力。

最终建议保留 5 个核心 Tool：

1. `explore_neighbors`
2. `extend_expression`
3. `verify_expression`
4. `consult_experience`
5. `inspect_path`

说明：

- `START / JOIN / AND / ARG / CMP / TC / COUNT / STOP` 这些不再作为对外 Skill。
- 它们属于内部 DSL primitive，由 `extend_expression` 统一封装。
- LLM 不需要知道底层拼接函数名，只需要会调用 Tool。

对应位置：

- `/data/gt/omgv2-o1/skills/lf_construction.py`：底层 primitive
- `/data/gt/omgv2-o1/skills/skill_registry.py`：Tool registry / schema registry
- `/data/gt/omgv2-o1/skills/tools/`：最终应新增，放 5 个 Tool 定义

### Layer D: Skill 层

职责：提供 LLM 可读的 KBQA 领域知识，不是函数，而是文档。

Skill 的本质：

- 不是“被调用的 Python 函数”
- 而是“被 LLM 阅读并执行的程序性知识”

典型 Skill 类型：

- relation direction 判断
- multi-hop pattern
- constraint merge
- CVT navigation
- error recovery

最终形式：

- 每个 Skill 是一个 `SKILL.md`
- 包含触发条件、适用场景、步骤、常见错误、避免方式
- 由 LLM 读取后，结合 Tool 去执行

对应位置：

- `/data/gt/omgv2-o1/skills/domain_skills/`：最终应新增，存放 KBQA 领域 Skill 文档

### Layer E: Experience KB 层

职责：把历史规则、错误修复经验、多跳模式沉淀成可检索知识，并向 Skill 层供给内容。

Experience KB 的作用：

- 存规则
- 做检索
- 记录 success / fail count
- 支持后续排序和在线反馈闭环

最终关系：

- Experience KB 是底层知识存储
- Skill 是从 Experience KB 中抽取、整理、包装出来的可读知识接口
- `consult_experience` Tool 负责把这些知识送给 LLM

对应位置：

- `/data/gt/experience_kb/`
- `/data/gt/omgv2-o1/skills/experience_kb_skill.py`

### Layer F: Execution / Validation 层

职责：把“推理”变成“可执行、可验证、可反馈”的闭环。

核心能力：

- partial execution
- final execution
- syntax validation
- intermediate answer checking
- error message feedback

这一层是 Tool Observation 的来源，也是 LLM 自修正的关键。

对应位置：

- `/data/gt/omgv2-o1/skills/execution_feedback.py`
- `/data/gt/omgv2-o1/skills/validate_syntax.py`
- `/data/gt/omgv2-o1/executor/`

### Layer G: Evaluation / Trace 层

职责：验证系统是否真的比 Path-Guided 更强，并保留完整推理轨迹。

包含：

- smoke test
- closed-loop evaluation
- trace export
- case study / ablation 数据

对应位置：

- `/data/gt/omgv2-o1/test_smoke.py`
- `/data/gt/omgv2-o1/test_closed_loop.py`
- `/data/gt/t4_llm_first_gpu_20260418.json`
- `/data/gt/t4_llm_first_gpu_20260418.log`

---

## 4. 最终目录形态（简洁版）

下面是“最终形态”的推荐目录，不要求当前全部已经落地，但这是后续应收敛到的结构：

```text
omgv2-o1/
  reasoning/
    agent.py                # Greedy / LLMGuided agent
    beam_agent.py           # Beam search agent
    llm_agent.py            # LLM 调用与 tool choice
    subgraph.py             # 受限子图对象

  skills/
    lf_construction.py      # Layer 1: 内部 S-expression primitive
    skill_registry.py       # Tool registry + schema registry
    execution_feedback.py   # partial / final execution feedback
    validate_syntax.py      # 语法验证
    path_to_lf.py           # T5 path -> draft hint
    experience_kb_skill.py  # consult_experience 接口

    tools/                  # Layer 2: LLM-callable tools
      explore_neighbors.py
      extend_expression.py
      verify_expression.py
      consult_experience.py
      inspect_path.py

    domain_skills/          # Layer 3: KBQA domain skills (SKILL.md)
      relation_direction/
        SKILL.md
      multi_hop_pattern/
        SKILL.md
      constraint_merge/
        SKILL.md
      cvt_navigation/
        SKILL.md
      error_recovery/
        SKILL.md

  executor/
    ...

  ontology/
    ...

  test_smoke.py
  test_closed_loop.py

experience_kb/
  modules/                  # 知识库核心逻辑
  data/                     # rules / episodes / extracted rules
  main.py
```

---

## 5. 每个模块的最终定位

### T5

- 输入：问题、起点实体
- 输出：candidate paths、draft hint、restricted subgraph
- 定位：搜索空间压缩器

### Subgraph

- 输入：T5 给出的候选 relation / path
- 输出：LLM 可探索的小图
- 定位：LLM tool use 的边界条件

### LLM

- 输入：问题、当前表达式、Tool Observation、Skill guidance、T5 hint
- 输出：下一步 action / tool call / finish
- 定位：真正的推理控制器

### Tools

- 输入：LLM 的结构化调用
- 输出：结构化 observation
- 定位：代码级执行能力

### Skills

- 输入：KBQA 经验知识整理后的文档
- 输出：LLM 可读的程序性知识
- 定位：领域知识接口

### Experience KB

- 输入：规则、案例、错误修复经验、在线反馈
- 输出：可检索的经验知识
- 定位：Skill 的底层知识源

### Execution Feedback

- 输入：中间表达式 / 最终表达式
- 输出：valid / answers / num_answers / error
- 定位：自修正闭环

---

## 6. 最终要表达的论文故事

最终项目要讲清楚的不是“我们把 T5、LLM、知识库都放进来了”，而是：

1. **T5 先把 KBQA 搜索空间压缩成受限子图。**
2. **LLM 在小图上通过 Tool-use 做真实推理。**
3. **Skill 提供领域知识，Experience KB 提供可演化经验。**
4. **Execution feedback 让整个系统形成可验证、自修正闭环。**

一句话概括就是：

> **Subgraph-Constrained Tool-Augmented KBQA Agent with Skill-Grounded Experience Guidance**
