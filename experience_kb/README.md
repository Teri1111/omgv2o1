# KBQA Experience Knowledge Base

将 RL 课程中的"经验"转化为结构化的错误纠正知识库，用于在 Logical Form / SPARQL 生成时引导 LLM，无需进一步微调。

## 核心思想

借鉴 KnowCoder-A1 的 RL 课程方法，以及 Reflexion、ExpeL、Voyager、Memento-Skills 等论文的思想：

1. **经验收集**: 从 RL 训练轨迹或教师模型中收集成功/失败的 SPARQL 生成轨迹
2. **规则提取**: 用 LLM 分析每一步的状态→动作对，提取可复用的错误纠正规则
3. **知识库存储**: 向量化规则，存入 FAISS 索引，支持快速检索
4. **运行时引导**: 在每一步 SPARQL 生成前，检索相关规则注入 LLM 提示词

## 架构

```
[RL 轨迹 / 教师示范]
       |
       v
[TrajectoryCollector]  -- 解析为 step-level episodes
       |
       v
[ExperienceExtractor]  -- LLM 提取结构化规则
       |
       v
[ExperienceKB]         -- FAISS 向量索引 + 元数据存储
       |
       v
[RuleRetriever]        -- RAG 检索 + prompt 格式化
       |
       v
[PipelineIntegration]  -- 注入规则到 LLM prompt
```

## 模块说明

| 模块 | 功能 |
|------|------|
| `trajectory_collector.py` | 解析 KBQA/SPARQL 轨迹为 step-level episodes |
| `experience_extractor.py` | LLM 分析 episodes，提取结构化错误纠正规则 |
| `knowledge_base.py` | FAISS 向量索引 + JSON 元数据存储 |
| `rule_retriever.py` | 运行时检索相关规则，格式化为 prompt guidance |
| `pipeline_integration.py` | 集成适配器，将经验 KB 接入现有 KBQA 管线 |
| `config.py` | 集中配置管理 |

## 规则类型

| 类型 | 触发场景 | 示例 |
|------|----------|------|
| ERROR_RECOVERY | SPARQL 执行失败/返回空 | "空结果时检查实体类型，用 SearchTypes 验证" |
| CONSTRAINT_GUIDE | 约束过严/过松 | "how many 问题用 COUNT + GROUP BY" |
| TYPE_MISMATCH | 实体/谓词类型冲突 | "Person 误链为 Organization，重新链接" |
| SUCCESS_SHORTCUT | 高效路径发现 | "birth_date 直接查属性，不需要复杂 join" |
| LOGICAL_STRUCTURE | 复杂查询模式 | "否定问题用 FILTER NOT EXISTS" |
| Semantic_Rule | 从多个 episodes 抽象出的元规则 | 自动合并产生 |

## 快速开始

### 1. 安装依赖

```bash
pip install sentence-transformers faiss-cpu numpy
```

### 2. 构建知识库

```bash
# 从轨迹目录构建（简单模式，不使用 LLM）
python main.py build data/trajectories/

# 使用 LLM 提取更高质量的规则
python main.py build data/trajectories/ --use_llm
```

### 3. 搜索知识库

```bash
python main.py search "SPARQL returns empty result for entity query"
python main.py search "how many questions need COUNT" --format_guidance
```

### 4. 查看统计

```bash
python main.py stats
```

### 5. 合并抽象规则

```bash
python main.py consolidate
```

### 6. 快速演示

```bash
python main.py demo
```

## 与现有 vkbqa 系统的关系

| vkbqa 组件 | experience_kb 适配 |
|------------|-------------------|
| memory_worker.py | experience_extractor.py (相同 LLM 提取模式) |
| memory_manager.py | knowledge_base.py (相同 FAISS + JSON 存储) |
| meta_consolidation.py | knowledge_base.py.consolidate() (相同聚类逻辑) |
| iterative_reasoner.py Step 0 | rule_retriever.py.retrieve_for_state() |
| CLIP embeddings | sentence-transformers (纯文本) |
| Visual QA 状态 | SPARQL 生成状态 |

## 输入轨迹格式

```json
{
  "trajectory_id": "traj_xxx",
  "question": "What college did Obama attend?",
  "gold_sparql": "SELECT ?x WHERE { ... }",
  "steps": [
    {
      "step_id": 0,
      "state": {
        "question": "...",
        "linked_entities": ["Barack_Obama"],
        "current_sparql": "...",
        "sparql_result_count": 0,
        "error_type": "empty_result",
        "error_message": "No results found"
      },
      "action": {
        "type": "revise",
        "reasoning": "Need to add institution join",
        "new_sparql": "..."
      },
      "outcome": "success"
    }
  ]
}
```

## 环境变量配置

```bash
export EXP_KB_DIR="/path/to/kb"                    # KB 存储目录
export EXP_TRAJECTORY_DIR="/path/to/trajectories"  # 轨迹目录
export EXP_EMBEDDING_MODEL="all-MiniLM-L6-v2"     # Embedding 模型
export EXP_LLM_BASE_URL="http://localhost:8000/v1" # LLM API 地址
export EXP_LLM_API_KEY="EMPTY"                     # LLM API Key
export EXP_LLM_MODEL_NAME="qwen3.5-9b"            # LLM 模型名
```

## 代码结构

```
experience_kb/
  main.py               # CLI 入口
  config.py             # 配置管理
  DESIGN.md             # 详细架构设计文档
  README.md             # 本文件
  modules/
    __init__.py
    trajectory_collector.py    # 轨迹收集器
    experience_extractor.py    # 经验提取器 (LLM)
    knowledge_base.py          # FAISS 知识库
    rule_retriever.py          # 规则检索器
    pipeline_integration.py    # 管线集成适配器
  data/
    trajectories/              # 存放轨迹文件
    rules/                     # KB 存储目录
  scripts/                     # 辅助脚本
```
