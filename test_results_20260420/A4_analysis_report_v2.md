# A4 (LLM-First + Beam-3) 大规模测试分析报告 v2

> 测试时间: 2026-04-21
> 数据集: CWQ test 全集 (3390 样本)
> LLM: deepseek-r1:14b (node07 ollama, 8002 SSH 隧道)
> 评估: --eval-mode full (HIT + F1/P/R/EM)
> 代码版本: phase9-p2-eval-setup (HEAD=8d1b004, 包含 P1 重构 + P2 F1 eval)

---

## 〇、重要背景 (v2 新增)

**本次 A4 测试使用的已经是 P1 重构后的代码:**

- Git commit 7d6aa89: "P1-T1~T4: Chat API + KBQA-o1 format + think stripping + scratchpad slim + unified parser"
  - `/v1/completions` → `/v1/chat/completions` ✅
  - `max_tokens` 256→1024 ✅
  - `` 块剥离 ✅
  - KBQA-o1 风格 Thought/Action/Observation prompt ✅
  - 统一 Action 解析器 `parse_action()` ✅
  - scratchpad 精简 ✅

- Git commit 8d1b004: "P2F-1~3: Complex action wiring + regression fix + standard F1 eval"
  - Merge/Compare/Time_constraint 动作接入 ✅
  - 标准 F1/P/R/EM 评估 ✅

**这意味着: A4 的负面结果不是因为没做 P1 重构，而是 P1 重构后仍然不够。**

---

## 一、三组实验总体结果

| 指标 | A0 (PG-only) | A3 (Beam-3) | A4 (LLM+Beam-3) |
|------|-------------|------------|-----------------|
| HIT Rate | 69.5% | 69.6% | **58.2%** |
| F1 | 0.3193 | 0.3194 | **0.2323** |
| Precision | 0.2688 | 0.2689 | **0.1859** |
| Recall | 0.6890 | 0.6905 | **0.5787** |
| EM | 16.0% | 16.0% | **9.4%** |
| Avg Answers | 188.2 | 188.7 | **709.1** |
| Median Answers | 6 | 6 | **8** |

样本总数: 3390

### A0 vs A3 (Beam 增益)

- HIT: 69.5% → 69.6% (+0.1pp)
- F1: 0.3193 → 0.3194 (+0.0001)
- **结论: Beam-3 几乎无增益**。PG-only 贪心已经接近 beam 上界。

### A3 vs A4 (LLM 增益)

- HIT: 69.6% → 58.2% (**-11.4pp**)
- F1: 0.3194 → 0.2323 (**-0.0871**)
- **结论: LLM 干预全面负面**。即使在 P1 重构后。

---

## 二、逐样本对比 (A3 vs A4)

- A3 胜 (A3 HIT, A4 MISS): **441** (13.0%)
- A4 胜 (A3 MISS, A4 HIT): **54** (1.6%)
- 平局 (同 HIT 或同 MISS): **2895** (85.4%)
- F1 差值均值: -0.0871

**净损失: 387 个样本 (441 回归 - 54 改善)**

---

## 三、F1 分布 (A4)

| 区间 | 样本数 | 占比 |
|------|--------|------|
| 0.0 | 1416 | 41.8% |
| (0, 0.25) | 897 | 26.5% |
| [0.25, 0.5) | 332 | 9.8% |
| [0.5, 0.75) | 391 | 11.5% |
| [0.75, 1.0) | 34 | 1.0% |
| 1.0 (精确匹配) | 320 | 9.4% |

---

## 四、答案数量分析

| 统计 | A0 | A3 | A4 |
|------|----|----|----|
| 平均答案数 | 188.2 | 188.7 | 709.1 |
| 中位数 | 6 | 6 | 8 |
| 最大值 | 50,000 | 50,027 | 106,627 |

A4 答案数膨胀 3.8 倍 (188→709) 是核心问题。

---

## 五、回归样本分析 (A3 HIT → A4 MISS)

共 **441** 个样本退化

### 退化模式

1. **答案膨胀** (主要): LLM 选择过宽关系路径，SPARQL 返回海量无关答案
   - 典型案例: A3 返回 1 个正确答案 (F1=1.0) → A4 返回 150+ 答案 (F1≈0)
2. **空结果**: LLM 选择无效关系，SPARQL 返回 0 结果
   - 典型案例: A3 返回 1 个正确答案 (F1=1.0) → A4 返回 0 答案 (F1=0)

### 前 5 个最大 F1 退化样本

1. "What is the hometown of the person who said 'Forgive your enemies...'"
   A3: F1=1.0 (1 answer) → A4: F1=0.0 (150 answers) — 膨胀

2. "On which continent is there a position of Governor-General of the Bahamas?"
   A3: F1=1.0 (1 answer) → A4: F1=0.0 (0 answers) — 空结果

3. "Where did the author of Unfinished Tales attend school?"
   A3: F1=1.0 (4 answers) → A4: F1=0.0 (305 answers) — 膨胀

4. "What genre of music was performed by the writer of 'Nuit sans Fin?'"
   A3: F1=1.0 (7 answers) → A4: F1=0.0 (508 answers) — 膨胀

5. "What is the currency in the governmental jurisdiction..."
   A3: F1=1.0 (1 answer) → A4: F1=0.0 (0 answers) — 空结果

---

## 六、改善样本 (A3 MISS → A4 HIT)

仅 **54** 个样本改善，F1 提升幅度很小 (0.003~0.105)

---

## 七、关键发现 (v2 更新)

### 7.1 核心结论

**即使在 P1 重构 (Chat API + KBQA-o1 format + unified parser) 之后，deepseek-r1:14b 的 LLM 干预仍然全面负面。**

- HIT -11.4pp, F1 -0.087, Precision -0.083, Recall -0.112
- 净损失 387 样本
- 答案膨胀 3.8 倍

### 7.2 根因分析

**P1 重构解决了"LLM 无法输出有效 Action"的问题，但没解决"LLM 选择的关系质量差"的问题。**

具体来说:
1. **P1 修复了格式问题**: Chat API + KBQA-o1 prompt 确保 LLM 能输出解析成功的 Action
2. **但 LLM 的关系选择质量差**: deepseek-r1:14b 是通用推理模型，未经过 KBQA 领域训练
3. **Beam search 放大了 LLM 错误**: Beam-3 保留了 LLM 的错误选择作为候选路径
4. **答案膨胀机制**: LLM 倾向于选择更"泛化"的关系 (如 location.location.contains)，导致 SPARQL 返回大量无关实体

### 7.3 与论文的差距分析

| 维度 | KBQA-o1 | KnowCoder-A1 | omgv2-o1 (当前) |
|------|---------|-------------|----------------|
| 模型 | Llama3.1-8B **SFT** | Qwen2.5-7B **SFT+RL** | deepseek-r1:14b **zero-shot** |
| F1 | ~53-58% | 59.3% | 23.2% (LLM) / 31.9% (heuristic) |
| 训练 | 数万条 KBQA 轨迹 SFT | 6.7k 样本 rejection sampling + GRPO | **无训练** |

**根本差距: 对手用了 SFT/RL 训练，我们是 zero-shot。**

### 7.4 Beam-3 无增益的发现

A0 (PG-only) 和 A3 (Beam-3) 结果几乎相同:
- HIT: 69.5% vs 69.6% (+0.1pp)
- F1: 0.3193 vs 0.3194 (+0.0001)

说明: 当前 PG 贪心策略已经足够好，beam search 无法在子图空间内找到更好的路径。**改善空间不在搜索宽度，而在搜索方向 (即 LLM 或训练)。**

### 7.5 下一步建议 (v2 更新)

1. **P1 重构已被验证不够**: 不应继续投入 P1 相关的 prompt 工程
2. **直接进入 P4 训练流程**:
   - 用强模型 (GPT-4 / R1-70B) 在 CWQ train 上收集成功轨迹
   - 对 deepseek-r1:14b 做 rejection sampling SFT
   - 这是 KBQA-o1 和 KnowCoder-A1 成功的核心原因
3. **保留 A0/A3 作为基线**: F1 0.319 是当前无训练系统的上界
4. **考虑换用更大的通用模型**: 如果暂时不做 SFT，可试 qwen2.5-coder:14b (审查报告提到历史测试 68% HIT)

---

## 附录: 与 v1 报告的差异

v1 报告未明确说明 A4 使用的是 P1 重构后的代码，可能给人"做了 P1 就能改善"的误解。
v2 更正: P1 已落地但仍然不够，问题出在模型能力而非格式。

---

*报告生成时间: 2026-04-21 v2*
*数据来源: /data/gt/test_results_20260420/A0_pg_only.jsonl, A3_beam3.jsonl, A4_llm_beam3.jsonl*
*代码版本: phase9-p2-eval-setup@8d1b004 (P1+P2 committed)*
