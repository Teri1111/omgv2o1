# A4 (LLM-First + Beam-3) 大规模测试分析报告

> 测试时间: 2026-04-21
> 数据集: CWQ test 全集 (3390 样本)
> LLM: deepseek-r1:14b (node07 ollama, 8002 SSH 隧道)
> 评估: --eval-mode full (HIT + F1/P/R/EM)

---

## 一、总体结果

| 指标 | A3 (Beam-3 纯启发式) | A4 (LLM+Beam-3) | Δ |
|------|---------------------|----------------|---|
| HIT Rate | 69.6% | 58.2% | -11.4pp |
| F1 | 0.3194 | 0.2323 | -0.0871 |
| Precision | 0.2689 | 0.1859 | -0.0829 |
| Recall | 0.6905 | 0.5787 | -0.1118 |
| EM | 16.0% | 9.4% | -6.5pp |
| Avg Answers | 188.7 | 709.1 | +520.4 |
| Median Answers | 6 | 8 | +2 |

样本总数: 3390 (A3: 3390)

## 二、逐样本对比

- A3 胜 (A3 HIT, A4 MISS): **441** (13.0%)
- A4 胜 (A3 MISS, A4 HIT): **54** (1.6%)
- 平局 (同 HIT 或同 MISS): **2895** (85.4%)
- F1 差值均值: -0.0871

**净损失: 387 个样本 (441 回归 - 54 改善)**

## 三、F1 分布

| 区间 | 样本数 | 占比 |
|------|--------|------|
| 0.0 | 1416 | 41.8% |
| (0, 0.25) | 897 | 26.5% |
| [0.25, 0.5) | 332 | 9.8% |
| [0.5, 0.75) | 391 | 11.5% |
| [0.75, 1.0) | 34 | 1.0% |
| 1.0 (精确匹配) | 320 | 9.4% |

## 四、答案数量分析

| 统计 | A3 | A4 |
|------|----|----|
| 平均答案数 | 188.7 | 709.1 |
| 中位数 | 6 | 8 |
| 最大值 | 50,027 | 106,627 |
| 0 答案 (空结果) | 134 | 347 |
| >1000 答案 (超膨胀) | 51 | 150 |

## 五、回归样本分析 (A3 HIT → A4 MISS)

共 **441** 个样本从 A3 的 HIT 退化为 A4 的 MISS

### 前 10 个最大 F1 退化样本

| # | 问题 | A3 F1 | A3 答案数 | A4 F1 | A4 答案数 | 退化模式 |
|---|------|-------|----------|-------|----------|---------|
| 1 | What is the hometown of the person who said "Forgive your enemies... | 1.000 | 1 | 0.000 | 150 | 膨胀 |
| 2 | On which continent is there a position of Governor-General of the Bahamas? | 1.000 | 1 | 0.000 | 0 | 空结果 |
| 3 | Where did the author, who published, Unfinished Tales, attended school? | 1.000 | 4 | 0.000 | 305 | 膨胀 |
| 4 | What genre of music was performed by the writer of "Nuit sans Fin?" | 1.000 | 7 | 0.000 | 508 | 膨胀 |
| 5 | What is the currency in the governmental jurisdiction... | 1.000 | 1 | 0.000 | 0 | 空结果 |
| 6 | What political offices were held by the person who said... | 1.000 | 3 | 0.000 | 234 | 膨胀 |
| 7 | What is the currency of the place whose religious organization... | 1.000 | 1 | 0.000 | 2 | 偏差 |
| 8 | What happed at the end of the war to the person who went to... | 1.000 | 2 | 0.000 | 0 | 空结果 |
| 9 | What is the mascot for the Bernie Brewer team? | 1.000 | 3 | 0.000 | 142 | 膨胀 |
| 10 | The location that appointed Benjamin Netanyahu to governmental position... | 1.000 | 1 | 0.000 | 0 | 空结果 |

### 退化模式分类

从回归样本中可识别两种主要退化模式:

1. **答案膨胀** (主要): LLM 选择了一条过宽的关系路径，导致 SPARQL 返回海量无关答案 (如 150, 305, 508, 234, 142 个答案)
2. **空结果**: LLM 选择了不存在或无效的关系路径，SPARQL 返回 0 结果

## 六、改善样本分析 (A3 MISS → A4 HIT)

共 **54** 个样本从 A3 的 MISS 改善为 A4 的 HIT

改善样本的 F1 提升幅度普遍很小 (0.003 ~ 0.105)，说明即使 LLM 找到了正确路径，也通常只命中少量答案。

## 七、关键发现

### 7.1 LLM 干预效果

**整体结论: A4 (LLM-First + Beam-3) 全面弱于 A3 (纯 Beam-3)**

- HIT 率下降: 69.6% → 58.2% (-11.4pp)
- F1 下降: 0.3194 → 0.2323 (-0.0871)
- Precision 下降: 0.2689 → 0.1859 (-0.0829)
- Recall 下降: 0.6905 → 0.5787 (-0.1118)
- EM 下降: 16.0% → 9.4% (-6.5pp)
- 净损失: 387 个样本 (回归 441 - 改善 54)

### 7.2 根因分析

1. **答案膨胀**: A4 平均答案数 709 vs A3 的 188，LLM 选择了过宽的关系导致答案集膨胀 3.8 倍
2. **Precision 拖累**: 多出的答案绝大多数是误判，Precision 从 0.272 降至 0.188
3. **Recall 同步下降**: 说明 LLM 有时偏离了正确路径，丢失了本该命中的答案
4. **Beam 与 LLM 冲突**: Beam search 的多路径探索 + LLM 的不确定选择 = 更多噪声路径被保留

### 7.3 与审查报告的对照

结果完全符合 `/data/gt/新版审查报告-v1.md` 的诊断:

- 审查结论: "LLM 调用有效率极低 (2.4% function call 成功率)"
- 本次实测: deepseek-r1:14b 虽然能产生输出，但选择的关系质量不如启发式规则
- 审查结论: "当前系统更接近 T5 子图 + 启发式贪心遍历"
- 本次实测: LLM 加入后反而引入噪声，验证了 "LLM 实际贡献 ≈ 0" 的判断

### 7.4 F1 与论文对比

| 方法 | 数据集 | F1 | 模型 |
|------|--------|-----|------|
| KBQA-o1 (MCTS) | CWQ | ~53-58% | 8B SFT |
| KnowCoder-A1 | CWQ | 59.3 | 7B SFT+RL |
| **omgv2-o1 A3 (Beam-3)** | **CWQ** | **0.3194** | **无 LLM** |
| **omgv2-o1 A4 (LLM+Beam-3)** | **CWQ** | **0.2323** | **14B zero-shot** |

当前最佳 (A3) F1 0.3194 距 KnowCoder-A1 的 0.593 仍有 **46% 相对差距**。

### 7.5 下一步建议

1. **立即执行 P1-T1~T4**: 重构 Prompt 为 KBQA-o1 风格，统一 Action 解析器
2. **不建议继续跑 LLM 实验**: 在 P1 重构完成前，LLM 实验只会确认已知问题
3. **优先跑 A0 (PG-only)**: 建立不带 LLM 的全集基线，作为后续改善的参照
4. **F1 0.2323 是当前 LLM 的真实水平**: 可直接与论文结果对比
5. **考虑 SFT 蒸馏**: 如果 P1 重构后 LLM 仍有偏差，需要走 KBQA-o1 的 SFT 路线

---

*报告生成时间: 2026-04-21*
*数据来源: /data/gt/test_results_20260420/A4_llm_beam3.jsonl*
