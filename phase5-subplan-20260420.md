# Phase 5 子计划 - 2026-04-20 本轮执行

> 基于最新审查报告
> 目标：真正实现长跑评测入口 + 评测工程化护栏
> 审查结论：发现 3（长跑入口）和发现 4（评测护栏）未关闭

---

## 审查发现（2 项中）

| 编号 | 发现 | 严重度 | 当前状态 |
|------|------|--------|---------|
| 发现 3 | 长跑评测入口只"参数已声明"，未真正实现 | **中** | ❌ 未关闭 |
| 发现 4 | 回归护栏缺评测工程化覆盖 | **中** | ❌ 未关闭 |

---

## 本轮任务范围

| ID | 任务 | 优先级 | 负责 | 目标文件 |
|----|------|--------|------|---------|
| P5F-1 | run_test() 实现落盘/续跑/并行 | **P0** | SubAgent-A | `/data/gt/omgv2-o1/test_closed_loop.py` |
| P5F-2 | CLI 传递参数到 run_test() | **P0** | SubAgent-A | `/data/gt/omgv2-o1/test_closed_loop.py` |
| P5F-3 | 评测工程化回归测试 | **P0** | SubAgent-B | `/data/gt/omgv2-o1/test_t7_review.py` |

---

## P5F-1: run_test() 实现落盘/续跑/并行

**问题**：run_test() 函数只接收 num_samples/use_llm/trace/beam/eval_mode，没有 output_path/resume/workers 参数，内部也没有对应实现。

**改动**：

### 1a. 修改 run_test() 签名
```python
def run_test(num_samples=5, use_llm=False, trace=False, trace_export=None, 
             beam_width=None, llm_first=False, show_stats=False, eval_mode="hit",
             output_path=None, resume=False, workers=1):
```

### 1b. 实现断点续跑（resume）
```python
completed_ids = set()
if resume and output_path and os.path.exists(output_path):
    with open(output_path, "r") as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                completed_ids.add(result.get("question_id"))
    print(f"Resuming: {len(completed_ids)} samples already completed")
```

在主循环中跳过已完成样本：
```python
for idx, sample in enumerate(samples):
    question_id = sample.get("index", idx)
    if question_id in completed_ids:
        continue
    # ... 处理样本 ...
```

### 1c. 实现结果落盘（output_path）
每个样本处理完后立即写入 JSONL：
```python
if output_path:
    result_line = {
        "question_id": question_id,
        "question": question,
        "predicted_answers": sorted(list(predicted_answers)),
        "golden_answers": sorted(list(golden_answers)),
        "lf_sexpr": lf_sexpr,
        "merged_sexpr": merged_sexpr,
        "f1": f1_score,
        "precision": precision,
        "recall": recall,
        "em": em,
        "hit": hit,
    }
    with open(output_path, "a") as f:
        f.write(json.dumps(result_line) + "\n")
```

### 1d. 实现并行执行（workers > 1）
```python
if workers > 1:
    from multiprocessing import Pool
    with Pool(workers) as pool:
        results = pool.map(process_sample, sample_list)
else:
    results = [process_sample(s) for s in sample_list]
```

注意：需要将单样本处理逻辑抽取为独立函数 `process_sample(sample)`。

---

## P5F-2: CLI 传递参数到 run_test()

**问题**：第473行调用 run_test() 时没传 output_path, resume, workers。

**改动**：
```python
run_test(n, use_llm=use_llm, trace=trace, trace_export=trace_export, 
         beam_width=beam_width, llm_first=llm_first, show_stats=show_stats, 
         eval_mode=eval_mode, output_path=output_path, resume=resume, workers=workers)
```

---

## P5F-3: 评测工程化回归测试

**问题**：现有测试没有覆盖 compute_f1、--output 落盘、--resume 续跑。

**改动**：在 test_t7_review.py 新增 TestT7EvalEngineering 类：

### 3a. test_compute_f1_local
```python
def test_compute_f1_local(self):
    """Test local compute_f1 without external imports."""
    from evaluate import compute_f1
    # Test edge cases
    self.assertEqual(compute_f1(set(), set()), {"precision": 1.0, "recall": 1.0, "f1": 1.0})
    self.assertEqual(compute_f1({"a"}, set())["f1"], 0.0)
    # Test partial match
    result = compute_f1({"a", "b"}, {"a", "c"})
    self.assertAlmostEqual(result["precision"], 0.5)
    self.assertAlmostEqual(result["recall"], 0.5)
```

### 3b. test_output_jsonl_written
```python
def test_output_jsonl_written(self):
    """Test --output writes JSONL file."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name
    try:
        # Run test with --output
        subprocess.run([
            sys.executable, "test_closed_loop.py", "--output", output_path, "1"
        ], cwd="/data/gt/omgv2-o1", timeout=30)
        # Verify file exists and has content
        self.assertTrue(os.path.exists(output_path))
        with open(output_path) as f:
            lines = [l for l in f if l.strip()]
        self.assertGreater(len(lines), 0)
        # Verify JSON structure
        result = json.loads(lines[0])
        self.assertIn("question_id", result)
        self.assertIn("f1", result)
    finally:
        os.unlink(output_path)
```

### 3c. test_resume_skips_completed
```python
def test_resume_skips_completed(self):
    """Test --resume skips already completed samples."""
    import tempfile
    # Create a fake result file with 1 completed sample
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write(json.dumps({"question_id": "WebQTest-0", "f1": 1.0}) + "\n")
        output_path = f.name
    try:
        # Run with --resume, check output shows "1 samples already completed"
        result = subprocess.run([
            sys.executable, "test_closed_loop.py", "--resume", "--output", output_path, "2"
        ], cwd="/data/gt/omgv2-o1", capture_output=True, text=True, timeout=30)
        self.assertIn("already completed", result.stdout)
    finally:
        os.unlink(output_path)
```

---

## 验收标准

1. `python3 test_closed_loop.py --output /tmp/test.json 3` 生成 JSONL 文件
2. `python3 test_closed_loop.py --resume --output /tmp/test.json 3` 显示 "N samples already completed"
3. `python3 -m unittest test_t7_review.TestT7EvalEngineering -v` 全部通过
4. `python3 test_t7_review.py` 全部通过（包括新增测试）

---

*创建时间: 2026-04-20 17:45*
*基于: 最新审查报告*