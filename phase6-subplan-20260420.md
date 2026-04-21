# Phase 6 子计划 - 2026-04-20

> 目标：关闭发现 3（--full-test 样本数 + workers）和发现 4（回归护栏）

---

## 本轮任务

| ID | 任务 | 负责 | 文件 |
|----|------|------|------|
| P6F-1 | --full-test 切到全集大小 | SubAgent-A | test_closed_loop.py |
| P6F-2 | 移除 workers 空参数 | SubAgent-A | test_closed_loop.py |
| P6F-3 | 补 returncode 断言 + full-test 测试 | SubAgent-B | test_t7_review.py |

---

## P6F-1: --full-test 切到全集大小

**问题**：第481-484行只设 eval_mode="full"，没改样本数。第496行默认 n=5。

**改动**：
```python
elif sys.argv[i] == "--full-test":
    eval_mode = "full"
    # 加载全集：读取数据文件长度
    with open("/data/gt/omg/data/CWQ/search_mid/CWQ_context_test.json") as f:
        n = len(json.load(f))
    i += 1
    continue
```

**验收**：`python3 test_closed_loop.py --full-test 2>&1 | head -5` 显示 "Loading 3322 samples"（或实际全集大小）

---

## P6F-2: 移除 workers 空参数

**问题**：workers 被解析和传递但没有并行逻辑，形成伪能力。

**改动**：
1. 删除 CLI 解析（第489-492行）
2. 删除 run_test() 签名中的 workers 参数
3. 删除 CLI 调用中的 workers=workers

**验收**：`python3 test_closed_loop.py --help` 不再显示 --workers

---

## P6F-3: 补 returncode 断言 + full-test 测试

**问题**：现有 subprocess 测试没断言 returncode==0，缺 --full-test 样本数测试。

**改动**：修改 test_t7_review.py 中 TestT7EvalEngineering 类：

### 3a. 给现有测试加 returncode 断言
```python
result = subprocess.run([...], capture_output=True, text=True, timeout=30)
self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
```

### 3b. 新增 test_full_test_loads_all_samples
```python
def test_full_test_loads_all_samples(self):
    """Test --full-test loads full dataset, not default 5."""
    result = subprocess.run([
        sys.executable, "test_closed_loop.py", "--full-test"
    ], capture_output=True, text=True, timeout=30, cwd="/data/gt/omgv2-o1")
    self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
    # Should NOT print "Loading 5 samples"
    self.assertNotIn("Loading 5 samples", result.stdout)
    # Should print "Loading XXXX samples" where XXXX > 100
    import re
    match = re.search(r"Loading (\d+) samples", result.stdout)
    self.assertIsNotNone(match, f"No 'Loading N samples' found in: {result.stdout[:200]}")
    self.assertGreater(int(match.group(1)), 100, "Full test should load >100 samples")
```

**验收**：
1. `python3 -m unittest test_t7_review.TestT7EvalEngineering -v` 全部通过
2. `python3 test_t7_review.py` 全部通过

---

*创建时间: 2026-04-20 18:10*