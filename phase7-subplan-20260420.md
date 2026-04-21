# Phase 7 子计划 - 2026-04-20

> 目标：修复 full 模式样本数回归 + 测试自包含

---

## 本轮任务

| ID | 任务 | 负责 | 文件 |
|----|------|------|------|
| P7F-1 | --full-test 才覆盖 n，--eval-mode full 不覆盖 | SubAgent-A | test_closed_loop.py |
| P7F-2 | 测试自包含：setUp 生成小样本数据 | SubAgent-B | test_t7_review.py |

---

## P7F-1: 修复 full 模式样本数回归

**问题**：第493-496行 `if eval_mode == "full":` 无条件覆盖 n，导致 `--eval-mode full 3` 也变成 3390。

**根因**：应该只在 `--full-test` 时覆盖 n，而不是所有 full 模式。

**改动**：
1. 新增 `full_test = False` 变量
2. `--full-test` 时设 `full_test = True`
3. 改条件为 `if full_test:` 而不是 `if eval_mode == "full":`

```python
# 第481行附近
elif sys.argv[i] == "--full-test":
    eval_mode = "full"
    full_test = True  # 新增
    i += 1
    continue

# 第492-496行
n = int(remaining_args[0]) if remaining_args else 5
if full_test:  # 改为 full_test
    _data_path = os.environ.get("TEST_DATA_PATH", "/data/gt/omg/data/CWQ/search_mid/CWQ_context_test.json")
    with open(_data_path) as f:
        n = len(json.load(f))
```

**验收**：
- `python3 test_closed_loop.py --eval-mode full 3` 显示 "Loading 3 samples"
- `python3 test_closed_loop.py --full-test` 显示 "Loading 3390 samples"

---

## P7F-2: 测试自包含

**问题**：测试依赖 `/tmp/test_data_small.json` 外部文件，不是仓内自包含。

**改动**：在 test_t7_review.py 的 TestT7EvalEngineering 类中添加 setUp 方法，在测试目录生成小样本数据：

```python
class TestT7EvalEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Generate small test fixtures in the test directory."""
        import shutil
        cls.fixture_dir = "/data/gt/omgv2-o1/test_fixtures"
        os.makedirs(cls.fixture_dir, exist_ok=True)
        
        # 从完整数据取前 3 条
        with open("/data/gt/omg/data/CWQ/search_mid/CWQ_context_test.json") as f:
            full_data = json.load(f)
        cls.data_path = os.path.join(cls.fixture_dir, "cwq_3samples.json")
        with open(cls.data_path, "w") as f:
            json.dump(full_data[:3], f)
        
        with open("/data/gt/omg/data/CWQ/t5_search_output/CWQ_final_test.json") as f:
            full_t5 = json.load(f)
        cls.t5_path = os.path.join(cls.fixture_dir, "t5_3samples.json")
        with open(cls.t5_path, "w") as f:
            json.dump(full_t5[:3], f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up fixtures."""
        if os.path.exists(cls.fixture_dir):
            shutil.rmtree(cls.fixture_dir)
```

然后修改测试中的 env 设置：
```python
env["TEST_DATA_PATH"] = self.data_path
env["TEST_T5_PATH"] = self.t5_path
```

**验收**：
- 删除 `/tmp/test_data_small.json` 和 `/tmp/test_t5_small.json`
- `python3 -m unittest test_t7_review.TestT7EvalEngineering -v` 仍然全部通过

---

*创建时间: 2026-04-20 18:30*