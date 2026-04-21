# Phase 8 子计划 - 2026-04-20

> 目标：修复测试自包含、tearDown 清理范围、帮助文案

---

## 本轮任务

| ID | 任务 | 负责 | 文件 |
|----|------|------|------|
| P8F-1 | setUpClass 内联小样本数据 | SubAgent-A | test_t7_review.py |
| P8F-2 | tearDownClass 只删自己创建的文件 | SubAgent-A | test_t7_review.py |
| P8F-3 | 修正 --full-test 帮助文案 | SubAgent-A | test_closed_loop.py |

---

## P8F-1: setUpClass 内联小样本数据

**问题**：第569-577行读取 /data/gt/omg 完整数据集，CI 无此目录则失败。

**改动**：直接在 setUpClass 中内联 3 条硬编码样本，不读取外部文件：

```python
@classmethod
def setUpClass(cls):
    cls.fixture_dir = os.path.join(os.path.dirname(__file__), "test_fixtures")
    os.makedirs(cls.fixture_dir, exist_ok=True)
    
    # 内联 3 条样本，不依赖外部数据
    cls.data_path = os.path.join(cls.fixture_dir, "cwq_3samples.json")
    mini_data = [
        {"index": "0", "question": "Lou Seal is the mascot for the team that last won the World Series when?",
         "topic_entities": ["m.03_dwn"], "answers": [{"answer_mid": "m.0117q3yz"}]},
        {"index": "1", "question": "Where did the \"Country Nation World Tour\" concert artist go to college?",
         "topic_entities": ["m.010qhfmm"], "answers": [{"answer_mid": "m.01qdhx"}]},
        {"index": "2", "question": "What is the name of the battle where the founder of the Sikh religion died?",
         "topic_entities": ["m.06w4g"], "answers": [{"answer_mid": "m.013zny"}]},
    ]
    with open(cls.data_path, "w") as f:
        json.dump(mini_data, f)
    
    cls.t5_path = os.path.join(cls.fixture_dir, "t5_3samples.json")
    mini_t5 = [
        {"index": "0", "candidate_paths": [["sports.sports_team.team_mascot", "sports.sports_team.championships"]]},
        {"index": "1", "candidate_paths": [["music.artist.concert_tours", "people.person.education", "education.education.institution"]]},
        {"index": "2", "candidate_paths": [["people.person.places_lived", "location.location.people_born_here"]]},
    ]
    with open(cls.t5_path, "w") as f:
        json.dump(mini_t5, f)
```

---

## P8F-2: tearDownClass 只删自己创建的文件

**问题**：rmtree 整个 test_fixtures 目录，可能误删其他文件。

**改动**：只删除自己创建的两个文件，不删除整个目录：

```python
@classmethod
def tearDownClass(cls):
    for path in [cls.data_path, cls.t5_path]:
        if os.path.exists(path):
            os.remove(path)
    # 只在目录为空时才删除
    if os.path.exists(cls.fixture_dir) and not os.listdir(cls.fixture_dir):
        os.rmdir(cls.fixture_dir)
```

---

## P8F-3: 修正 --full-test 帮助文案

**问题**：第447行写 "--full-test: Shorthand for --eval-mode full"，但现在还有全集语义。

**改动**：
```python
print("  --full-test         Run full dataset in full eval mode")
```

---

## 验收标准

1. `python3 -m unittest test_t7_review.TestT7EvalEngineering -v` 全部通过
2. 测试后 test_fixtures 目录被清理（或仅保留目录）
3. 删除 /data/gt/omg 目录后测试仍通过（验证自包含）
4. `python3 test_closed_loop.py --help` 显示正确文案

---

*创建时间: 2026-04-20 18:45*