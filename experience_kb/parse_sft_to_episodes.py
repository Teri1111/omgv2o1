#!/usr/bin/env python3
"""
parse_sft_to_episodes.py

将 SFT messages 格式的轨迹解析为 step-level episodes。
每个 step = 一次 assistant 的 think+action 配对下一个 user 的 observation。

用法:
  python3 parse_sft_to_episodes.py --input <sft_json> --output <output_jsonl> [--limit N]
"""

import json
import os
import sys
import uuid
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


def extract_question_from_user_msg(user_content: str) -> str:
    """从第一个 user 消息中提取原始问题。"""
    m = re.search(r'Question:\s*(.+?)(?:\n|$)', user_content)
    if m:
        return m.group(1).strip()
    m2 = re.search(r'Topic Entities:\s*(.+?)(?:\n|$)', user_content)
    if m2:
        return m2.group(1).strip()
    lines = user_content.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if "?" in line and len(line) > 10:
            return line
    return user_content.strip()[:200]


def parse_think_and_action(assistant_content: str):
    """解析 assistant 消息，分离 think 和 action。返回 (think, action, action_type)"""
    think_match = re.search(r"<think>(.*?)</think>", assistant_content, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""

    if think_match:
        action = assistant_content[think_match.end():].strip()
    else:
        action = assistant_content.strip()

    action_lower = action.lower()
    if "\\boxed{" in action or "final answer" in action_lower or action.startswith("<answer>"):
        action_type = "answer"
    elif "\\boxed{" in think and not action:
        action_type = "answer"
        action = "\\boxed{...}"
    elif action:
        action_type = "tool_call"
    else:
        action_type = "unknown"

    return think, action, action_type


def parse_observation(user_content: str):
    """解析 observation（user 消息，通常是工具返回结果）。返回 (observation_text, success)"""
    obs = user_content.strip()
    if "error" in obs.lower() and len(obs) < 100:
        success = False
    elif '"results": ""' in obs or '"results": []' in obs:
        success = False
    elif obs:
        success = True
    else:
        success = False
    return obs, success


def summarize_preceding_steps(steps_so_far: list) -> str:
    """生成前几步的简要描述。"""
    if not steps_so_far:
        return "初始步骤"
    summaries = []
    for s in steps_so_far[-3:]:
        atype = s.get("action_type", "?")
        if atype == "answer":
            summaries.append("给出答案")
        else:
            tp = s.get("think", "")[:80].replace("\n", " ")
            summaries.append(f"[{atype}] {tp}")
    return " -> ".join(summaries)


def parse_trajectory(messages: list, source_index: int) -> list:
    """将一条完整轨迹的 messages 解析为 step-level episodes。"""
    episodes = []

    # 找 system prompt（简要摘要）
    system_summary = ""
    for m in messages:
        if m["role"] == "system":
            # 支持 JSON ("name": "X") 和 Python dict ('name': 'X') 两种格式
            tool_names = re.findall(r"['\"]name['\"]:\s*['\"](\w+)['\"]", m["content"])
            if tool_names:
                system_summary = "Tools: " + ", ".join(tool_names)
            else:
                for line in m["content"].split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        system_summary = line[:100]
                        break
            break

    # 找第一个 user 消息中的问题
    question = ""
    for m in messages:
        if m["role"] == "user":
            question = extract_question_from_user_msg(m["content"])
            break

    # 遍历消息，配对 assistant -> user
    parsed_steps = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "assistant":
            think, action, action_type = parse_think_and_action(msg["content"])

            observation = ""
            obs_success = True
            is_final = (action_type == "answer")

            if i + 1 < len(messages) and messages[i + 1]["role"] == "user":
                observation, obs_success = parse_observation(messages[i + 1]["content"])
                i += 2
            else:
                i += 1

            step_data = {
                "think": think,
                "action": action,
                "action_type": action_type,
                "observation": observation,
                "observation_success": obs_success,
            }
            parsed_steps.append(step_data)

            episode = {
                "episode_id": str(uuid.uuid4()),
                "source_index": source_index,
                "step_index": len(parsed_steps) - 1,
                "question": question,
                "system_prompt_summary": system_summary,
                "think": think,
                "action": action,
                "action_type": action_type,
                "observation": observation,
                "observation_success": obs_success,
                "is_final_step": is_final,
                "full_trajectory_length": len(messages),
                "preceding_steps_summary": summarize_preceding_steps(parsed_steps[:-1]),
            }
            episodes.append(episode)
        else:
            i += 1
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input SFT JSON file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--limit", type=int, default=None, help="Max trajectories to process")
    parser.add_argument("--verbose", action="store_true", help="Print sample episodes to stdout")
    args = parser.parse_args()

    print(f"[INFO] Loading: {args.input}")
    with open(args.input, "r") as f:
        data = json.load(f)
    print(f"[INFO] Total: {len(data)}")

    if args.limit:
        data = data[:args.limit]

    all_episodes = []
    stats = {"total_trajectories": len(data), "total_episodes": 0, "action_types": {}}

    for idx, item in enumerate(data):
        messages = item.get("messages", [])
        if not messages:
            continue
        episodes = parse_trajectory(messages, idx)
        all_episodes.extend(episodes)
        for ep in episodes:
            at = ep["action_type"]
            stats["action_types"][at] = stats["action_types"].get(at, 0) + 1

    stats["total_episodes"] = len(all_episodes)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Wrote {len(all_episodes)} episodes")
    print(f"[STATS] {json.dumps(stats, ensure_ascii=False)}")

    if args.verbose and all_episodes:
        for ep in all_episodes[:2]:
            print(json.dumps(ep, ensure_ascii=False, indent=2))
            print("-" * 40)


if __name__ == "__main__":
    main()
