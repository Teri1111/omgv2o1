# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import logging

# Configure basic logging. Library users can override the configuration as needed.
logging.basicConfig(level=logging.INFO)

WHITE_LIST_RELATIONS = set([
    "type.object.type",
    "type.object.name",
    "type.object.description",
    "type.object.image",
])

def _extract_ordered_blocks(solution_str):
    if solution_str is None:
        return []
    pattern = r'<\|im_start\|>(assistant|tool|user)\n(.*?)<\|im_end\|>'
    return re.findall(pattern, solution_str, re.DOTALL)

def _extract_relations_from_text(text):
    if not text:
        return set()
    relations = set()
    for token in re.findall(r'[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.]*[A-Za-z0-9_]', text):
        if re.match(r'^(?:ns:)?[mg]\.[0-9a-z_]+$', token):
            continue
        relations.add(token)
    return relations

def compute_relation_hallucination_penalty(solution_str, enable=False, penalty_strength=0.2):
    """
    关系幻觉惩罚：若某次 <tool_call> 参数中的关系未在之前所有 observation 中出现，则计一次幻觉。
    每出现一次，惩罚 penalty_strength（默认 0.2）。
    """
    if not enable or solution_str is None:
        return 0.0

    ordered_blocks = _extract_ordered_blocks(solution_str)
    observed_relations = set()
    hallucination_count = 0
    white_list_relations = WHITE_LIST_RELATIONS

    for role, content in ordered_blocks:
        if role in ['tool', 'user']:
            tool_resp_matches = re.findall(r'<tool_response>(.*?)</tool_response>', content, re.DOTALL)
            if tool_resp_matches:
                for tool_resp in tool_resp_matches:
                    observed_relations.update(_extract_relations_from_text(tool_resp))
            else:
                observed_relations.update(_extract_relations_from_text(content))
        elif role == 'assistant':
            tool_call_matches = re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
            for tool_call in tool_call_matches:
                params_relations = _extract_relations_from_text(tool_call)
                unseen = {r for r in params_relations if (r not in observed_relations and r not in white_list_relations)}
                hallucination_count += len(unseen)
    return -penalty_strength * hallucination_count

def compute_timeout_penalty(solution_str, enable=False, penalty_strength=0.2):
    """
    超时查询惩罚：检查所有 observation（tool 块）中是否出现
    "API request timed out" 字样（例如 "xxx API request timed out after xxx seconds"）。
    每出现一次，惩罚 penalty_strength。
    """
    if not enable or solution_str is None:
        return 0.0
    ordered_blocks = _extract_ordered_blocks(solution_str)
    timeout_count = 0
    timeout_pattern = re.compile(r'api request timed out', re.IGNORECASE)
    for role, content in ordered_blocks:
        if role in ['tool', 'user']:
            tool_resp_matches = re.findall(r'<tool_response>(.*?)</tool_response>', content, re.DOTALL)
            if tool_resp_matches:
                for tool_resp in tool_resp_matches:
                    if timeout_pattern.search(tool_resp):
                        timeout_count += 1
            else:
                if timeout_pattern.search(content):
                    timeout_count += 1
    return -penalty_strength * timeout_count

def normalize_answer(s):
    """
    对答案字符串进行标准化处理：
    1. 转为小写
    2. 去除首尾空白
    3. 移除标点符号
    4. 合并多余空白

    Args:
        s (str): 待标准化的字符串

    Returns:
        str: 处理后的标准化结果
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r"[{}]".format(re.escape(string.punctuation)), "", s)
    s = " ".join(s.split())
    return s


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0.0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1.0
            break
    return score

def hit_check(prediction, golden_answers):
    score = 0.0
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    try:
        prediction = eval(prediction)
    except:
        # print(prediction)
        return score
    
    for pred in prediction:
        if pred in golden_answers:
            score = 1.0
            break
    return score

def hit_check_loose(prediction, golden_answers):
    score = 0.0
    if not isinstance(golden_answers, list):
        golden_answers = golden_answers.tolist()
    try:
        prediction = eval(prediction)
    except:
        return score
    
    for pred in prediction:
        if pred in golden_answers:
            score = 1.0
            break
    for answer in golden_answers:
        for pred in prediction:
            if answer in pred:
                score = 1.0
                break
        if score > 0.0:
            break
    return score
    
def f_beta(prediction_str, golden_answers, beta):
    if not isinstance(golden_answers, list):
        golden_answers = golden_answers.tolist()

    try:
        prediction = eval(prediction_str)
    except Exception:
        return 0.0

    pred_set = set(prediction)
    true_set = set(golden_answers)

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        score = 0.0
    else:
        score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)

    if score == 0.0:
        pred_set = {item.lower() for item in prediction}
        true_set = {item.lower() for item in golden_answers}

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall != 0:
            score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)

    return score
def extract_solution(solution_str):
    answer_pattern = r'<answer>.*?\{([^}]*)\}.*?</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def compute_score_format(solution_str):
    if solution_str is None:
        return 0.0
    
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)

        format_reward = 0.0
        
        # If no blocks found, return 0
        if not assistant_blocks:
            return 0.0
        
        for i, assistant_block in enumerate(assistant_blocks[:-1]):
            if assistant_block.count('<think>') == 1 and assistant_block.count('</think>') == 1 and assistant_block.count('<tool_call>') == 1 and assistant_block.count('</tool_call>') == 1:
                think_match = re.search(r'<think>(.*?)</think>(.*?)<tool_call>(.*?)</tool_call>', assistant_block, re.DOTALL)
                if think_match:
                    format_reward += 0.1
                else:
                    format_reward -= 0.1
        logging.debug("Format reward after checking assistant blocks: %s", format_reward)
        if len(assistant_blocks) > 1:
            format_reward = format_reward / (len(assistant_blocks) - 1)
    except Exception as e:
        logging.debug("Error in compute_score_format: %s", e)
        return 0.0
    
    return format_reward

def compute_score_format_boxed(solution_str):
    if solution_str is None:
        return 0.0
    
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)

        format_reward = 0.0
        
        if not assistant_blocks:
            return 0.0
        
        if assistant_blocks:
            last_assistant_block = assistant_blocks[-1]
            think_answer_match = re.search(
                r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$',
                last_assistant_block,
                re.DOTALL
            )
            if think_answer_match:
                format_reward += 0.1
    except Exception as e:
        logging.debug("Error in compute_score_format_boxed: %s", e)
        return 0.0
    
    return format_reward

def compute_score_answer(solution_str, ground_truth):
    """The scoring function for exact match (EM) with format reward.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    
    Returns:
        float: Total reward score (format reward + answer reward)
    """
    if not isinstance(ground_truth, list):
        ground_truth = ground_truth.tolist()
    if not ground_truth:
        return 1.0
    if solution_str is None:
        return 0.0
    
    try:
        # Extract answer from <answer> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        logging.debug("Extracted answer: %s", answer)
        logging.debug("Ground truth: %s", ground_truth)
        answer_reward = 0.0
        
        if answer is not None:
            if hit_check_loose(answer, ground_truth):
                answer_reward = 1.0
        
        if answer_reward == 0.0:
            if hit_check_loose(solution_str, ground_truth):
                answer_reward = 0.2
    except Exception as e:
        logging.debug("Error in compute_score_answer: %s", e)
        return 0.0
    
    return answer_reward

def compute_score_answer_fbeta(solution_str, ground_truth, beta=1.0):
    """使用F-beta分数计算答案奖励
    
    Args:
        solution_str: 预测答案文本
        ground_truth: 正确答案列表
    
    Returns:
        float: F-beta分数 (beta=1, 即F1分数)
    """
    if not isinstance(ground_truth, list):
        ground_truth = ground_truth.tolist()
    if not ground_truth:
        return 1.0
    if solution_str is None:
        return 0.0
    
    try:
        # 提取最后一个assistant块的答案
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        logging.debug("solution_str: %s", solution_str)
        logging.debug("predict answer: %s", answer)
        
        if answer is None:
            return 0.0
        
        score = f_beta(answer, ground_truth, beta)
        return score
        
    except Exception as e:
        logging.debug("Error compute_score_answer: %s", e)
        return 0.0

def compute_score_format_answer_fbeta(solution_str, ground_truth, beta=1.0, enable_relation_hallucination=False, relation_hallucination_penalty=0.2, enable_timeout_penalty=False, timeout_penalty_strength=0.2):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    # print('[DEBUG] compute_score_format_answer called with solution_str:', solution_str)
    if beta is None:
        beta = 1.0
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        boxed_reward = compute_score_format_boxed(solution_str)
        answer_reward = compute_score_answer_fbeta(solution_str, ground_truth, beta)
        rel_penalty = compute_relation_hallucination_penalty(
            solution_str,
            enable=enable_relation_hallucination,
            penalty_strength=relation_hallucination_penalty,
        )
        timeout_penalty = compute_timeout_penalty(
            solution_str,
            enable=enable_timeout_penalty,
            penalty_strength=timeout_penalty_strength,
        )

        format_reward = min(format_reward, 0)
        rel_penalty = max(rel_penalty, -0.5)
        timeout_penalty = max(timeout_penalty, -0.5)
        return min(format_reward + boxed_reward + answer_reward + rel_penalty + timeout_penalty, 1.0)
    except Exception as e:
        logging.debug("Error in compute_score_format_answer_fbeta: %s", e)
        return 0.0


