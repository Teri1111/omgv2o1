import argparse
import json
from typing import List, Tuple
import logging
import os
import time
from datetime import datetime
import re
import ast
import sys
import random
import string

from tqdm import tqdm
from openai import OpenAI

import pebble
from concurrent.futures import TimeoutError

from tool_generation import ToolGenerationConfig, ToolGenerationManager
from tool_env import ToolEnv
from kb_tool import SearchGraphPatterns, ExecuteSPARQL, SearchTypes
from utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AGENT_KBQA_PATH = "/mnt/bn/maliva-gen-ai-v2/lishilong/projects/Agent-R2/AgentKBQA"

system_prompt = """#Tools

You are an expert in knowledge base query language SPARQL programming. The user gives a question, and you need to iteratively call the tool to continuously improve the SPARQL query until it can get the answer to the question.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{'type': 'function', 'function': {'name': 'SearchGraphPatterns', 'description': 'This tool searches for relevant one-hop and two-hop subgraphs tied to a specified variable. It queries subgraphs where the chosen variable (?x, assuming the SPARQL query begins with "SELECT DISTINCT ?x WHERE") appears as the head or tail entity and returns them collectively. The semantic parameter indicates the expected predicate semantics. When provided, the tool ranks the subgraphs based on these semantics. If unspecified, it returns the complete subgraph.', 'parameters': {'type': 'object', 'properties': {'sparql': {'type': 'string', 'description': 'SPARQL query'}, 'semantic': {'type': 'string', 'description': 'The semantic parameter represents the expected predicate semantics.'}}, 'required': ['sparql']}}}
{'type': 'function', 'function': {'name': 'ExecuteSPARQL', 'description': 'This tool executes a SPARQL query and returns the results.', 'parameters': {'type': 'object', 'properties': {'sparql': {'type': 'string', 'description': 'SPARQL query'}}, 'required': ['sparql']}}}
{'type': 'function', 'function': {'name': 'SearchTypes', 'description': 'Search the knowledge base for matching semantic types, used to initiate queries from a type when no topic entities are available, or to find a type to refine the query when multiple entities are returned. When use the type, please give the sparql as: SELECT DISTINCT ?x WHERE { ?x ns:type.object.type ns:<type_name> }', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'the semantic of type to search for'}}, 'required': ['query']}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""

user_start_message = """When you encounter a complex question, you should break it down into several sub-questions and answer them step by step. You can use the tools provided. You can use the tool as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.
When you have the final answer, you can output the answer in the python list format inside <answer> tag, such as: <answer> the answer is \\boxed{{[...]}} </answer>.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>

Question: {question}
Topic Entities: {topic_entity_text}
Assistant:
"""

tool_response_template = """<tool_response>
{tool_response}
</tool_response>"""

def qwen_max_llm_func(messages, config):
    client = OpenAI(
        api_key="sk-",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=messages,
        extra_body={"enable_thinking": False},
        **config,
    )
    return completion.choices[0].message.content

def custom_sft_llm_func(messages, config):
    client = OpenAI(api_key="0",base_url="http://0.0.0.0:18000/v1")
    result = client.chat.completions.create(
        messages=messages,
        model="agent",
        timeout=60.0,
        **config)
    return result.choices[0].message.content

def deepseek_llm_func(messages, config):
    client = OpenAI(
        api_key="sk-",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.chat.completions.create(
        model="deepseek-v3",
        messages=messages,
        **config,
        stream=False)
    return response.choices[0].message.content

def qwq(messages, config):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-v1-",
    )
    response = client.chat.completions.create(
        extra_headers={},
        extra_body={},
        model="qwen/qwq-32b:free",
        messages=messages,
        **config)
    return response.choices[0].message.content

def parse_args():
    parser = argparse.ArgumentParser(description="Run Search O1 for various datasets and models.")
    parser.add_argument(
        '-d',
        '--dataset_name',
        type=str,
        required=True,
        choices=['webqsp', 'cwq', 'grailqa'],
        help="Name of the dataset to rollout."
    )
    parser.add_argument('--model_name', type=str, default='agent', help="Model to use.")
    parser.add_argument('-m', '--max_turn', type=int, default=12, help="Maximum number of turns.")
    parser.add_argument('-r', '--rollout_num', type=int, default=1, help="Number of rollouts.")
    parser.add_argument('--save_file', type=str, required=True, help="Where to save.")
    parser.add_argument('-n', '--num_processes', type=int, default=10, help="Number of processes to use.")
    parser.add_argument('--mode', type=str, default='test', help="to evaluate or generate.")
    parser.add_argument('--temperature', type=float, default=1.0, help="--temperature.")
    parser.add_argument('--top_p', type=float, default=0.9, help="top p.")
    parser.add_argument('--timeout', type=int, default=90, help="Timeout in seconds for a single rollout task.")
    return parser.parse_args()

def process_rollout(generation_manager, env, sample, rollout_idx, model_name, save_file, sample_idx, max_turns, dataset_name):
    logging.info(f"Starting processing for sample_idx: {sample_idx}, rollout_idx: {rollout_idx}")
    topic_entities = sample.get('topic_entities', {})
    topic_entity_text = ", ".join([f"{val}({key})" for key, val in topic_entities.items()])

    question_key = 'question' if 'question' in sample else 'ProcessedQuestion'
    question = sample[question_key]

    local_start_message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_start_message.format(question=question, topic_entity_text=topic_entity_text)}
    ]

    time.sleep(random.uniform(1.0, 3.0))

    gold_answers = []
    if dataset_name == 'webqsp':
        gold_answers = sample.get('answers', [])
    elif dataset_name == 'cwq':
        gold_answers = sample.get('answers', [])
    else: # grailqa
        answer_key = "answer"
        id_key = 'entity_name'
        arg_key = 'answer_argument'
        gold_answers = [ans.get(id_key) or ans.get(arg_key) for ans in sample.get(answer_key, [])]

    rollout_result = generation_manager.run_llm_api_loop(local_start_message, envs=[env], answers=gold_answers)

    MAX_RESPONSE_LENGTH = 8192
    for message in rollout_result:
        if 'content' in message and isinstance(message['content'], str) and len(message['content']) > MAX_RESPONSE_LENGTH:
            pid = os.getpid()
            logging.warning(f"[Worker PID: {pid}] Response length > {MAX_RESPONSE_LENGTH}. Truncating.")
            message['content'] = message['content'][:MAX_RESPONSE_LENGTH] + "... [TRUNCATED]"

    save_dir = os.path.abspath(f'./data/{dataset_name}/{model_name}/{save_file}/rollouts')
    os.makedirs(save_dir, exist_ok=True)
    rollout_filename = os.path.join(save_dir, f"sample_{sample_idx}_rollout_{rollout_idx}_temp.json")

    try:
        with open(rollout_filename, 'w', encoding='utf-8') as f:
            json.dump(rollout_result, f, ensure_ascii=False, indent=4)
        logging.info(f"Successfully saved rollout {rollout_idx} for sample {sample_idx} to {rollout_filename}")
    except Exception as e:
        logging.error(f"Failed to save rollout {rollout_idx} for sample {sample_idx}: {e}")
        raise

    return f"Success: sample {sample_idx}, rollout {rollout_idx}"

def clean_xsd(item):
    if isinstance(item, list):
        return sorted(list(set(clean_xsd(i) for i in item)))
    if isinstance(item, str) and "^^xsd:" in item:
        item = item.split("^^xsd:")[0].strip('"')
    return str(item)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_final_answer_from_dialog(dialog: List[dict]) -> List[str]:
    if not dialog:
        return []
    for message in reversed(dialog):
        content = message.get("content", "")
        if message.get("role") == "assistant" and "<answer>" in content:
            match = re.search(r"\\boxed\{(.*?)\}", content, re.DOTALL)
            if match:
                answer_str = match.group(1).strip()
                try:
                    pred_ans = ast.literal_eval(answer_str)
                    if isinstance(pred_ans, list):
                        return [clean_xsd(item) for item in pred_ans if item]
                    else:
                        return [clean_xsd(pred_ans)] if pred_ans else []
                except (ValueError, SyntaxError):
                    logging.warning(f"Could not parse answer string: {answer_str}")
                    return []
    return []

def get_predictions(item: dict) -> List[str]:
    if 'pred_answers' in item and isinstance(item['pred_answers'], list):
        return [str(x) for x in item['pred_answers'] if x is not None]

    if 'prediction' in item:
        pred = item['prediction']
        if isinstance(pred, list):
            return [str(x) for x in pred if x is not None]
        if isinstance(pred, str):
            return [pred]

    if 'rollout_results' in item and isinstance(item['rollout_results'], list) and item['rollout_results']:
        dialog = item['rollout_results'][0]
        return extract_final_answer_from_dialog(dialog)

    return []

def get_golds(item: dict, dataset_name: str) -> List[str]:
    if dataset_name in ['webqsp', 'cwq']:
        return [str(x) for x in item.get('answers', [])]
    elif dataset_name == 'grailqa':
        golds = []
        for ans in item.get('answer', []):
            val = ans.get('entity_name') or ans.get('answer_argument')
            if val is not None:
                golds.append(str(val))
        return golds
    else:
        return [str(x) for x in item.get('answers', [])]

def prf1(gold: List[str], pred: List[str]) -> Tuple[float, float, float]:
    gold_norm = [str(g).lower() for g in gold]
    pred_norm = [str(p).lower() for p in pred]
    if not gold_norm and not pred_norm:
        return 1.0, 1.0, 1.0
    if not pred_norm:
        return 0.0, 0.0, 0.0
    gold_set, pred_set = set(gold_norm), set(pred_norm)
    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def hit_at_1(gold: List[str], pred: List[str]) -> float:
    if not gold or not pred:
        return 0.0
    gold_set = set(g.lower() for g in gold)
    return 1.0 if pred[0].lower() in gold_set else 0.0

def random_hit_at_1_expected(gold: List[str], pred: List[str]) -> float:
    if not gold or not pred:
        return 0.0
    gold_set = set(g.lower() for g in gold)
    pred_list = [p.lower() for p in pred]
    matches = sum(1 for p in pred_list if p in gold_set)
    return matches / len(pred_list)

def em_any_normalized(gold: List[str], pred: List[str]) -> float:
    if not gold or not pred:
        return 1.0 if (not gold and not pred) else 0.0
    gold_norm = [normalize_answer(g) for g in gold]
    gold_set = set(gold_norm)
    for p in pred:
        if normalize_answer(p) in gold_set:
            return 1.0
    return 0.0

def evaluate(dataset: List[dict], dataset_name: str):
    total = 0
    sum_prec, sum_rec, sum_f1 = 0.0, 0.0, 0.0
    sum_hit1, sum_rand_hit1 = 0.0, 0.0
    sum_em_any_norm = 0.0

    for item in tqdm(dataset, desc="Evaluating"):
        gold = get_golds(item, dataset_name)
        pred = get_predictions(item)

        total += 1
        prec, rec, f1 = prf1(gold, pred)
        hit1 = hit_at_1(gold, pred)
        rand_hit1 = random_hit_at_1_expected(gold, pred)
        em_any = em_any_normalized(gold, pred)

        sum_prec += prec
        sum_rec += rec
        sum_f1 += f1
        sum_hit1 += hit1
        sum_rand_hit1 += rand_hit1
        sum_em_any_norm += em_any

    if total == 0:
        return None

    return {
        "num_questions_evaluated": total,
        "average_precision": sum_prec / total,
        "average_recall": sum_rec / total,
        "average_f1": sum_f1 / total,
        "hit_at_1": sum_hit1 / total,
        "random_hit_at_1": sum_rand_hit1 / total,
        "exact_match_any_normalized": sum_em_any_norm / total
    }


def extract_mids(sparql_query):
    pattern = r'ns:[mg]\.[0-9a-zA-Z_]+' if 'ns:' in sparql_query else r':[mg]\.[0-9a-zA-Z_]+'
    replace_pattern = 'ns:' if 'ns:' in sparql_query else ':'
    mids = re.findall(pattern, sparql_query)
    return [mid.replace(replace_pattern, '') for mid in mids]

def get_name_from_mid(mid, fb):
    return fb.get_mid_name(mid)

def preprocess_topic_entities(dataset):
    sys.path.insert(0, AGENT_KBQA_PATH)
    fb_client = None
    try:
        from tool.client_freebase import FreebaseClient
        fb_client = FreebaseClient(end_point="http://localhost:8890/sparql")
        logging.info("Successfully initialized FreebaseClient.")

        for item in tqdm(dataset, desc="Processing topic entities"):
            topic_entities = {}
            sparql = ""
            if 'sparql' in item:
                sparql = item['sparql']
            elif 'sparql_query' in item:
                sparql = item['sparql_query']
            elif item.get('Parses'):
                sparql = item.get('Parses', [{}])[0].get('Sparql', "")

            if not sparql:
                logging.warning(f"Item {item.get('id', 'unknown')} has no SPARQL query.")
                continue

            mids = extract_mids(sparql)
            for mid in mids:
                name = get_name_from_mid(mid, fb_client)
                if name:
                    topic_entities[mid] = name
            item['topic_entities'] = topic_entities

    except ImportError as e:
        logging.error(f"Failed to import FreebaseClient: {e}. Please check the path '{AGENT_KBQA_PATH}'.")
    except Exception as e:
        logging.error(f"An error occurred during topic entity processing: {e}")
    finally:
        if AGENT_KBQA_PATH in sys.path:
            sys.path.remove(AGENT_KBQA_PATH)
    return dataset

def worker_wrapper(args):
    return process_rollout(*args)


def main():
    args = parse_args()

    logging.info(f"Loading dataset: {args.dataset_name}, mode: {args.mode}")
    DATA_PATH_MAP = {
        'webqsp': {'test': './evaluation/test_official/webqsp_filtered.json', 'train': './data/webqsp/WebQSP.train.searchPattern_v3.{level}.json'},
        'cwq': {'test': './evaluation/test_official/cwq_filtered.json', 'train': './data/cwq/CWQ.train.searchPattern_v3.{level}.json'},
        'grailqa': {'test': './evaluation/test_official/grailqa_all_with_topic_entities.json', 'train': './data/grailqa/GrailQA.train.{level}.json'}
    }
    filepath = DATA_PATH_MAP[args.dataset_name]['test'] if args.mode == 'test' else DATA_PATH_MAP[args.dataset_name]['train'].format(level=args.save_file)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logging.info(f"Loaded {len(dataset)} samples from {filepath}")
    except FileNotFoundError:
        logging.error(f"Dataset file not found at {filepath}.")
        return

    if len(dataset) > 100:
        logging.warning(f"Dataset contains {len(dataset)} samples. Processing only the first 100 for testing.")
        if args.dataset_name == 'cwq':
            pass
        else:
            dataset = dataset[:100]

    if args.dataset_name in ["grailqa"]:
        preprocess_topic_entities(dataset)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_file_name = f"{args.save_file}_{timestamp}"

    tools = [SearchGraphPatterns(), ExecuteSPARQL(), SearchTypes()]
    env = ToolEnv(tools=tools, max_turns=args.max_turn)

    llm_functions = {
        'deepseek': deepseek_llm_func, 'qwq': qwq, 'qwen_max': qwen_max_llm_func, 'agent': custom_sft_llm_func
    }
    llm_func = llm_functions.get(args.model_name, custom_sft_llm_func)

    config_params = {
        "max_turns": args.max_turn,
        "generate_config": {"temperature": args.temperature, "max_tokens": 512, "top_p": args.top_p},
        "tool_custom_response_template": tool_response_template
    }
    config_params["generate_config"]["stop"] = ["</tool_call>", "</answer>"]

    config = ToolGenerationConfig(**config_params)
    generation_manager = ToolGenerationManager(llm_func=llm_func, config=config)

    tasks_args_list = []
    for sample_idx, sample in enumerate(dataset):
        for rollout_idx in range(args.rollout_num):
            args_tuple = (
                generation_manager, env, sample, rollout_idx,
                args.model_name, save_file_name, sample_idx,
                args.max_turn, args.dataset_name
            )
            tasks_args_list.append(args_tuple)

    logging.info(f"Starting parallel processing with pebble. Timeout set to {args.timeout}s per task.")
    try:
        with pebble.ProcessPool(max_workers=args.num_processes, max_tasks=1) as pool:
            future = pool.map(worker_wrapper, tasks_args_list, timeout=args.timeout)
            results_iterator = future.result()
            for i in tqdm(range(len(tasks_args_list)), desc="Executing rollouts"):
                try:
                    next(results_iterator)
                except TimeoutError:
                    failed_args = tasks_args_list[i]
                    sample_idx = failed_args[6]
                    rollout_idx = failed_args[3]
                    logging.error(f"Task for sample {sample_idx}, rollout {rollout_idx} timed out after {args.timeout} seconds and was terminated.")
                except Exception as e:
                    failed_args = tasks_args_list[i]
                    sample_idx = failed_args[6]
                    rollout_idx = failed_args[3]
                    logging.error(f"Task for sample {sample_idx}, rollout {rollout_idx} failed with an exception: {e}")

    except Exception as e:
        logging.error(f"A critical error occurred in the main process during multiprocessing: {e}")
        logging.error("The pool was terminated. Some results may be lost.")

    save_dir_final = os.path.abspath(f'./data/{args.dataset_name}')
    final_filename = os.path.join(save_dir_final, f"{args.model_name}_{args.rollout_num}r_{save_file_name}_total{len(dataset)}.json")
    os.makedirs(save_dir_final, exist_ok=True)

    aggregated_data = []
    temp_dir = os.path.abspath(f'./data/{args.dataset_name}/{args.model_name}/{save_file_name}/rollouts')

    if os.path.exists(temp_dir):
        for sample_idx, sample in enumerate(tqdm(dataset, desc="Aggregating rollouts")):
            sample_copy = sample.copy()
            rollout_results = []

            for rollout_idx in range(args.rollout_num):
                rollout_filename = os.path.join(temp_dir, f"sample_{sample_idx}_rollout_{rollout_idx}_temp.json")
                if os.path.exists(rollout_filename):
                    try:
                        with open(rollout_filename, 'r', encoding='utf-8') as f:
                            rollout_results.append(json.load(f))
                        os.remove(rollout_filename)
                    except Exception as e:
                        logging.error(f"Failed to process {rollout_filename}: {e}")
                else:
                    logging.warning(f"Temp file not found for sample {sample_idx}, rollout {rollout_idx}: {rollout_filename}")

            sample_copy['rollout_results'] = rollout_results
            aggregated_data.append(sample_copy)

        try:
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
                parent_dir = os.path.dirname(temp_dir)
                if not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
        except OSError as e:
            logging.warning(f"Could not clean up temporary directories: {e}")

    try:
        with open(final_filename, 'w', encoding='utf-8') as f:
            json.dump(aggregated_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Successfully saved aggregated data to {final_filename}")
    except Exception as e:
        logging.error(f"Failed to save final aggregated data: {e}")
        raise

    if aggregated_data:
        summary = evaluate(aggregated_data, args.dataset_name)
        if summary is None:
            logging.warning("No items were evaluated.")
        else:
            print("\n" + "=" * 30)
            print("--- Final Evaluation Results ---")
            for k, v in summary.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
            print("=" * 30 + "\n")

            summary_filename = final_filename.replace('.json', '_evaluation_summary.json')
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            logging.info(f"Evaluation summary saved to {summary_filename}")
    else:
        logging.warning("No data was aggregated, skipping evaluation.")

if __name__ == '__main__':
    main()
