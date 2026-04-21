import argparse
import json
from typing import List, Tuple, Dict, Optional
import logging
import os
import re
import ast
import string
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_xsd(item):
    if isinstance(item, list):
        return sorted(list(set(clean_xsd(i) for i in item)))
    if isinstance(item, str) and "^^xsd:" in item:
        item = item.split("^^xsd:")[0].strip('"')
    return str(item)

def normalize_answer(s):
    s = s.strip().lower()

    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", s) or re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        s = s.replace(',', '')
        return s
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc_except(text, keep=set(".-:/")):
        return "".join(ch for ch in text if (ch not in string.punctuation or ch in keep))

    s = remove_punc_except(s)
    s = remove_articles(s)
    s = white_space_fix(s)
    return s

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

def get_subcategory_key(item: dict, dataset_name: str) -> Optional[str]:
    if dataset_name == 'grailqa':
        return str(item.get('level', 'unknown'))
    elif dataset_name in ['webqsp']:
        return str(item.get('data_type', 'unknown'))
    elif dataset_name in['cwq']:
        return str(item.get('compositionality_type', 'unknown'))
    else:
        return None

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

def evaluate(dataset: List[dict], dataset_name: str) -> Optional[Dict[str, float]]:
    total = 0
    sum_prec, sum_rec, sum_f1 = 0.0, 0.0, 0.0
    sum_hit1, sum_rand_hit1 = 0.0, 0.0
    sum_em_any_norm = 0.0

    cat_aggr: Dict[str, Dict[str, float]] = {}

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

        cat_key = get_subcategory_key(item, dataset_name)
        if cat_key is not None:
            slot = cat_aggr.setdefault(cat_key, {"count": 0, "sum_f1": 0.0, "sum_rhit1": 0.0})
            slot["count"] += 1
            slot["sum_f1"] += f1
            slot["sum_rhit1"] += rand_hit1

    if total == 0:
        return None

    summary: Dict[str, float] = {
        "num_questions_evaluated": total,
        "average_precision": sum_prec / total,
        "average_recall": sum_rec / total,
        "average_f1": sum_f1 / total,
        "hit_at_1": sum_hit1 / total,
        "random_hit_at_1": sum_rand_hit1 / total,
        "exact_match_any_normalized": sum_em_any_norm / total
    }

    if cat_aggr:
        by_category = {}
        for k, v in cat_aggr.items():
            cnt = max(1, int(v["count"]))
            by_category[k] = {
                "count": int(v["count"]),
                "average_f1": v["sum_f1"] / cnt,
                "rhit1": v["sum_rhit1"] / cnt
            }
        summary["by_category"] = by_category

    return summary

def find_matching_files(root_dir: str, dataset_name: str, prefix: str, suffix: str, recursive: bool = True) -> List[str]:
    base_dir = os.path.join(root_dir, dataset_name)
    if not os.path.isdir(base_dir):
        logging.error(f"Directory not found: {base_dir}")
        return []
    matched = []
    walker = os.walk(base_dir) if recursive else [(base_dir, [], os.listdir(base_dir))]
    for dirpath, _, filenames in walker:
        for fn in filenames:
            if fn.startswith(prefix) and fn.endswith(suffix):
                matched.append(os.path.join(dirpath, fn))
    matched.sort()
    return matched

def evaluate_one_file(input_path: str, dataset_name: str) -> Optional[Dict[str, float]]:
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return None
    with open(input_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            logging.error(f"JSON load error for {input_path}: {e}")
            return None
    if not isinstance(data, list):
        logging.error(f"Input JSON must be a list of items: {input_path}")
        return None
    summary = evaluate(data, dataset_name)
    return summary

def save_summary_for_file(input_path: str, summary: Dict[str, float], output_file: str = "", save=True) -> str:
    if output_file:
        out_path = output_file
    else:
        base, _ = os.path.splitext(input_path)
        out_path = base + "_evaluation_summary.json"
    if save:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        logging.info(f"Evaluation summary saved to {out_path}")
    return out_path

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions (single file or batch by prefix/suffix).")

    parser.add_argument("-d", "--dataset_name", type=str, required=True,
                        choices=["webqsp", "cwq", "grailqa"], help="Dataset name.")

    parser.add_argument("-i", "--input_file", type=str, default="",
                        help="Path to input JSON file that contains predictions.")

    parser.add_argument("--root_dir", type=str, help="Root directory that contains <dataset_name>/ ...")
    parser.add_argument("--prefix", type=str, help="Filename prefix to match (startswith).")
    parser.add_argument("--suffix", type=str, help="Filename suffix to match (endswith). E.g., .json")
    parser.add_argument("--non_recursive", action="store_true", help="Do NOT search recursively (default: recursive).")

    parser.add_argument("-o", "--output_file", type=str, default="",
                        help="(Single file mode) Path to save evaluation summary JSON. "
                             "Defaults to <input>_evaluation_summary.json")
    parser.add_argument("--batch_summary", type=str, default="",
                        help="(Batch mode) Save an aggregated summary JSON across files. "
                             "Defaults to <root>/<dataset>/batch_eval_summary_<timestamp>.json")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.input_file:
        summary = evaluate_one_file(args.input_file, args.dataset_name)
        if summary is None:
            logging.warning("No items were evaluated.")
            return
        print("\n" + "=" * 30)
        print("--- Final Evaluation Results ---")
        for k, v in summary.items():
            if k == "by_category" and isinstance(v, dict):
                print("by_category:")
                for cat, stats in v.items():
                    print(f"  - {cat}: count={stats.get('count', 0)}, "
                          f"average_f1={stats.get('average_f1', 0.0):.4f}, "
                          f"rhit1={stats.get('rhit1', 0.0):.4f}")
                continue
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        print("=" * 30 + "\n")
        save_summary_for_file(args.input_file, summary, args.output_file)
        return

    missing = [k for k in ["root_dir", "prefix", "suffix"] if not getattr(args, k)]
    if missing:
        logging.error(f"Batch mode requires --root_dir, --prefix, --suffix (missing: {', '.join(missing)})")
        return

    recursive = not args.non_recursive
    files = find_matching_files(args.root_dir, args.dataset_name, args.prefix, args.suffix, recursive=recursive)
    if not files:
        logging.warning("No matching files found.")
        return

    logging.info(f"Found {len(files)} file(s). Start evaluating...")
    aggregated = []
    for fp in files:
        logging.info(f"Evaluating: {fp}")
        summary = evaluate_one_file(fp, args.dataset_name)
        if summary is None:
            logging.warning(f"Skip (no summary): {fp}")
            continue
        print("\n" + "-" * 30)
        print(os.path.basename(fp))
        for k, v in summary.items():
            if k == "by_category" and isinstance(v, dict):
                print("by_category:")
                for cat, stats in v.items():
                    print(f"  - {cat}: count={stats.get('count', 0)}, "
                          f"average_f1={stats.get('average_f1', 0.0):.4f}, "
                          f"rhit1={stats.get('rhit1', 0.0):.4f}")
                continue
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        print("-" * 30 + "\n")
        per_file_out = save_summary_for_file(fp, summary, save=False)
        row = {"file": fp, "summary_path": per_file_out}
        row.update(summary)
        aggregated.append(row)

    if not aggregated:
        logging.warning("No summaries produced in batch.")
        return

    dataset_dir = os.path.join(args.root_dir, args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    if args.batch_summary:
        agg_out = args.batch_summary
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        agg_out = os.path.join(dataset_dir, f"batch_eval_summary_{ts}.json")
    with open(agg_out, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=4, ensure_ascii=False)
    logging.info(f"Batch aggregated summary saved to {agg_out}")

if __name__ == "__main__":
    main()