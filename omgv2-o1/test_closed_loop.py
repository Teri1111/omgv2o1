#!/usr/bin/env python3
"""
End-to-end closed loop test for OMGv2.
Includes path answer + merge_answers + T5 candidate path bias.
Supports --llm flag to use LLMGuidedAgent instead of GreedyAgent.
Supports --trace to print full trace per sample.
Supports --trace-export PATH to export structured JSON trace.
Supports --beam N to use BeamAgent with beam_width=N.
"""

import sys
import os
import json
from collections import defaultdict


from reasoning.subgraph import SubgraphBuilder
from reasoning.agent import GreedyAgent, LLMGuidedAgent, TraceCollector
from reasoning.beam_agent import BeamAgent
from skills.validate_syntax import validate_syntax
from executor.sparql_executor import execute_lf, get_sparql_endpoint, is_sparql_available

# P2F-3: import compute_f1 from local evaluate module
from evaluate import compute_f1


def print_trace(idx, trace_data):
    """Pretty-print trace for a sample to stdout."""
    print("  [TRACE] Sample %d" % idx)
    print("  [TRACE] T5 alignment: %s" % json.dumps(trace_data.get("t5_alignment"), indent=4, default=str))
    print("  [TRACE] Candidate relations: %s" % trace_data.get("candidate_relations"))
    sg = trace_data.get("subgraph_build", {})
    print("  [TRACE] Subgraph build: type=%s count=%d" % (
        sg.get("build_input_type", "?"), sg.get("build_input_count", 0)))
    if sg.get("added_edges"):
        print("  [TRACE]   added_edges (%d): first=%s ... last=%s" % (
            len(sg["added_edges"]), sg["added_edges"][0], sg["added_edges"][-1]))
    for step in trace_data.get("steps", []):
        sn = step.get("step_num", "?")
        print("  [TRACE] Step %d: entity_before=%s action=%s chosen_relation=%s entity_after=%s" % (
            sn, step.get("current_entity_before"), step.get("action"),
            step.get("chosen_relation"), step.get("current_entity_after")))
        if step.get("outgoing_raw_edges"):
            print("  [TRACE]   outgoing_raw_edges(%d): %s" % (
                len(step["outgoing_raw_edges"]), step["outgoing_raw_edges"][:3]))
        if step.get("incoming_raw_edges"):
            print("  [TRACE]   incoming_raw_edges(%d): %s" % (
                len(step["incoming_raw_edges"]), step["incoming_raw_edges"][:3]))
        print("  [TRACE]   filtered_out=%s" % step.get("filtered_out"))
        print("  [TRACE]   filtered_in=%s" % step.get("filtered_in"))
        if step.get("llm_trace"):
            lt = step["llm_trace"]
            print("  [TRACE]   LLM: http_called=%s parsed_action=%s parsed_relation=%s parsed_choice=%s" % (
                lt.get("http_called"), lt.get("parsed_action"), lt.get("parsed_relation"), lt.get("parsed_choice")))
            if lt.get("fallback_reason"):
                print("  [TRACE]   LLM fallback: %s" % lt["fallback_reason"])
    print("  [TRACE] Path replay: %s" % trace_data.get("path_replay"))
    print("  [TRACE] Final selected_relations: %s" % trace_data.get("final_selected_relations"))


def run_test(num_samples=5, use_llm=False, trace=False, trace_export=None, beam_width=None, llm_first=False, show_stats=False, eval_mode="hit", output_path=None, resume=False):
    data_path = os.environ.get("TEST_DATA_PATH", "/data/gt/omg/data/CWQ/search_mid/CWQ_context_test.json")
    t5_path = os.environ.get("TEST_T5_PATH", "/data/gt/omg/data/CWQ/t5_search_output/CWQ_final_test.json")

    sparql_available = is_sparql_available()

    with open(t5_path, "r") as f:
        t5_data = json.load(f)
    t5_by_idx = {}
    for item in t5_data:
        t5_by_idx[item["index"]] = item

    print("=" * 70)
    print("OMGv2 Closed Loop Test")
    print("=" * 70)
    if beam_width:
        mode_label = "BEAM (width=%d)" % beam_width
    elif use_llm:
        mode_label = "LLM-GUIDED"
    else:
        mode_label = "GREEDY (LLM-First)" if llm_first else "GREEDY (baseline)"
    print("Mode: " + mode_label)
    if eval_mode == "full":
        print("Eval mode: full (HIT + F1/P/R/EM)")
    if trace:
        print("Tracing: ENABLED (full trace per sample)")
    if trace_export:
        print("Trace export: " + trace_export)
    status = "AVAILABLE" if sparql_available else "NOT AVAILABLE"
    print("SPARQL endpoint: %s (%s)" % (status, get_sparql_endpoint()))
    print("Loading %d samples" % num_samples)
    print()

    with open(data_path, "r") as f:
        samples = json.load(f)[:num_samples]

    results = []
    completed_ids = set()
    if resume and output_path and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    completed_ids.add(result.get("question_id"))
        print(f"Resuming: {len(completed_ids)} samples already completed")
    all_traces = []
    if beam_width:
        AgentClass = BeamAgent
    elif use_llm:
        AgentClass = LLMGuidedAgent
    else:
        AgentClass = GreedyAgent

    for idx, sample in enumerate(samples):
        question = sample["question"]
        start_entity = sample["start_entity"]
        triplets = sample["triplets"]
        golden_answers = set(sample.get("answer_id", []))
        sample_idx = sample.get("index", idx)
        if sample_idx in completed_ids:
            continue
        t5_entry = t5_by_idx.get(sample_idx, {})
        candidate_paths = t5_entry.get("context", [])

        print("--- Sample %d/%d ---" % (idx + 1, num_samples))
        print("Q: " + question)
        print("Entity: " + start_entity)
        print("Golden: " + str(golden_answers))
        if candidate_paths:
            top1 = candidate_paths[0]
            top1_path = top1[0] if isinstance(top1, list) else top1
            print("T5 top-1: " + top1_path[:200])
        else:
            print("T5 top-1: (none)")

        # 1. Build subgraph (with optional tracing)
        subgraph = SubgraphBuilder()
        sg_trace = {}
        if trace:
            subgraph.set_trace(sg_trace)
        subgraph.build_from_triplets(triplets)
        print("Subgraph: " + str(subgraph))

        # 2. Run agent with T5 candidate paths (with optional tracing)
        trace_collector = TraceCollector() if trace else None
        if trace_collector:
            trace_collector.record_raw_sample(question, start_entity, triplets, golden_answers, sample_idx)
        if trace and sg_trace:
            trace_collector.record_subgraph_build(sg_trace)
        agent_kwargs = dict(
            question=question,
            entities=[start_entity],
            subgraph=subgraph,
            candidate_paths=candidate_paths,
            trace_collector=trace_collector,
        )
        if beam_width:
            agent_kwargs['beam_width'] = beam_width
        agent_kwargs['llm_first'] = llm_first
        agent = AgentClass(**agent_kwargs)
        func_list, sexpr = agent.run()

        print("Function list (%d steps):" % len(func_list))
        for f in func_list:
            print("  " + f)
        print("S-expression: " + sexpr)

        path_answers = agent.path_answer
        print("Path answer: " + str(path_answers))
        print("Selected relations: " + str(agent.selected_relations))

        if use_llm and hasattr(agent, '_llm_chooser_calls'):
            http_r = getattr(agent, '_llm_http_requests', '?')
            print("LLM tool chooser invocations: %d, HTTP requests: %s, fallbacks: %d" % (
                agent._llm_chooser_calls, http_r, agent._llm_chooser_fallbacks))

        # Syntax validation
        syntax_result = validate_syntax(sexpr)
        syntax_status = "OK" if syntax_result["valid"] else "FAIL: " + str(syntax_result["error"])
        print("Syntax: " + syntax_status)
        if trace_collector:
            trace_collector.data["syntax_validation"] = syntax_result

        # Execute SPARQL
        lf_answers = set()
        exec_result = None
        if sparql_available and syntax_result["valid"]:
            try:
                exec_result = execute_lf(sexpr)
                lf_answers = set(exec_result["answers"])
                print("LF SPARQL answers: " + str(lf_answers))
                if exec_result.get("sparql"):
                    print("SPARQL: " + exec_result["sparql"][:200])
                if exec_result.get("error"):
                    print("Error: " + exec_result["error"])
            except Exception as e:
                exec_result = {"sexpr": sexpr, "sparql": "", "answers": [], "error": str(e)}
                print("SPARQL execution error: " + str(e))
        else:
            print("(SPARQL execution skipped)")
            exec_result = {"sexpr": sexpr, "sparql": "", "answers": [], "error": "skipped" if not syntax_result["valid"] else "endpoint_unavailable"}
        if trace_collector:
            trace_collector.data["lf_execution"] = exec_result

        # Merge: LF answers first, path answers supplement
        merged = list(lf_answers)
        seen = set(lf_answers)
        for a in path_answers:
            if a not in seen:
                merged.append(a)
                seen.add(a)
        print("Merged answers: " + str(merged))

        lf_hit = len(lf_answers & golden_answers) > 0 if golden_answers else None
        merged_hit = len(set(merged) & golden_answers) > 0 if golden_answers else None
        # hit@1: exactly 1 answer returned AND it matches golden
        lf_hit1 = (len(lf_answers) == 1 and lf_hit) if golden_answers else None
        merged_hit1 = (len(merged) == 1 and merged_hit) if golden_answers else None
        if lf_hit is not None:
            print("LF HIT: " + str(lf_hit))
        if merged_hit is not None:
            print("Merged HIT: " + str(merged_hit))
        if lf_hit1 is not None:
            print("LF HIT@1: " + str(lf_hit1) + " (answers=%d)" % len(lf_answers))
        if merged_hit1 is not None:
            print("Merged HIT@1: " + str(merged_hit1) + " (answers=%d)" % len(merged))

        # P2F-3: F1/P/R/EM computation in full eval mode
        lf_f1_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        merged_f1_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        lf_em = False
        merged_em = False
        if eval_mode == "full":
            lf_f1_metrics = compute_f1(lf_answers, golden_answers)
            merged_f1_metrics = compute_f1(set(merged), golden_answers)
            lf_em = (lf_answers == golden_answers) if golden_answers else (len(lf_answers) == 0)
            merged_em = (set(merged) == golden_answers) if golden_answers else (len(merged) == 0)
            print("LF F1: %.4f (P=%.4f R=%.4f) EM=%s" % (
                lf_f1_metrics["f1"], lf_f1_metrics["precision"], lf_f1_metrics["recall"], lf_em))
            print("Merged F1: %.4f (P=%.4f R=%.4f) EM=%s" % (
                merged_f1_metrics["f1"], merged_f1_metrics["precision"], merged_f1_metrics["recall"], merged_em))
            if output_path:
                result_line = {
                    "question_id": sample_idx,
                    "question": question,
                    "predicted_answers": sorted(list(merged)),
                    "golden_answers": sorted(list(golden_answers)),
                    "lf_sexpr": sexpr,
                    "merged_sexpr": "",
                    "f1": merged_f1_metrics["f1"],
                    "precision": merged_f1_metrics["precision"],
                    "recall": merged_f1_metrics["recall"],
                    "em": merged_em,
                    "hit": merged_hit,
                }
                with open(output_path, "a") as f:
                    f.write(json.dumps(result_line) + "\n")
        if trace_collector:
            trace_collector.data["path_answers"] = list(path_answers)
            trace_collector.data["merged_answers"] = merged
            trace_collector.data["lf_hit"] = lf_hit
            trace_collector.data["merged_hit"] = merged_hit
            trace_collector.data["lf_hit1"] = lf_hit1
            trace_collector.data["merged_hit1"] = merged_hit1
            trace_collector.data["lf_answer_count"] = len(lf_answers)
            trace_collector.data["merged_answer_count"] = len(merged)

        # Print trace if enabled
        sample_trace = None
        if trace and trace_collector:
            sample_trace = trace_collector.data
            print_trace(idx, sample_trace)
            all_traces.append({
                "sample_idx": sample_idx,
                "question": question,
                "start_entity": start_entity,
                "golden_answers": list(golden_answers),
                "trace": sample_trace,
            })

        print()

        results.append({
            "idx": idx,
            "question": question,
            "sexpr": sexpr,
            "syntax_ok": syntax_result["valid"],
            "lf_answers": list(lf_answers),
            "path_answers": list(path_answers),
            "merged_answers": merged,
            "golden": list(golden_answers),
            "lf_hit": lf_hit,
            "merged_hit": merged_hit,
            "lf_hit1": lf_hit1,
            "merged_hit1": merged_hit1,
            "lf_answer_count": len(lf_answers),
            "merged_answer_count": len(merged),
            "lf_f1": lf_f1_metrics["f1"],
            "lf_precision": lf_f1_metrics["precision"],
            "lf_recall": lf_f1_metrics["recall"],
            "lf_em": lf_em,
            "merged_f1": merged_f1_metrics["f1"],
            "merged_precision": merged_f1_metrics["precision"],
            "merged_recall": merged_f1_metrics["recall"],
            "merged_em": merged_em,
        })

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    syntax_ok_count = sum(1 for r in results if r["syntax_ok"])
    lf_hit_count = sum(1 for r in results if r["lf_hit"] is True)
    merged_hit_count = sum(1 for r in results if r["merged_hit"] is True)
    lf_hit1_count = sum(1 for r in results if r.get("lf_hit1") is True)
    merged_hit1_count = sum(1 for r in results if r.get("merged_hit1") is True)
    evaluable = sum(1 for r in results if r["lf_hit"] is not None)

    # Answer count stats
    lf_counts = [r["lf_answer_count"] for r in results if r["lf_hit"] is not None]
    merged_counts = [r["merged_answer_count"] for r in results if r["lf_hit"] is not None]

    print("Syntax OK: %d/%d" % (syntax_ok_count, len(results)))
    if evaluable > 0:
        print("LF hit rate: %d/%d (%.1f%%)" % (lf_hit_count, evaluable, 100.0 * lf_hit_count / evaluable))
        print("Merged hit rate: %d/%d (%.1f%%)" % (merged_hit_count, evaluable, 100.0 * merged_hit_count / evaluable))
        print("LF HIT@1: %d/%d (%.1f%%)" % (lf_hit1_count, evaluable, 100.0 * lf_hit1_count / evaluable))
        print("Merged HIT@1: %d/%d (%.1f%%)" % (merged_hit1_count, evaluable, 100.0 * merged_hit1_count / evaluable))
        print("LF answer count: avg=%.1f median=%d min=%d max=%d" % (
            sum(lf_counts) / len(lf_counts), sorted(lf_counts)[len(lf_counts)//2],
            min(lf_counts), max(lf_counts)))
        print("Merged answer count: avg=%.1f median=%d min=%d max=%d" % (
            sum(merged_counts) / len(merged_counts), sorted(merged_counts)[len(merged_counts)//2],
            min(merged_counts), max(merged_counts)))

    # P2F-3: Aggregate F1/P/R/EM table in full eval mode
    if eval_mode == "full" and evaluable > 0:
        lf_f1_sum = sum(r["lf_f1"] for r in results if r["lf_hit"] is not None)
        lf_p_sum = sum(r["lf_precision"] for r in results if r["lf_hit"] is not None)
        lf_r_sum = sum(r["lf_recall"] for r in results if r["lf_hit"] is not None)
        lf_em_count = sum(1 for r in results if r.get("lf_em") is True)
        merged_f1_sum = sum(r["merged_f1"] for r in results if r["lf_hit"] is not None)
        merged_p_sum = sum(r["merged_precision"] for r in results if r["lf_hit"] is not None)
        merged_r_sum = sum(r["merged_recall"] for r in results if r["lf_hit"] is not None)
        merged_em_count = sum(1 for r in results if r.get("merged_em") is True)

        print()
        print("-" * 70)
        print("F1 / PRECISION / RECALL / EM (set-level, eval_mode=full)")
        print("-" * 70)
        print("  %-12s  %8s  %8s  %8s  %8s" % ("", "F1", "Prec", "Recall", "EM"))
        print("  %-12s  %8.4f  %8.4f  %8.4f  %8.1f%%" % (
            "LF",
            lf_f1_sum / evaluable,
            lf_p_sum / evaluable,
            lf_r_sum / evaluable,
            100.0 * lf_em_count / evaluable))
        print("  %-12s  %8.4f  %8.4f  %8.4f  %8.1f%%" % (
            "Merged",
            merged_f1_sum / evaluable,
            merged_p_sum / evaluable,
            merged_r_sum / evaluable,
            100.0 * merged_em_count / evaluable))
        print("-" * 70)

    # Export traces if requested
    if trace_export and all_traces:
        export_data = {
            "mode": mode_label,
            "num_samples": num_samples,
            "traces": all_traces,
            "summary": {
                "syntax_ok": syntax_ok_count,
                "total": len(results),
                "lf_hit_count": lf_hit_count,
                "merged_hit_count": merged_hit_count,
                "lf_hit1_count": lf_hit1_count,
                "merged_hit1_count": merged_hit1_count,
                "evaluable": evaluable,
            },
        }
        with open(trace_export, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        print("Traces exported to: " + trace_export)

    # T15: Print aggregated statistics if requested
    if show_stats and all_traces:
        print("\n" + "=" * 70)
        print("AGGREGATED STATISTICS")
        print("=" * 70)
        
        # Aggregate tool usage statistics
        tool_usage = defaultdict(int)
        action_dist = defaultdict(int)
        multi_tool_calls = 0
        active_consultations = 0
        
        for trace_entry in all_traces:
            trace_data = trace_entry.get("trace", {})
            
            # Aggregate tool usage
            tool_stats = trace_data.get("tool_usage_stats", {})
            for tool, count in tool_stats.items():
                tool_usage[tool] += count
            
            # Aggregate action distribution
            action_stats = trace_data.get("action_distribution", {})
            for action, count in action_stats.items():
                action_dist[action] += count
            
            # Aggregate multi-tool calls
            multi_tool_calls += trace_data.get("multi_tool_chain_stats", {}).get("total_chains", 0)
            
            # Aggregate active consultations
            active_consultations += trace_data.get("kb_usage_stats", {}).get("active_consultations", 0)
        
        print("\nTool Usage:")
        for tool, count in sorted(tool_usage.items()):
            print("  {}: {} calls".format(tool, count))
        
        print("\nAction Distribution:")
        for action, count in sorted(action_dist.items()):
            if count > 0:
                print("  {}: {} times".format(action, count))
        
        print("\nKey Metrics:")
        print("  Multi-tool calls: {}".format(multi_tool_calls))
        print("  Active KB consultations: {}".format(active_consultations))
        print("  Multi-tool call rate: {:.1f}%".format(multi_tool_calls / len(all_traces) * 100 if all_traces else 0))
        print("  Active consultation rate: {:.1f}%".format(active_consultations / len(all_traces) * 100 if all_traces else 0))

    return results


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python3 test_closed_loop.py [OPTIONS] [NUM_SAMPLES]")
        print()
        print("Options:")
        print("  --llm               Use LLMGuidedAgent")
        print("  --llm-first         Try LLM first, fallback to greedy")
        print("  --beam N            Use BeamAgent with beam_width=N")
        print("  --trace             Print full trace per sample")
        print("  --trace-export PATH Export structured JSON trace")
        print("  --stats             Print aggregated statistics")
        print("  --eval-mode MODE    Eval mode: 'hit' (default) or 'full'")
        print("  --full-test         Run full dataset in full eval mode")
        print("  --output PATH       Output file for results JSON")
        print("  --resume            Resume from last output file")
        print("  --help, -h          Show this help message")
        print()
        print("NUM_SAMPLES defaults to 5.")
        sys.exit(0)
    use_llm = "--llm" in sys.argv
    llm_first = "--llm-first" in sys.argv
    beam_width = None
    trace = "--trace" in sys.argv
    trace_export = None
    show_stats = "--stats" in sys.argv
    eval_mode = "hit"
    full_test = False
    output_path = None
    resume = "--resume" in sys.argv
    remaining_args = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--trace-export" and i + 1 < len(sys.argv):
            trace_export = sys.argv[i + 1]
            trace = True  # implied
            i += 2
            continue
        elif sys.argv[i] == "--beam" and i + 1 < len(sys.argv):
            beam_width = int(sys.argv[i + 1])
            i += 2
            continue
        elif sys.argv[i] == "--eval-mode" and i + 1 < len(sys.argv):
            eval_mode = sys.argv[i + 1]
            i += 2
            continue
        elif sys.argv[i] == "--full-test":
            eval_mode = "full"
            full_test = True
            i += 1
            continue
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
            continue

        elif sys.argv[i] not in ("--llm", "--trace", "--llm-first", "--stats", "--resume"):
            remaining_args.append(sys.argv[i])
        i += 1
    n = int(remaining_args[0]) if remaining_args else 5
    # full-test 模式：加载全集样本数
    if full_test:
        _data_path = os.environ.get("TEST_DATA_PATH", "/data/gt/omg/data/CWQ/search_mid/CWQ_context_test.json")
        with open(_data_path) as f:
            n = len(json.load(f))
    run_test(n, use_llm=use_llm, trace=trace, trace_export=trace_export, beam_width=beam_width, llm_first=llm_first, show_stats=show_stats, eval_mode=eval_mode, output_path=output_path, resume=resume)