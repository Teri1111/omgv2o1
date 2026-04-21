def _default_compute_score_format(data_source, solution_str, extra_info=None):
    if data_source == 'hotpotqa/hotpot_qa' or data_source == 'bdsaglam/musique' or data_source == 'xanhho/2WikiMultihopQA':
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_format(solution_str)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score_format(solution_str)
    elif 'webqsp' in data_source or 'cwq' in data_source or 'grailqa' in data_source or data_source in ['webqsp', 'cwq', 'grailqa']:
        from . import kbqa
        res = kbqa.compute_score_format(solution_str)
    elif data_source == 'BytedTsinghua-SIA/DAPO-Math-17k':
        from . import retool
        res = retool.compute_score_format(solution_str)
    else:
        raise NotImplementedError
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
def _default_compute_score_answer(data_source, solution_str, ground_truth, extra_info=None, beta=1.0):
    if data_source == 'hotpotqa/hotpot_qa' or data_source == 'bdsaglam/musique' or data_source == 'xanhho/2WikiMultihopQA':
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_em(solution_str, ground_truth)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score_answer(solution_str, ground_truth)
    elif 'webqsp' in data_source or 'cwq' in data_source or 'grailqa' in data_source or data_source in ['webqsp', 'cwq', 'grailqa']:
        from . import kbqa
        res = kbqa.compute_score_answer_fbeta(solution_str, ground_truth, beta)
    elif data_source == 'BytedTsinghua-SIA/DAPO-Math-17k':
        from . import retool
        res = retool.compute_score_answer(solution_str, ground_truth)
    else:
        raise NotImplementedError
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
def _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info=None, beta=1.0, **kwargs):
    if data_source == 'hotpotqa/hotpot_qa' or data_source == 'bdsaglam/musique' or data_source == 'xanhho/2WikiMultihopQA':
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_format_answer(solution_str, ground_truth)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score_format_answer(solution_str, ground_truth)
    elif 'webqsp' in data_source or 'cwq' in data_source or 'grailqa' in data_source or data_source in ['webqsp', 'cwq', 'grailqa']:
        from . import kbqa
        # 透传关系幻觉相关超参（若 extra_info 中提供 reward_kwargs 则优先使用）
        enable_rel = kwargs.get('enable_relation_hallucination', False)
        rel_penalty = kwargs.get('relation_hallucination_penalty', 0.2)
        enable_timeout = kwargs.get('enable_timeout_penalty', False)
        timeout_strength = kwargs.get('timeout_penalty_strength', 0.2)
        
        if isinstance(extra_info, dict):
            rk = extra_info.get('reward_kwargs') or {}
            enable_rel = rk.get('enable_relation_hallucination', enable_rel)
            rel_penalty = rk.get('relation_hallucination_penalty', rel_penalty)
            enable_timeout = rk.get('enable_timeout_penalty', enable_timeout)
            timeout_strength = rk.get('timeout_penalty_strength', timeout_strength)
        res = kbqa.compute_score_format_answer_fbeta(
            solution_str,
            ground_truth,
            beta,
            enable_relation_hallucination=enable_rel,
            relation_hallucination_penalty=rel_penalty,
            enable_timeout_penalty=enable_timeout,
            timeout_penalty_strength=timeout_strength,
        )
    elif data_source == 'BytedTsinghua-SIA/DAPO-Math-17k':
        from . import retool
        res = retool.compute_score_format_answer(solution_str, ground_truth)
    else:
        raise NotImplementedError
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    beta = kwargs.get('beta', 1.0)
    result = {
        "score": _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info, **kwargs),
        "answer_score": _default_compute_score_answer(data_source, solution_str, ground_truth, extra_info, beta),
        "format": _default_compute_score_format(data_source, solution_str, extra_info),
    }
    
    if 'webqsp' in data_source or 'cwq' in data_source or 'grailqa' in data_source or data_source in ['webqsp', 'cwq', 'grailqa']:
        from . import kbqa
        
        enable_rel = kwargs.get('enable_relation_hallucination', False)
        rel_penalty = kwargs.get('relation_hallucination_penalty', 0.2)
        enable_timeout = kwargs.get('enable_timeout_penalty', False)
        timeout_strength = kwargs.get('timeout_penalty_strength', 0.2)
        
        if isinstance(extra_info, dict):
            rk = extra_info.get('reward_kwargs') or {}
            enable_rel = rk.get('enable_relation_hallucination', enable_rel)
            rel_penalty = rk.get('relation_hallucination_penalty', rel_penalty)
            enable_timeout = rk.get('enable_timeout_penalty', enable_timeout)
            timeout_strength = rk.get('timeout_penalty_strength', timeout_strength)
        
        if enable_rel:
            relation_penalty = kbqa.compute_relation_hallucination_penalty(
                solution_str,
                enable=enable_rel,
                penalty_strength=rel_penalty,
            )
            result["relation_hallucination_penalty"] = relation_penalty
        
        if enable_timeout:
            timeout_penalty = kbqa.compute_timeout_penalty(
                solution_str,
                enable=enable_timeout,
                penalty_strength=timeout_strength,
            )
            result["timeout_penalty"] = timeout_penalty
    
    return result