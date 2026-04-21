import os, sys
from collections import namedtuple
from typing import List
import time
import re

from loguru import logger
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from tool.client_freebase import DATATYPE, TYPE, FreebaseClient, add_not_exists_or_exists_filter
from tool.openai import get_embedding, get_embedding_batch
from tool.searchnode_fb import SearchMidName
from tool_prepare.fb_vectorization import (
    CHROMA_CVT_PREDICATE_PAIR_NAME,
    CHROMA_PREDICATE_NAME,
    CHROMA_TYPE_NAME,
    get_chroma_client,
)

subgraph = namedtuple("subgraph", ["p", "fact_triple", "type"])

PREFIX_NS = "PREFIX ns: <http://rdf.freebase.com/ns/> "

# Scan caps for subquery LIMIT (used as fallback for one-hop, primary for two-hop/CVT).
SCAN_LIMIT_FALLBACK = 50000  # one-hop fallback when Phase 1 times out
SCAN_LIMIT_TWO_HOP = 20000
SCAN_LIMIT_CVT = 20000

# Time budget for a single SearchGraphPatterns call.
# If elapsed time exceeds this, stop expanding and return collected patterns so far.
SGP_TIME_BUDGET = 45  # seconds

def _build_predicate_prefilter(pred_var: str) -> str:
    """Predicate pre-filter for SPARQL subqueries.
    Disabled: STRSTARTS filters in subqueries trigger Virtuoso colsearch.c crash
    (GPF: filled dc past end). All predicate filtering now done in Python."""
    return ""

def _is_useful_predicate(p: str) -> bool:
    """Python-side predicate filter (mirrors post-processing in SearchGraphPatterns)."""
    if p.startswith("common.") and p not in ("common.topic.image", "common.topic.type"):
        return False
    if p.startswith("freebase.") or p == "type.object.name":
        return False
    if "common.topic.description" in p:
        return False
    if "has_no_value" in p or "has_value" in p:
        return False
    # Filter out Virtuoso internal URIs (not cleaned by _clean_http)
    if p.startswith("http://") or p.startswith("urn:") or p.startswith("local:") or p.startswith("mailto:"):
        return False
    return True


def _clean_http(text):
    text = text.replace("http://rdf.freebase.com/ns/", "")
    return text


def remove_splitline(text):
    text = [line.strip() for line in text.splitlines()]
    text = " ".join(text)
    return text


def init_fb_actions():
    def SearchNodes(query=None, n_results=10):
        """
        There are cases where multiple MIDs correspond to one surface. Solution: Query by surface, sort based on the score of FACC1, return the corresponding descriptions of different MIDs for distinction.

        return:
            "Oxybutynin Chloride 5 extended release film coated tablet" | Description: Oxybutynin chloride (Mylan Pharmaceuticals), manufactured drug form of ... max 200 chars.
            or
            "xxx" | No description.
        """
        if not query:
            return "Error! query is required."
        query = str(query).strip()

        def _process_desc(desc):
            desc = desc.strip()
            if desc:
                if not desc.endswith("."):
                    desc += "."
                if len(desc) > 100:
                    desc = desc[:100] + "..."
            return desc

        # search by es. get names
        names = SearchMidName(query, size=100)

        results = []
        visited = set()

        # if name is exact match, get descs and concat them (name may be the same).
        # Glastonbury | Description: {desc1} | Description: {desc2} ...
        # cased/uncased: match by lower, search by original.
        for n in names:
            if n in visited:
                continue
            if n.lower() == query.lower():
                desc_res = ""
                descs = fb.get_common_topic_description_by_name(n)
                for desc in descs:
                    desc = _process_desc(desc)
                    if desc:
                        desc_res += f"| Description: {desc}"
                if desc_res:
                    results.append(f'"{n}" {desc_res[:1000]}')
                else:
                    results.append(f'"{n}" | No description.')
                visited.add(n)

        # group by name, get top 1 mid; add desc.
        # e.g. ["Obama", "OBAMA"] -> one group, reduce to one mid, rank by facc1 score.

        for name in names:
            mids = fb.get_mids_facc1(name)
            desc = None
            for mid in mids:
                desc = fb.get_mid_desc(mid)
                if desc:
                    name = fb.get_mid_name(mid)
                    break

            # keep one name in a group.
            if name.lower() in visited or name in visited:
                continue
            visited.add(name.lower())

            if desc:
                desc = _process_desc(desc)
                results.append(f'"{name}" | Description: {desc}')
            else:
                results.append(f'"{name}" | No description.')
            if len(results) == n_results:
                break

        return str(results)

    vec_client = get_chroma_client()
    collection_predicate = vec_client.get_collection(name=CHROMA_PREDICATE_NAME)
    collection_type = vec_client.get_collection(name=CHROMA_TYPE_NAME)
    collection_cvt_predicate_pair = vec_client.get_collection(name=CHROMA_CVT_PREDICATE_PAIR_NAME)

    fb = FreebaseClient(end_point="http://localhost:8890/sparql")

    # Phase-1 timeout: if DISTINCT predicates query exceeds this, fall back to LIMIT.
    PHASE1_TIMEOUT = 15  # seconds

    _phase1_cache = {}  # LRU cache: sparql_txt → bindings (avoids re-running DISTINCT on buffer eviction)

    def _phase1_query(sparql_txt):
        """Run a DISTINCT-predicates query with a short timeout.
        Returns list of bindings on success, or None on timeout/error.
        Results are cached so repeated queries for the same entity skip Phase 1."""
        if sparql_txt in _phase1_cache:
            return _phase1_cache[sparql_txt]
        if sparql_txt in fb._timeout_query:
            return None
        try:
            from SPARQLWrapper import JSON, SPARQLWrapper, POST
            _client = SPARQLWrapper("http://localhost:8890/sparql", returnFormat=JSON)
            _client.setTimeout(PHASE1_TIMEOUT)
            _client.setQuery(sparql_txt)
            _client.setMethod(POST)
            res = _client.query().convert()
            if "results" in res and "bindings" in res["results"]:
                bindings = res["results"]["bindings"]
            else:
                bindings = []
            _phase1_cache[sparql_txt] = bindings
            if len(_phase1_cache) > 2000:
                _oldest = next(iter(_phase1_cache))
                del _phase1_cache[_oldest]
            return bindings
        except Exception as e:
            logger.warning(f"Phase-1 timed out ({PHASE1_TIMEOUT}s), will use LIMIT fallback: {sparql_txt[:120]}")
            return None

    def _get_head_p(sparql, deadline=None):
        """
        e.g. (?c, organization.organization.leadership -> organization.leadership.person, "Terry Collins"@en)
        deadline: absolute time.time() after which we skip expensive queries.
        """
        valid_patterns: List[subgraph] = []
        _idx = sparql.find("{")
        sparql_body = sparql[_idx + 1 :]
        
        pattern = r"SELECT\s+(\?\w+)\s+WHERE"
        match = re.search(pattern, sparql)
        if match:
            variable = match.group(1) 
        else:
            print("No variable in sparql!", sparql)
            return "Error: No variable found in SPARQL."

        # one hop — try two-phase first; fall back to LIMIT if Phase 1 is too slow
        phase1_pin = PREFIX_NS + f"SELECT DISTINCT ?_pin WHERE {{ ?_h ?_pin {variable} . " + sparql_body
        pred_res_pin = _phase1_query(phase1_pin)

        if pred_res_pin is not None:
            # Phase 1 succeeded → Phase 2 with targeted VALUES lookups (full recall)
            useful_pin_uris = [
                item["_pin"]["value"] for item in pred_res_pin
                if _is_useful_predicate(_clean_http(item["_pin"]["value"]))
            ]
            if useful_pin_uris:
                pin_vals = " ".join(f"<{u}>" for u in useful_pin_uris)
                phase2_pin = (
                    PREFIX_NS
                    + f"SELECT (MIN(?_h) as ?_h) ?_pin WHERE {{ ?_h ?_pin {variable} . "
                    + f"FILTER (?_h != {variable}) VALUES ?_pin {{ {pin_vals} }} "
                    + sparql_body
                )
                if "GROUP BY" not in phase2_pin.upper():
                    phase2_pin = phase2_pin.rstrip() + " GROUP BY ?_pin"
                query_res_pin = fb.query(phase2_pin)
                if isinstance(query_res_pin, str) and query_res_pin.startswith("Error"):
                    return query_res_pin
            else:
                query_res_pin = []
        else:
            # Phase 1 timed out → fall back to subquery LIMIT (bounded time, partial recall)
            pin_filter = _build_predicate_prefilter("?_pin")
            inner_pin = f"SELECT ?_h ?_pin WHERE {{ ?_h ?_pin {variable} . FILTER (?_h != {variable}) {pin_filter}" + sparql_body
            if "LIMIT" not in inner_pin.upper():
                inner_pin = inner_pin.rstrip() + f" LIMIT {SCAN_LIMIT_FALLBACK}"
            sparql_qpin = PREFIX_NS + f"SELECT (MIN(?_h) as ?_h) ?_pin WHERE {{ {{ {inner_pin} }} }} GROUP BY ?_pin"
            query_res_pin = fb.query(sparql_qpin)
            if isinstance(query_res_pin, str) and query_res_pin.startswith("Error"):
                return query_res_pin
        for item in query_res_pin:
            if "_h" not in item or "_pin" not in item:
                continue
            h = _clean_http(item["_h"]["value"])
            p = _clean_http(item["_pin"]["value"])

            # entity: m. g.
            if item["_pin"]["type"] == "uri" and h[:2] in ["m.", "g."]:
                _hname = fb.get_mid_name(h)
                if _hname:
                    _fact_triple = f'"{_hname}", {p}, {variable}'
                else:
                    # print(f"Warning!! {t} has no name.")
                    # ignore it.
                    continue
            else:
                # skip non-freebase nodes (e.g. Virtuoso internal URIs)
                continue

            _g = subgraph(
                p=p,
                fact_triple=f"({_fact_triple})",
                type="in",
            )
            valid_patterns.append(_g)

        # two hops — subquery with scan cap (lower limit due to join cost)
        if deadline and time.time() > deadline:
            print(f"[SGP-discover] incoming: skipped two-hop (time budget exceeded), returning {len(valid_patterns)} one-hop incoming", flush=True)
            return valid_patterns
        inner_body_pin2 = (
            f"?_h ?_pin1 ?_c . ?_c ?_pin2 {variable} . FILTER (?_h != {variable}) "
            + _build_predicate_prefilter("?_pin1")
            + sparql_body
        )
        inner_sparql_pin2 = f"SELECT ?_h ?_pin1 ?_pin2 WHERE {{ {inner_body_pin2}"
        if "LIMIT" not in inner_sparql_pin2.upper():
            inner_sparql_pin2 = inner_sparql_pin2.rstrip() + f" LIMIT {SCAN_LIMIT_TWO_HOP}"
        sparql_qpin2 = (
            PREFIX_NS
            + f"SELECT (MIN(?_h) as ?_h) ?_pin1 ?_pin2 WHERE {{ {{ {inner_sparql_pin2} }} }} GROUP BY ?_pin1 ?_pin2"
        )
        query_res_pin2 = fb.query(sparql_qpin2, _print=True)

        if isinstance(query_res_pin2, str) and query_res_pin2.startswith("Error"):
            return valid_patterns
        for item in query_res_pin2:
            if "_h" not in item or "_pin1" not in item or "_pin2" not in item:
                continue
            h = _clean_http(item["_h"]["value"])
            p1 = _clean_http(item["_pin1"]["value"])
            p2 = _clean_http(item["_pin2"]["value"])

            # entity: m. g. (only valid type)
            if h[:2] in ["m.", "g."]:
                _hname = fb.get_mid_name(h)
                if _hname:
                    _fact_triple = f'"{_hname}", {p1} -> {p2}, {variable}'
                else:
                    # print(f"Warning!! {t} has no name.")
                    # ignore it.
                    continue
            _g = subgraph(
                p=f"{p1} -> {p2}",
                fact_triple=f"({_fact_triple})",
                type="in",
            )
            valid_patterns.append(_g)

        return valid_patterns

    def _get_p_tail(sparql, deadline=None):
        """
        possible p:
            uri
            literal
            typed-literal
        deadline: absolute time.time() after which we stop expanding CVTs.
        """

        def _handle_out(item: dict, _var_p: str, _var_t: str, _var="?e") -> subgraph:
            """
            design for cvt.
            Args:
                item: element in Virtuoso query result.
                    - has ?_pout2 and ?_t2
                _var_p: var name of predicate.
                _var_t: var name of tail node.
                _var: var name of head node.
            """
            p = _clean_http(item[_var_p]["value"])
            t = _clean_http(item[_var_t]["value"])

            # entity: m. g.
            if item[_var_p]["type"] == "uri" and t[:2] in ["m.", "g."]:
                _tname = fb.get_mid_name(t)
                if _tname:
                    _fact_triple = f'{_var}, {p}, "{_tname}"'
                else:
                    # print(f"Warning!! {t} has no name.")
                    # ignore it.
                    return None

            # literal
            elif item[_var_t]["type"] == "literal":
                _fact_triple = f'{_var}, {p}, "{t}"'

            # typed-literal
            elif item[_var_t]["type"] == "typed-literal":
                datatype = item[_var_t]["datatype"].split("XMLSchema#")[-1]
                if datatype in DATATYPE:
                    _fact_triple = f'{_var}, {p}, "{t}"^^xsd:{datatype}'
                else:
                    _fact_triple = f"{_var}, {p}, {t}"

            # type
            elif p in TYPE:
                return None

            else:
                if not (t.startswith("http://") or t.startswith("urn:") or t.startswith("local:") or t.startswith("mailto:")):
                    logger.warning(f"Unseen tail node type: {t}, skipping. sparql: {sparql[:120]}")
                return None

            return _fact_triple

        _idx = sparql.find("{")
        sparql_body = sparql[_idx + 1 :]
        pattern = r"SELECT\s+(\?\w+)\s+WHERE"
        match = re.search(pattern, sparql)
        
        if match:
            variable = match.group(1)  
        else:
            print("No variable in sparql!", sparql)
            return "Error: No variable found in SPARQL."

        # Outgoing one-hop — try two-phase first; fall back to LIMIT if Phase 1 is too slow
        phase1_pout = PREFIX_NS + f"SELECT DISTINCT ?_pout WHERE {{ {variable} ?_pout ?_t . " + sparql_body
        phase1_pout = add_ns_prefix(phase1_pout)
        pred_res_pout = _phase1_query(phase1_pout)

        valid_patterns: List[subgraph] = []
        if pred_res_pout is not None:
            # Phase 1 succeeded → Phase 2 with targeted VALUES lookups (full recall)
            useful_pout_uris = [
                item["_pout"]["value"] for item in pred_res_pout
                if _is_useful_predicate(_clean_http(item["_pout"]["value"]))
            ]
            if useful_pout_uris:
                pout_vals = " ".join(f"<{u}>" for u in useful_pout_uris)
                phase2_body = (
                    f"{variable} ?_pout ?_t . FILTER (?_t != {variable}) "
                    + f"VALUES ?_pout {{ {pout_vals} }} "
                    + sparql_body
                )
                sparql_qpout = PREFIX_NS + f"SELECT ?_pout (MIN(?_t) as ?_t) WHERE {{ {phase2_body}"
                sparql_qpout = add_ns_prefix(sparql_qpout)
                if "GROUP BY" not in sparql_qpout.upper():
                    sparql_qpout = sparql_qpout.rstrip() + " GROUP BY ?_pout"
                query_res_pout = fb.query(sparql_qpout)
                if isinstance(query_res_pout, str) and query_res_pout.startswith("Error"):
                    return query_res_pout
            else:
                query_res_pout = []
        else:
            # Phase 1 timed out → fall back to subquery LIMIT (bounded time, partial recall)
            pout_filter = _build_predicate_prefilter("?_pout")
            inner_pout = f"SELECT ?_pout ?_t WHERE {{ {variable} ?_pout ?_t . FILTER (?_t != {variable}) {pout_filter}" + sparql_body
            inner_pout = add_ns_prefix(inner_pout)
            if "LIMIT" not in inner_pout.upper():
                inner_pout = inner_pout.rstrip() + f" LIMIT {SCAN_LIMIT_FALLBACK}"
            sparql_qpout = PREFIX_NS + f"SELECT ?_pout (MIN(?_t) as ?_t) WHERE {{ {{ {inner_pout} }} }} GROUP BY ?_pout"
            query_res_pout = fb.query(sparql_qpout)
            if isinstance(query_res_pout, str) and query_res_pout.startswith("Error"):
                return query_res_pout

        _discovered_cvt_preds = []
        _discovered_onehop_preds = []
        _skipped_cvt = 0
        for item in query_res_pout:
            if "_pout" not in item:
                continue
            p = _clean_http(item["_pout"]["value"])

            # cvt
            if item["_pout"]["type"] == "uri" and fb.is_cvt_predicate(p):
                if deadline and time.time() > deadline:
                    _skipped_cvt += 1
                    continue
                cvt_filter = _build_predicate_prefilter("?_pout2")
                # Sample a few CVT nodes instead of scanning all — same-type CVT
                # records share the same schema, so 5 samples cover all sub-predicates.
                cvt_sample = (
                    f"{{ SELECT ?_cvt WHERE {{ {variable} ns:{p} ?_cvt . {sparql_body} LIMIT 5 }} "
                )
                inner_body_cvt = (
                    cvt_sample
                    + f"?_cvt ?_pout2 ?_t2 . FILTER (?_t2 != {variable}) "
                    + cvt_filter
                )
                inner_sparql_cvt = f"SELECT ?_pout2 ?_t2 WHERE {{ {inner_body_cvt} }}"
                sparql_cvt_out = (
                    PREFIX_NS
                    + f"SELECT ?_pout2 (MIN(?_t2) as ?_t2) WHERE {{ {{ {inner_sparql_cvt} }} }} GROUP BY ?_pout2"
                )
                query_res_cvt_out = fb.query(sparql_cvt_out)
                if isinstance(query_res_cvt_out, str) and query_res_cvt_out.startswith("Error"):
                    print(f"[SGP-discover] CVT sub-query error for {p}: {query_res_cvt_out[:150]}", flush=True)
                    continue
                _discovered_cvt_preds.append(p)
                for item_cvt in query_res_cvt_out:
                    if "_pout2" not in item_cvt:
                        continue
                    p2 = _clean_http(item_cvt["_pout2"]["value"])
                    _fact_triple_cvt = _handle_out(item_cvt, _var_p="_pout2", _var_t="_t2", _var="?cvt")
                    if _fact_triple_cvt:
                        _fact_triple = f"{variable}, {p} -> " + _fact_triple_cvt.replace("?cvt, ", "")
                        # predicates in a cvt.
                        _g = subgraph(
                            p=f"{p} -> {p2}",
                            fact_triple=f"({_fact_triple})",
                            type="out",
                        )
                        valid_patterns.append(_g)
            else:
                _fact_triple = _handle_out(item, _var_p="_pout", _var_t="_t", _var=variable)
                if _fact_triple:
                    _g = subgraph(
                        p=p,
                        fact_triple=f"({_fact_triple})",
                        type="out",
                    )
                    valid_patterns.append(_g)
                    _discovered_onehop_preds.append(p)

        print(
            f"[SGP-discover] outgoing: {len(valid_patterns)} patterns | "
            f"onehop={_discovered_onehop_preds[:10]} | cvt_roots={_discovered_cvt_preds[:10]}"
            + (f" | skipped_cvt={_skipped_cvt}" if _skipped_cvt else ""),
            flush=True,
        )
        return valid_patterns

    def SearchTypes(query=None, n_results=10):
        type_emb = get_embedding(query)
        _predicates_vec = collection_type.query(
                    query_embeddings=type_emb,
                    n_results=n_results,
                )
        results = _predicates_vec['documents'][0]
        return results
    def SearchGraphPatterns(
        sparql: str = None,
        semantic: str = None,
        topN_vec=50,
        topN_return=10,
    ):
        """
        The graph pattern to be queried must start with SELECT ?e WHERE; compatible with Freebase CVT format.
        1. Given ?e, query the one-hop in/out fact triples of that entity.
        2. "xx" -> ?cvt -> ?tail is considered as one hop.
        """
        if not sparql:
            return "Error! SPARQL is required."

        # must has ns:
        if "ns:" not in sparql:
            return "Error! SPARQL must contain ns:."
        if PREFIX_NS not in sparql:
            sparql = PREFIX_NS + sparql

        sparql = sparql.strip().replace(" DISTINCT", "")

        if "SELECT" not in sparql:
            return "Error! SPARQL must start with: SELECT ?xx WHERE. You should use `ExecuteSPARQL` to execute SPARQL statements."

        # Intercept type.object.type full-table scans:
        # queries like "?x type.object.type ns:people.person" without other constraining
        # predicates enumerate millions of rows and always time out.
        if "type.object.type" in sparql:
            _body = sparql[sparql.find("{") + 1 :].strip()
            _ns_preds = re.findall(r'ns:([a-zA-Z_]+(?:\.[a-zA-Z_]+)+)', _body)
            _other_preds = [p for p in _ns_preds if p != "type.object.type"]
            if not _other_preds:
                return (
                    "Error! Scanning all entities of a type via type.object.type is too expensive "
                    "and will time out. Please add specific constraining predicates to narrow the search, "
                    "or use SearchGraphPatterns with a specific entity MID."
                )

        # Intercept type.object.name: resolve name→MID(s) first,
        # then rewrite to VALUES for efficient index lookup.
        # Strategy: FACC1 (popularity-ranked) → type-count ranked Virtuoso → plain Virtuoso LIMIT.
        _name_pat = re.search(
            r'(\?\w+)\s+ns:type\.object\.name\s+"([^"]+)"(?:@en)?\s*\.?\s*',
            sparql,
        )
        if _name_pat:
            _var = _name_pat.group(1)
            _name = _name_pat.group(2)
            _var_clean = _var.lstrip("?")
            _mids = []

            # 1) FACC1: MIDs ordered by web anchor-text frequency (popular entities first)
            try:
                _facc1_mids = fb.get_mids_facc1(_name)
                if _facc1_mids:
                    _mids = _facc1_mids[:5]
                    logger.info(f"[SearchGraphPatterns] Resolved '{_name}' via FACC1 → {_mids}")
            except Exception as _e:
                logger.warning(f"[SearchGraphPatterns] FACC1 lookup failed for '{_name}': {_e}")

            if not _mids:
                # 2) Virtuoso: rank by type count (prominent entities have more types)
                _rank_q = (
                    PREFIX_NS
                    + f'SELECT {_var} (COUNT(?_t) AS ?_cnt) WHERE {{ '
                    + f'{_var} ns:type.object.name "{_name}"@en . {_var} ns:type.object.type ?_t . '
                    + f'}} GROUP BY {_var} ORDER BY DESC(?_cnt) LIMIT 5'
                )
                _mid_res = fb.query(_rank_q)
                if _mid_res and isinstance(_mid_res, list) and len(_mid_res) > 0:
                    _mids = [_clean_http(r[_var_clean]["value"]) for r in _mid_res]
                    logger.info(f"[SearchGraphPatterns] Resolved '{_name}' via Virtuoso type-count → {_mids}")

            if not _mids:
                # 3) Final fallback: plain Virtuoso LIMIT
                _resolve_q = PREFIX_NS + f'SELECT {_var} WHERE {{ {_var} ns:type.object.name "{_name}"@en }} LIMIT 5'
                _mid_res = fb.query(_resolve_q)
                if _mid_res and isinstance(_mid_res, list) and len(_mid_res) > 0:
                    _mids = [_clean_http(r[_var_clean]["value"]) for r in _mid_res]
                    logger.info(f"[SearchGraphPatterns] Resolved '{_name}' via Virtuoso plain LIMIT → {_mids}")

            if _mids:
                _mid_vals = " ".join([f"ns:{m}" for m in _mids])
                sparql = re.sub(
                    r'\?\w+\s+ns:type\.object\.name\s+"[^"]*"(?:@en)?\s*\.?\s*',
                    f"VALUES {_var} {{ {_mid_vals} }} ",
                    sparql,
                    count=1,
                )
                logger.info(f"[SearchGraphPatterns] Rewrote type.object.name '{_name}' → {len(_mids)} MIDs")

        # add head and tail var.
        _idx = sparql.find("{")
        if _idx == -1:
            return "Error! SPARQL must contain {."

        valid_patterns: List[subgraph] = []
        _sgp_start = time.time()
        _deadline = _sgp_start + SGP_TIME_BUDGET

        # ?e -> ?t (outgoing)
        _r = _get_p_tail(sparql, deadline=_deadline)
        if isinstance(_r, str):
            print(f"[SGP] _get_p_tail returned error (continuing to incoming): {_r[:200]}", flush=True)
        else:
            valid_patterns += _r
        visited_p = {i.p for i in valid_patterns}

        # ?h -> ?e (incoming) — skip entirely if time budget exhausted
        if time.time() < _deadline:
            _r = _get_head_p(sparql, deadline=_deadline)
            if isinstance(_r, str):
                print(f"[SGP] _get_head_p returned error (continuing with outgoing patterns): {_r[:200]}", flush=True)
            else:
                valid_patterns += [i for i in _r if i.p not in visited_p]
        else:
            print(f"[SGP] skipping incoming patterns (time budget {SGP_TIME_BUDGET}s exceeded after outgoing)", flush=True)

        # filer. not start with "common.", no has "has_no_value"
        valid_patterns = [
            i
            for i in valid_patterns
            # if not i.p.startswith("common.") and "has_no_value" not in i.p and "has_value" not in i.p
            if (not i.p.startswith("common.") and "has_no_value" not in i.p and "has_value" not in i.p) or i.p in ('common.topic.image', 'common.topic.type')
        ]

        # exclude "type.object.name"
        valid_patterns = [i for i in valid_patterns if i.p != "type.object.name"]
        # exclude "common.topic.description"
        valid_patterns = [i for i in valid_patterns if "common.topic.description" not in i.p]
        # exclude "freebase."
        valid_patterns = [i for i in valid_patterns if "freebase." not in i.p]

        # sort
        valid_patterns.sort(key=lambda x: x.p)

        # semantic ranking: compute embedding similarity between semantic query
        # and each discovered predicate directly (avoids ChromaDB top-N mismatch).
        if semantic and valid_patterns:
            sem_emb = get_embedding(semantic)
            pred_texts = [p.p for p in valid_patterns]
            pred_embs = get_embedding_batch(pred_texts)

            _score_map = {}
            for pattern, emb in zip(valid_patterns, pred_embs):
                dist = sum((a - b) ** 2 for a, b in zip(sem_emb, emb))
                _score_map[pattern.p] = dist

            valid_patterns.sort(key=lambda x: _score_map[x.p])

            top5_info = [f"{v.p}({_score_map[v.p]:.4f})" for v in valid_patterns[:5]]
            print(
                f"[SGP-rank] semantic='{semantic}' | total={len(valid_patterns)} | "
                f"top5=[{', '.join(top5_info)}]",
                flush=True,
            )

        _fact_triple = [i.fact_triple for i in valid_patterns[:topN_return]]
        return "[" + ", ".join(_fact_triple) + "]"


    def add_ns_prefix(query):
        pattern = r'(\s+)(?!ns:)([a-zA-Z_]+(?:\.[a-zA-Z_]+){2})'
        def replacer(match):
            return match.group(1) + 'ns:' + match.group(2)

        result = re.sub(pattern, replacer, query)
        return result




    def ExecuteSPARQL(sparql=None, str_mode=True):
        """
        Replace the "mid" in the returned results with "name".
        """
        if not sparql:
            return "Error! sparql is required."

        if PREFIX_NS not in sparql:
            sparql = PREFIX_NS + sparql

        sparql = sparql.strip()
        sparql = sparql.replace("'", '"')
        sparql = add_ns_prefix(sparql)

        try:
            # add `not exists || exists filter` for .from/.to predicates.
            # dont support all sparqls. may raise error.
            _fix_sparql = add_not_exists_or_exists_filter(sparql)
            results = fb.query(_fix_sparql)
        except Exception as e:
            logger.error(e)
            logger.error(f"Error in add_not_exists_or_exists_filter. raw sparql: {sparql}")
            results = fb.query(sparql)

        if isinstance(results, str) and results.startswith("Error"):
            return results
        if isinstance(results, bool):
            res = results
        elif isinstance(results, list):
            def _resolve_value(item):
                if not isinstance(item, dict):
                    return str(item) if item else ""
                _type = item["type"]
                v = _clean_http(item["value"])
                if _type == "uri" and v[:2] in ["m.", "g."]:
                    v = fb.get_mid_name(v)
                elif _type in ["typed-literal"]:
                    _t = item["datatype"].split("XMLSchema#")[-1]
                    if _t in DATATYPE:
                        v = f'"{v}"^^xsd:{_t}'
                elif _type == "literal":
                    pass
                else:
                    v = _clean_http(v)
                return v or ""

            n_vars = len(results[0]) if results else 0
            if n_vars <= 1:
                res = [_resolve_value(item) for row in results for item in row.values() if _resolve_value(item)]
            else:
                res = []
                for row in results:
                    vals = [_resolve_value(item) for item in row.values()]
                    if any(vals):
                        res.append("(" + ", ".join(vals) + ")")
        else:
            raise Exception(f"Unseen results type: {type(results)}")

        if isinstance(res, list):
            res = list(dict.fromkeys(res))
        if str_mode:
            res = str(res)
            if len(res) > 1000:
                logger.warning(f"Warning!! result is too long. len: {len(res)}. sparql: {sparql}")
                res = res[:1000] + "..."
        return res

    return SearchNodes, SearchTypes, SearchGraphPatterns, ExecuteSPARQL


def test_actions():
    SearchNodes, SearchTypes, SearchGraphPatterns, ExecuteSPARQL = init_fb_actions()

    r = ExecuteSPARQL("SELECT DISTINCT ?x0 AS ?label WHERE { ns:m.01r0zd ns:type.object.name ?x0 . FILTER (langMatches( lang(?x0), \"EN\" ) )}")
    print(r)

if __name__ == "__main__":
    """
    python tools/actions_fb.py
    """
    test_actions()