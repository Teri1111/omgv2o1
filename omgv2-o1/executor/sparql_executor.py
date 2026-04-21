"""
SPARQL execution for OMGv2.
Wraps omgv1 utils for LF -> SPARQL -> execution pipeline.
"""

import sys
import os

# Add omgv1 to path for imports
_OMGV1_DIR = "/data/gt/omgv1"
if _OMGV1_DIR not in sys.path:
    sys.path.insert(0, _OMGV1_DIR)

from executor.logic_form_util_local import lisp_to_sparql  # noqa: E402

# SPARQL execution via HTTP (simpler than ODBC, no .so dependency)
import requests  # noqa: E402

SPARQL_ENDPOINT = "http://localhost:3001/sparql"

def get_sparql_endpoint():
    "Return the authoritative SPARQL endpoint used by OMGv2."
    return SPARQL_ENDPOINT


def is_sparql_available(timeout: int = 5):
    "Check availability of the configured SPARQL endpoint."
    try:
        resp = requests.get(get_sparql_endpoint(), timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False



def execute_query_http(sparql_query: str):
    """Execute SPARQL via HTTP endpoint. Returns list of result values."""
    try:
        resp = requests.post(
            get_sparql_endpoint(),
            data={"query": sparql_query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()
        rtn = []
        for binding in results["results"]["bindings"]:
            for var in binding:
                val = binding[var]["value"]
                val = val.replace("http://rdf.freebase.com/ns/", "")
                rtn.append(val)
        return rtn
    except Exception as e:
        return []


def execute_lf(sexpr: str, verbose: bool = False):
    """
    Full pipeline: S-expression -> SPARQL -> execute -> results.
    Returns dict with sexpr, sparql, answers.
    """
    result = {"sexpr": sexpr, "sparql": "", "answers": [], "error": None}
    try:
        sparql_query = lisp_to_sparql(sexpr)
        result["sparql"] = sparql_query
        if verbose:
            print("[SPARQL]", sparql_query[:300])
        answers = execute_query_http(sparql_query)
        result["answers"] = answers
    except Exception as e:
        result["error"] = str(e)
    return result


def execute_function_list(func_list):
    """Execute a function_list (list of assignment strings)."""
    from skills.lf_construction import function_list_to_sexpr
    sexpr = function_list_to_sexpr(func_list)
    return execute_lf(sexpr)


def test_execute_function_list(func_list):
    """
    Test-execute: append STOP and check if results are non-empty.
    Returns the answer list (empty if failed).
    """
    from skills.lf_construction import function_list_to_sexpr
    sexpr = function_list_to_sexpr(func_list)
    if sexpr == "@BAD_EXPRESSION":
        return []
    result = execute_lf(sexpr)
    answers = result["answers"]
    return [a for a in answers if a != "None"]


def denormalize_lf(sexpr: str, entity_name_map: dict) -> str:
    """
    Replace entity names in LF with MIDs.
    KBQA-o1 generates LF with names like [ Lou Seal ], need to convert to m.xxx
    """
    for name, mid in entity_name_map.items():
        sexpr = sexpr.replace("[ {} ]".format(name), mid)
        sexpr = sexpr.replace(" {} ".format(name), " {} ".format(mid))
    return sexpr
