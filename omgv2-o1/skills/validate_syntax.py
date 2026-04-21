"""
LF syntax validation for OMGv2.
Lightweight checks without querying KB.
"""

import re


def validate_syntax(sexpr: str) -> dict:
    """
    Validate S-expression syntax.
    Returns: {"valid": bool, "error": str or None}
    """
    if not sexpr or sexpr == "@BAD_EXPRESSION":
        return {"valid": False, "error": "empty or bad expression"}
    
    # Check parentheses balance
    depth = 0
    for i, c in enumerate(sexpr):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if depth < 0:
            return {"valid": False, "error": "unmatched closing paren at position {}".format(i)}
    
    if depth != 0:
        return {"valid": False, "error": "unbalanced parentheses (depth={})".format(depth)}
    
    # Check valid operators - including (R relation) for reverse
    valid_ops = {"JOIN", "AND", "ARGMAX", "ARGMIN", "CMP", "TC", "COUNT", "le", "lt", "ge", "gt", "R"}
    # Find all (OPERATOR patterns
    tokens = re.findall(r"\((\w+)", sexpr)
    for token in tokens:
        if token not in valid_ops:
            return {"valid": False, "error": "unknown operator: {}".format(token)}
    
    return {"valid": True, "error": None}
