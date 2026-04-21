"""
S-expression construction functions for OMGv2.
Adapted from KBQA-o1 utils/prompt.py
"""


def START(entity: str):
    return entity


def JOIN(relation: str, expression: str):
    return "(JOIN {} {})".format(relation, expression)


def AND(expression: str, sub_expression: str):
    return "(AND {} {})".format(expression, sub_expression)


def ARG(operator: str, expression: str, relation: str):
    assert operator in ["ARGMAX", "ARGMIN"]
    return "({} {} {})".format(operator, expression, relation)


def CMP(operator: str, relation: str, expression: str):
    return "({} {} {})".format(operator, relation, expression)


def TC(expression: str, relation: str, entity: str):
    return "(TC {} {} {})".format(expression, relation, entity)


def COUNT(expression: str):
    return "(COUNT {})".format(expression)


def STOP(expression: str):
    return expression


exec_globals = {
    "START": START,
    "JOIN": JOIN,
    "AND": AND,
    "ARG": ARG,
    "CMP": CMP,
    "TC": TC,
    "COUNT": COUNT,
    "STOP": STOP,
}


def functions_to_expression(query_functions, query_target):
    """Convert function_list to S-expression string."""
    BAD_EXPRESSION = "@BAD_EXPRESSION"
    try:
        local = {}
        exec("\n".join(query_functions), exec_globals, local)
        return local[query_target]
    except Exception:
        return BAD_EXPRESSION


def function_list_to_sexpr(func_list):
    """Convert function list to S-expression (auto-detect target)."""
    if not func_list:
        return "@BAD_EXPRESSION"
    id_now = func_list[-1].split(" = ")[0].replace("expression", "")
    func_list_copy = list(func_list)
    func_list_copy.append("expression{} = STOP(expression{})".format(id_now, id_now))
    return functions_to_expression(func_list_copy, "expression" + id_now)
