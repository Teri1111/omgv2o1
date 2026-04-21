"""
Skill registry for OMGv2 reasoning agent.
Maps skill names to their callable implementations and metadata.
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class SkillInfo:
    """Metadata and callable for a single skill."""
    name: str
    description: str
    callable: Optional[Callable]
    category: str  # "subgraph", "lf_construction", "validation"
    # Optional dispatch metadata
    lf_op: Optional[str] = None  # e.g. "START", "JOIN" — maps to S-expression operator

    def __repr__(self):
        return f"SkillInfo({self.name}, cat={self.category})"


# ---------- Category 1: Subgraph Awareness ----------

from skills.explore_subgraph import explore_subgraph as _explore_subgraph
from skills.find_relation import find_relation as _find_relation
from skills.path_to_lf import path_to_lf_draft as _path_to_lf_draft

# ---------- Category 2: LF Construction ----------

from skills.lf_construction import (
    START as _START,
    JOIN as _JOIN,
    AND as _AND,
    ARG as _ARG,
    CMP as _CMP,
    TC as _TC,
    COUNT as _COUNT,
    STOP as _STOP,
    function_list_to_sexpr as _fl2sexpr,
)

# ---------- Category 3: Validation / Execution ----------

from skills.validate_syntax import validate_syntax as _validate_syntax
from skills.execution_feedback import (
    validate_tentative_step as _validate_tentative_step,
    evaluate_candidate_relation as _evaluate_candidate_relation,
    build_candidate_func_list as _build_candidate_func_list,
    execute_partial as _execute_partial,
    execute_final as _execute_final,
)


# ============================================================
# Registry
# ============================================================

SKILLS: Dict[str, SkillInfo] = {}

def _register(skill: SkillInfo):
    SKILLS[skill.name] = skill


# --- Category 1 ---

_register(SkillInfo(
    name="explore_subgraph",
    description="BFS from an entity in the restricted subgraph. Returns edges reachable within max_hops.",
    callable=_explore_subgraph,
    category="subgraph",
))

_register(SkillInfo(
    name="find_relation",
    description="Find direct relations between two entities in the subgraph.",
    callable=_find_relation,
    category="subgraph",
))

_register(SkillInfo(
    name="path_to_lf_draft",
    description="Generate a complete LF draft from the top-1 T5 candidate path. Returns function_list + sexpr or None.",
    callable=_path_to_lf_draft,
    category="subgraph",
    lf_op="PATH_DRAFT",
))

# --- Category 2: 8 atomic LF construction tools ---

_register(SkillInfo(
    name="extract_entity",
    description="Initialize expression with a starting entity. LF: START(entity)",
    callable=_START,
    category="lf_construction",
    lf_op="START",
))

_register(SkillInfo(
    name="find_relation_lf",
    description="Follow a one-hop relation. LF: JOIN(relation, expr) or JOIN((R relation), expr)",
    callable=_JOIN,
    category="lf_construction",
    lf_op="JOIN",
))

_register(SkillInfo(
    name="merge",
    description="AND-merge two sub-expressions. LF: AND(expr1, expr2)",
    callable=_AND,
    category="lf_construction",
    lf_op="AND",
))

_register(SkillInfo(
    name="order",
    description="ARGMAX/ARGMIN on an expression with a relation. LF: ARG(op, expr, rel)",
    callable=_ARG,
    category="lf_construction",
    lf_op="ARG",
))

_register(SkillInfo(
    name="compare",
    description="Numeric comparison (le/lt/ge/gt). LF: CMP(op, rel, expr)",
    callable=_CMP,
    category="lf_construction",
    lf_op="CMP",
))

_register(SkillInfo(
    name="time_constraint",
    description="Temporal filter on an expression. LF: TC(expr, rel, ent)",
    callable=_TC,
    category="lf_construction",
    lf_op="TC",
))

_register(SkillInfo(
    name="count",
    description="Count distinct results. LF: COUNT(expr)",
    callable=_COUNT,
    category="lf_construction",
    lf_op="COUNT",
))

_register(SkillInfo(
    name="finish",
    description="Stop and output the expression. LF: STOP(expr)",
    callable=_STOP,
    category="lf_construction",
    lf_op="STOP",
))

# --- Category 3: Validation / Execution ---

_register(SkillInfo(
    name="validate_syntax",
    description="Lightweight S-expression syntax check (parentheses, operators). No KB query.",
    callable=_validate_syntax,
    category="validation",
))

_register(SkillInfo(
    name="validate_tentative_step",
    description="Validate a tentative step: syntax + partial SPARQL execution. Returns valid/exec_ok/num_answers.",
    callable=_validate_tentative_step,
    category="validation",
))

_register(SkillInfo(
    name="evaluate_candidate_relation",
    description="Try both LF orientations for a traversal relation, pick the best by overlap/validity.",
    callable=_evaluate_candidate_relation,
    category="validation",
))

_register(SkillInfo(
    name="build_candidate_func_list",
    description="Clone current func_list and append one tentative JOIN step.",
    callable=_build_candidate_func_list,
    category="validation",
))

_register(SkillInfo(
    name="execute_partial",
    description="Single-step execution validation: syntax + partial SPARQL exec. Returns valid/exec_ok/num_answers/execution_score.",
    callable=_execute_partial,
    category="validation",
))

_register(SkillInfo(
    name="execute_final",
    description="Complete LF execution: convert function_list to SPARQL, execute, return answers.",
    callable=_execute_final,
    category="validation",
))


# ============================================================
# Query helpers
# ============================================================

def get_skill(name: str) -> SkillInfo:
    """Get a skill by name. Raises KeyError if not found."""
    return SKILLS[name]


def get_skills_by_category(category: str) -> Dict[str, SkillInfo]:
    """Get all skills in a category."""
    return {k: v for k, v in SKILLS.items() if v.category == category}


def list_skills() -> List[str]:
    """List all registered skill names."""
    return list(SKILLS.keys())


def get_available_skills(current_step: int, has_multiple_expressions: bool = False) -> List[str]:
    """Get available skill names for the current reasoning step.
    
    Implements KBQA-o1 dispatch rules:
    - Step 1: only extract_entity
    - find_relation_lf: not on step 1
    - merge: requires multiple expressions
    - order/compare/time_constraint/count: require at least one expression
    - finish: require at least one expression
    """
    available = ["extract_entity"]  # always available
    
    if current_step > 1:
        available.append("find_relation_lf")
        available.append("finish")
    
    if has_multiple_expressions:
        available.append("merge")
    
    if current_step > 1:
        available.extend(["order", "compare", "time_constraint", "count"])
    
    return available


# Convenience: category constants
CAT_SUBGRAPH = "subgraph"
CAT_LF = "lf_construction"
CAT_VALIDATION = "validation"


# ---------- Category 4: Guidance / Experience ----------

from skills.experience_kb_skill import search_experience_rules as _search_exp

_register(SkillInfo(
    name="experience_kb_search",
    description="Retrieve relevant error-correction rules from the Experience KB for LLM prompt guidance.",
    callable=_search_exp,
    category="guidance",
))

CAT_GUIDANCE = "guidance"


# ============================================================
# Tool Registry — JSON Schema based, for LLM function calling
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional


@dataclass
class ToolInfo:
    """Metadata + callable for a single LLM-callable Tool."""
    name: str
    description: str
    schema: Dict[str, Any]        # OpenAI function calling JSON Schema
    callable: Optional[Callable] = None
    category: str = "tool"

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {"type": "function", "function": self.schema}


class ToolRegistry:
    """Registry for LLM-callable Tools with JSON Schema support.

    Coexists with the legacy SKILLS dict. Tools are a subset of skills
    that have proper JSON Schema definitions for function calling.
    """

    def __init__(self):
        self._tools: Dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo):
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolInfo:
        return self._tools[name]

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Return all tool schemas in OpenAI function calling format."""
        return [t.to_openai_tool() for t in self._tools.values()]

    def get_schema_map(self) -> Dict[str, Dict[str, Any]]:
        """Return {tool_name: schema_dict}."""
        return {name: t.schema for name, t in self._tools.items()}


# Singleton
TOOLS = ToolRegistry()

# Import schemas
from skills.tools import (
    EXPLORE_NEIGHBORS_SCHEMA,
    EXTEND_EXPRESSION_SCHEMA,
    VERIFY_EXPRESSION_SCHEMA,
    CONSULT_EXPERIENCE_SCHEMA,
    INSPECT_PATH_SCHEMA,
)
from skills.tools.extend_expression_tool import extend_expression as _extend_expression

# Import adapters for tools that need parameter bridging
from skills.tools.adapters import (
    explore_neighbors_adapter,
    verify_expression_adapter,
    consult_experience_adapter,
    inspect_path_adapter,
)

# Register 5 tools — now using adapters where schema params don't match raw callables
TOOLS.register(ToolInfo(
    name="explore_neighbors",
    description=EXPLORE_NEIGHBORS_SCHEMA["description"],
    schema=EXPLORE_NEIGHBORS_SCHEMA,
    callable=explore_neighbors_adapter,
    category="tool",
))

TOOLS.register(ToolInfo(
    name="extend_expression",
    description=EXTEND_EXPRESSION_SCHEMA["description"],
    schema=EXTEND_EXPRESSION_SCHEMA,
    callable=_extend_expression,
    category="tool",
))

TOOLS.register(ToolInfo(
    name="verify_expression",
    description=VERIFY_EXPRESSION_SCHEMA["description"],
    schema=VERIFY_EXPRESSION_SCHEMA,
    callable=verify_expression_adapter,
    category="tool",
))

TOOLS.register(ToolInfo(
    name="consult_experience",
    description=CONSULT_EXPERIENCE_SCHEMA["description"],
    schema=CONSULT_EXPERIENCE_SCHEMA,
    callable=consult_experience_adapter,
    category="tool",
))

TOOLS.register(ToolInfo(
    name="inspect_path",
    description=INSPECT_PATH_SCHEMA["description"],
    schema=INSPECT_PATH_SCHEMA,
    callable=inspect_path_adapter,
    category="tool",
))
