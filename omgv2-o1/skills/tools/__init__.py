"""
Tool Schema definitions for OMGv2 LLM-First reasoning.
Used by ToolRegistry and LLM function calling prompts.
"""

# Tool 1: explore_neighbors
EXPLORE_NEIGHBORS_SCHEMA = {
    "name": "explore_neighbors",
    "description": "Explore 1-hop neighbors of an entity in the T5-restricted subgraph. Returns available relations and target entities grouped by direction.",
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "Entity MID to explore from"
            },
            "direction": {
                "type": "string",
                "enum": ["outgoing", "incoming", "both"],
                "description": "Direction of exploration",
                "default": "both"
            },
            "filter_pattern": {
                "type": "string",
                "description": "Regex pattern to filter relation names"
            }
        },
        "required": ["entity"]
    }
}

# Tool 2: extend_expression
EXTEND_EXPRESSION_SCHEMA = {
    "name": "extend_expression",
    "description": "Extend the current S-expression by one step. Wraps LF primitives (JOIN, AND, ARG, CMP, TC, COUNT) and automatically validates with partial execution. Returns structured feedback.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["join", "and", "argmax", "argmin", "cmp", "tc", "count"],
                "description": "LF primitive action to perform"
            },
            "relation": {
                "type": "string",
                "description": "Freebase relation to traverse (required for join/argmax/argmin/cmp/tc)"
            },
            "direction": {
                "type": "string",
                "enum": ["forward", "reverse"],
                "description": "Direction of relation traversal",
                "default": "forward"
            },
            "expression_id": {
                "type": "string",
                "description": "Target expression ID (for join/and). Defaults to current expression."
            },
            "sub_expression_id": {
                "type": "string",
                "description": "Second expression ID (for AND merge)"
            },
            "operator": {
                "type": "string",
                "description": "Comparison operator for CMP (le/lt/ge/gt)"
            },
            "entity": {
                "type": "string",
                "description": "Target entity for TC (transitive closure)"
            },
            "target_expr": {
                "type": "string",
                "description": "Full S-expression of the target (for and)"
            }
        },
        "required": ["action"]
    }
}

# Tool 3: verify_expression
VERIFY_EXPRESSION_SCHEMA = {
    "name": "verify_expression",
    "description": "Verify the current S-expression by syntax check and execution. Supports partial (single-step) and full (complete) execution modes.",
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["partial", "full"],
                "description": "Execution mode",
                "default": "partial"
            },
            "expression": {
                "type": "string",
                "description": "S-expression to verify. If omitted, uses current expression."
            }
        },
        "required": []
    }
}

# Tool 4: consult_experience
CONSULT_EXPERIENCE_SCHEMA = {
    "name": "consult_experience",
    "description": "Retrieve relevant error-correction rules from the Experience Knowledge Base. Returns guidance rules, suggested actions, and confidence scores.",
    "parameters": {
        "type": "object",
        "properties": {
            "state_description": {
                "type": "string",
                "description": "Current reasoning state description"
            },
            "last_error": {
                "type": "string",
                "description": "Last error message encountered"
            },
            "current_expression": {
                "type": "string",
                "description": "Current S-expression"
            },
            "available_relations": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Available relations to choose from"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top rules to retrieve",
                "default": 3
            },
            "query_type": {
                "type": "string",
                "enum": ["passive", "active", "skill_md"],
                "description": "Query mode: 'passive' returns formatted string, 'active' returns structured dict, 'skill_md' returns SKILL.md documents",
                "default": "passive"
            }
        },
        "required": []
    }
}

# Tool 5: inspect_path
INSPECT_PATH_SCHEMA = {
    "name": "inspect_path",
    "description": "Inspect a T5 candidate path to preview its S-expression and execution result. Use before committing to a path-guided answer.",
    "parameters": {
        "type": "object",
        "properties": {
            "path_index": {
                "type": "integer",
                "description": "Index of the candidate path to inspect (0 = top-1)",
                "default": 0
            }
        },
        "required": []
    }
}

# List for easy injection into LLM prompts
ALL_TOOL_SCHEMAS = [
    EXPLORE_NEIGHBORS_SCHEMA,
    EXTEND_EXPRESSION_SCHEMA,
    VERIFY_EXPRESSION_SCHEMA,
    CONSULT_EXPERIENCE_SCHEMA,
    INSPECT_PATH_SCHEMA,
]

# Dict for quick lookup
TOOL_SCHEMA_MAP = {t["name"]: t for t in ALL_TOOL_SCHEMAS}
