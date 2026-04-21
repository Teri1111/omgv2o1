---
name: "Cast string IDs to integer for numeric comparison"
tags: ["type mismatch", "empty result", "numeric filter", "string to integer", "xsd:integer", "BIND", "cast"]
triggers: ["type mismatch", "empty result", "numeric filter", "string to integer", "xsd:integer", "BIND", "cast"]
confidence: 0.95
success_count: 0
fail_count: 0
rule_type: "CONSTRAINT_PATTERN"
rule_id: "rule_f6a6d4f4"
version: "1.0"
source: "experience_kb"
---

# Cast string IDs to integer for numeric comparison

**Rule Type:** CONSTRAINT_PATTERN  
**Confidence:** 0.95  
**Rule ID:** rule_f6a6d4f4

## When to Apply

A FILTER with numeric comparison on an ID field returns empty results even though data exists, because the field is stored as a string.

## Steps

1. Run a diagnostic query without FILTER to see raw values and types
2. If values appear as strings, add explicit type cast using xsd:integer() or CAST
3. Use BIND(xsd:integer(?id_str) AS ?id_num) then FILTER(?id_num < value)
4. Execute the corrected query

## Common Cases

- When a filter with numeric comparison on an id field returns empty results even though data exists, because the field is stored as a string.
- Similar scenarios with the same pattern

## Avoid

Assuming numeric-looking fields are already typed as numbers; always verify with a diagnostic query when a FILTER returns empty.
