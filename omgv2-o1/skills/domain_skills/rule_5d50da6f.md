---
name: "Navigate CVT (Compound Value Type) nodes for intermediate data"
tags: ["CVT", "intermediate node", "compound value", "two-hop", "roster", "marriage", "education"]
triggers: ["CVT", "intermediate node", "compound value", "two-hop", "roster", "marriage", "education"]
confidence: 0.92
success_count: 0
fail_count: 0
rule_type: "MULTI_HOP_PATTERN"
rule_id: "rule_5d50da6f"
version: "1.0"
source: "experience_kb"
---

# Navigate CVT (Compound Value Type) nodes for intermediate data

**Rule Type:** MULTI_HOP_PATTERN  
**Confidence:** 0.92  
**Rule ID:** rule_5d50da6f

## When to Apply

The relationship between two entities passes through an intermediate CVT node that holds additional properties like dates or roles.

## Steps

1. Identify the CVT pattern from SearchGraphPatterns results (shown as pred1 -> pred2)
2. In SPARQL, bind the first predicate to a CVT variable: ?entity ns:pred1 ?cvt
3. Then traverse from the CVT: ?cvt ns:pred2 ?target
4. Add further constraints or ordering on properties attached to the CVT node

## Common Cases

- When the relationship between two entities passes through an intermediate cvt node that holds additional properties like dates or roles.
- Similar scenarios with the same pattern

## Avoid

Treating CVT chains as direct single-hop predicates or ignoring the intermediate node.
