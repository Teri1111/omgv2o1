---
name: "CVT chain traversal for multi-hop relationships"
tags: ["CVT", "multi_hop", "intermediate_node", "performance", "roster", "education"]
triggers: ["CVT", "multi_hop", "intermediate_node", "performance", "roster", "education"]
confidence: 0.95
success_count: 0
fail_count: 0
rule_type: "MULTI_HOP_PATTERN"
rule_id: "rule_68d034af"
version: "1.0"
source: "experience_kb"
---

# CVT chain traversal for multi-hop relationships

**Rule Type:** MULTI_HOP_PATTERN  
**Confidence:** 0.95  
**Rule ID:** rule_68d034af

## When to Apply

Need to traverse a 2-hop relationship where Freebase uses a Compound Value Type (CVT) intermediate node.

## Steps

1. Identify the CVT pattern from SearchGraphPatterns results (arrow notation)
2. Extract predicate1 (before arrow) and predicate2 (after arrow)
3. Construct two-triple chain through ?cvt variable
4. Use the chained pattern in ExecuteSPARQL

## Common Cases

- When need to traverse a 2-hop relationship where freebase uses a compound value type (cvt) intermediate node.
- Similar scenarios with the same pattern

## Avoid

Treating the CVT as a direct relationship or trying to skip the intermediate node.
