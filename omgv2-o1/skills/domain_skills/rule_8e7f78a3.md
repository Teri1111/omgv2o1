---
name: "CVT path reconstruction from SearchGraphPatterns output"
tags: ["CVT path", "arrow notation", "pred1 -> pred2", "intermediate traversal"]
triggers: ["CVT path", "arrow notation", "pred1 -> pred2", "intermediate traversal"]
confidence: 0.9
success_count: 0
fail_count: 0
rule_type: "CVT_NAVIGATION"
rule_id: "rule_8e7f78a3"
version: "1.0"
source: "experience_kb"
---

# CVT path reconstruction from SearchGraphPatterns output

**Rule Type:** CVT_NAVIGATION  
**Confidence:** 0.9  
**Rule ID:** rule_8e7f78a3

## When to Apply

SearchGraphPatterns shows a path like 'pred1 -> pred2' indicating traversal through a CVT node

## Steps

1. Parse 'A -> B' in results as two predicates connected via a CVT
2. SPARQL: ?entity ns:pred1 ?cvt . ?cvt ns:pred2 ?value
3. For nested CVTs (A -> B -> C), chain three predicates through two intermediate variables
4. Verify by checking that the intermediate variable represents a plausible node type

## Common Cases

- When searchgraphpatterns shows a path like 'pred1 -> pred2' indicating traversal through a cvt node
- Similar scenarios with the same pattern

## Avoid

Treating 'pred1 -> pred2' as a single predicate or reversing the traversal order
