---
name: "Decompose complex questions into sub-problem triples"
tags: ["multi-hop", "complex question", "sub-problem", "decomposition", "intersection"]
triggers: ["multi-hop", "complex question", "sub-problem", "decomposition", "intersection"]
confidence: 0.95
success_count: 0
fail_count: 0
rule_type: "MULTI_HOP_PATTERN"
rule_id: "rule_96bb0c91"
version: "1.0"
source: "experience_kb"
---

# Decompose complex questions into sub-problem triples

**Rule Type:** MULTI_HOP_PATTERN  
**Confidence:** 0.95  
**Rule ID:** rule_96bb0c91

## When to Apply

A complex question requires multiple hops or constraints that cannot be answered with a single predicate lookup.

## Steps

1. Identify all entities and relationships implied by the question
2. Map each sub-problem to a (subject, predicate, object) triple
3. Use SearchGraphPatterns to discover predicates for each sub-triple independently
4. Combine discovered predicates into a single ExecuteSPARQL query joining on shared variables

## Common Cases

- When a complex question requires multiple hops or constraints that cannot be answered with a single predicate lookup.
- Similar scenarios with the same pattern

## Avoid

Attempting to answer the full question in one SearchGraphPatterns call without decomposition.
