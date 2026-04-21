---
name: "Two-entity intersection query with shared variable"
tags: ["two constraints", "intersection", "shared variable", "multi-entity", "AND condition"]
triggers: ["two constraints", "intersection", "shared variable", "multi-entity", "AND condition"]
confidence: 0.95
success_count: 0
fail_count: 0
rule_type: "CONSTRAINT_PATTERN"
rule_id: "rule_67bd80d5"
version: "1.0"
source: "experience_kb"
---

# Two-entity intersection query with shared variable

**Rule Type:** CONSTRAINT_PATTERN  
**Confidence:** 0.95  
**Rule ID:** rule_67bd80d5

## When to Apply

Question asks for an entity that satisfies two separate conditions involving two different source entities (e.g., directed by X AND casting director Y)

## Steps

1. Use SearchGraphPatterns on the first source entity to find the predicate linking it to the answer
2. Use SearchGraphPatterns on the second source entity to find the predicate linking it to the answer
3. Construct a SPARQL query where both predicates constrain the same output variable
4. Execute the joined query to find entities satisfying both conditions

## Common Cases

- When question asks for an entity that satisfies two separate conditions involving two different source entities (e.g., directed by x and casting director y)
- Similar scenarios with the same pattern

## Avoid

Executing separate queries for each condition and trying to intersect results manually; a single joined query is more reliable
