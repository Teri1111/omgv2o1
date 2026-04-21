---
name: "Consider inverse relationships in SearchGraphPatterns"
tags: ["inverse", "reverse", "bidirectional", "opposite direction"]
triggers: ["inverse", "reverse", "bidirectional", "opposite direction"]
confidence: 0.85
success_count: 0
fail_count: 0
rule_type: "RELATION_DIRECTION"
rule_id: "rule_adb3f84a"
version: "1.0"
source: "experience_kb"
---

# Consider inverse relationships in SearchGraphPatterns

**Rule Type:** RELATION_DIRECTION  
**Confidence:** 0.85  
**Rule ID:** rule_adb3f84a

## When to Apply

SearchGraphPatterns shows the relationship in the reverse direction

## Steps

1. Check if predicate links subject->object or object->subject
2. Adjust SPARQL to match the actual direction in the knowledge base
3. Example: ?country ns:location.location.containedby ?region for 'countries in region'

## Common Cases

- When searchgraphpatterns shows the relationship in the reverse direction
- Similar scenarios with the same pattern

## Avoid

Assuming relationship direction without checking SearchGraphPatterns results
