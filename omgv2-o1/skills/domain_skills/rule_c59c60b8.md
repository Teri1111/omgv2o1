---
name: "Use bidirectional predicates for writer-film connections"
tags: ["story by", "written by", "film writer", "story credit", "film.film.story_by"]
triggers: ["story by", "written by", "film writer", "story credit", "film.film.story_by"]
confidence: 0.9
success_count: 0
fail_count: 0
rule_type: "RELATION_DIRECTION"
rule_id: "rule_c59c60b8"
version: "1.0"
source: "experience_kb"
---

# Use bidirectional predicates for writer-film connections

**Rule Type:** RELATION_DIRECTION  
**Confidence:** 0.9  
**Rule ID:** rule_c59c60b8

## When to Apply

Question requires finding films connected to a writer/author through story_by, written_by, or similar predicates.

## Steps

1. Use SearchGraphPatterns on the writer entity with semantic 'story by/written by'
2. Identify the relevant predicate (e.g., film.film_story_contributor.film_story_credits)
3. Build SPARQL connecting writer to film through the identified predicate
4. Add additional constraints (actor in same film) and execute

## Common Cases

- When question requires finding films connected to a writer/author through story_by, written_by, or similar predicates.
- Similar scenarios with the same pattern

## Avoid

Only searching in one direction; check both film-to-writer and writer-to-film predicates.
