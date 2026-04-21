"""Find relation skill for OMGv2. Finds direct relations between two entities."""

def find_relation(subgraph, entity1, entity2):
    if entity1 not in subgraph or entity2 not in subgraph:
        return []
    results = []
    for rel, tgt in subgraph.get_outgoing(entity1):
        if tgt == entity2:
            results.append({"source": entity1, "relation": rel, "target": entity2})
    return results
