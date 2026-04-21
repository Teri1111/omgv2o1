"""Explore subgraph skill for OMGv2. BFS from starting entity."""

def explore_subgraph(subgraph, entity_id, max_hops=2):
    if entity_id not in subgraph:
        return []
    visited = set()
    frontier = {entity_id}
    results = []
    for hop in range(1, max_hops + 1):
        next_frontier = set()
        for ent in frontier:
            if ent in visited:
                continue
            visited.add(ent)
            for rel, tgt in subgraph.get_outgoing(ent):
                results.append({"source": ent, "relation": rel, "target": tgt, "hop": hop})
                if tgt not in visited:
                    next_frontier.add(tgt)
            for rel, src in subgraph.get_incoming(ent):
                results.append({"source": src, "relation": "(R " + rel + ")", "target": ent, "hop": hop})
                if src not in visited:
                    next_frontier.add(src)
        frontier = next_frontier
        if not frontier:
            break
    return results
