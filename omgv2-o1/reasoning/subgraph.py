"""
Subgraph builder for OMGv2.
Builds restricted subgraph G_restricted from T5 retrieval paths/triplets.

Internal storage: entity -> [(relation, [target_entities])]
get_outgoing() returns flat [(rel, target)] for each target.
"""

from typing import Dict, List, Tuple, Set, Optional


class SubgraphBuilder:

    def __init__(self):
        self._outgoing: Dict[str, List[Tuple[str, List[str]]]] = {}
        self._incoming: Dict[str, List[Tuple[str, List[str]]]] = {}
        self._all_entities: Set[str] = set()
        self._all_relations: Set[str] = set()
        self._trace: Optional[dict] = None

    def set_trace(self, trace: Optional[dict]):
        """Attach a trace dict for recording build/edge events."""
        self._trace = trace

    def _clear(self):
        self._outgoing.clear()
        self._incoming.clear()
        self._all_entities.clear()
        self._all_relations.clear()

    def _add_edge(self, src, rel, tgt):
        """Add one directed edge src --rel--> tgt. Merge targets under same relation."""
        if src not in self._outgoing:
            self._outgoing[src] = []
        found = False
        for er, et_list in self._outgoing[src]:
            if er == rel:
                if tgt not in et_list:
                    et_list.append(tgt)
                found = True
                break
        if not found:
            self._outgoing[src].append((rel, [tgt]))

        if tgt not in self._incoming:
            self._incoming[tgt] = []
        found = False
        for er, es_list in self._incoming[tgt]:
            if er == rel:
                if src not in es_list:
                    es_list.append(src)
                found = True
                break
        if not found:
            self._incoming[tgt].append((rel, [src]))

        self._all_entities.add(src)
        self._all_entities.add(tgt)
        self._all_relations.add(rel)

        if self._trace is not None:
            if "added_edges" not in self._trace:
                self._trace["added_edges"] = []
            self._trace["added_edges"].append({"src": src, "rel": rel, "tgt": tgt})

    def build(self, paths):
        """Build from path dicts: [{"path": [e1, r1, e2, r2, e3], "score": float}]"""
        self._clear()
        if self._trace is not None:
            self._trace["build_input_type"] = "paths"
            self._trace["build_input_count"] = len(paths)
        for path_dict in paths:
            if not isinstance(path_dict, dict):
                continue
            path = path_dict.get("path", [])
            if len(path) < 3:
                continue
            i = 0
            while i + 2 < len(path):
                src, rel, tgt = path[i], path[i + 1], path[i + 2]
                if rel.startswith("(R ") and rel.endswith(")"):
                    actual_rel = rel[3:-1].strip()
                    self._add_edge(tgt, actual_rel, src)
                else:
                    self._add_edge(src, rel, tgt)
                i += 2

    def build_from_triplets(self, triplets_by_hop):
        """Build from CWQ triplets: {"0": [["m.src", "rel", ["m.tgt1", ...]], ...]}"""
        self._clear()
        if self._trace is not None:
            self._trace["build_input_type"] = "triplets"
            total = sum(len(v) for v in triplets_by_hop.values())
            self._trace["build_input_count"] = total
        for hop_key, triplets in triplets_by_hop.items():
            for triplet in triplets:
                if len(triplet) < 3:
                    continue
                src, rel, tgts = triplet[0], triplet[1], triplet[2]
                if not isinstance(tgts, list):
                    tgts = [tgts]
                for tgt in tgts:
                    self._add_edge(src, rel, tgt)

    def get_outgoing(self, entity_id):
        """Return flat [(rel, target)] for each target."""
        result = []
        for rel, tgts in self._outgoing.get(entity_id, []):
            for t in tgts:
                result.append((rel, t))
        return result

    def get_incoming(self, entity_id):
        """Return flat [(rel, source)] for each source."""
        result = []
        for rel, srcs in self._incoming.get(entity_id, []):
            for s in srcs:
                result.append((rel, s))
        return result

    def get_outgoing_relations(self, entity_id):
        return list({rel for rel, _ in self.get_outgoing(entity_id)})

    def get_incoming_relations(self, entity_id):
        return list({rel for rel, _ in self.get_incoming(entity_id)})

    def get_targets(self, entity_id, relation):
        return [t for rel, t in self.get_outgoing(entity_id) if rel == relation]

    def has_relation(self, entity_id, relation):
        return any(rel == relation for rel, _ in self.get_outgoing(entity_id))

    def get_all_entities(self):
        return list(self._all_entities)

    def get_all_relations(self):
        return list(self._all_relations)

    def __len__(self):
        return len(self._all_entities)

    def __contains__(self, entity_id):
        return entity_id in self._all_entities

    def __repr__(self):
        n = 0
        for entries in self._outgoing.values():
            for _, tgts in entries:
                n += len(tgts)
        return "SubgraphBuilder(entities=%d, relations=%d, edges=%d)" % (
            len(self._all_entities), len(self._all_relations), n)

    def snapshot(self):
        """Return a full serializable snapshot of the subgraph state."""
        outgoing = {}
        for ent, entries in self._outgoing.items():
            outgoing[ent] = [(rel, list(tgts)) for rel, tgts in entries]
        incoming = {}
        for ent, entries in self._incoming.items():
            incoming[ent] = [(rel, list(srcs)) for rel, srcs in entries]
        return {
            "outgoing": outgoing,
            "incoming": incoming,
            "entities": sorted(self._all_entities),
            "relations": sorted(self._all_relations),
        }
