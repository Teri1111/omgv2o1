"""Merge all rule batches, deduplicate, clear KB, reload."""
import json, os, sys, uuid
from collections import Counter

sys.path.insert(0, "/data/gt/experience_kb")
from modules.knowledge_base import ExperienceKB

# ── Merge all batches ───────────────────────────────────────────────────────
all_rules = []
for i in range(1, 11):
    path = "/tmp/rules_batch_%02d.json" % i
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        print("Batch %02d: %d rules" % (i, len(rules)))
        all_rules.extend(rules)

print("\nTotal raw: %d" % len(all_rules))

# ── Deduplicate by normalized title ─────────────────────────────────────────
seen = {}
unique = []
for r in all_rules:
    key = r.get("title", "").strip().lower()
    if key and key not in seen:
        seen[key] = True
        # Enrich
        if "rule_id" not in r:
            r["rule_id"] = "rule_" + uuid.uuid4().hex[:8]
        if "embedding_text" not in r:
            parts = [r.get("title",""), r.get("state_description",""),
                     r.get("action",{}).get("description",""),
                     " ".join(r.get("state_keywords",[]))]
            r["embedding_text"] = " ".join(p for p in parts if p).strip()
        unique.append(r)

print("Unique: %d" % len(unique))

# Type distribution
tc = Counter(r.get("rule_type","?") for r in unique)
print("\nTypes:")
for t, c in tc.most_common():
    print("  %20s: %d" % (t, c))

# ── Save merged JSONL ──────────────────────────────────────────────────────
merged_path = "/data/gt/experience_kb/data/extracted_rules/all_rules_merged.jsonl"
os.makedirs(os.path.dirname(merged_path), exist_ok=True)
with open(merged_path, "w", encoding="utf-8") as f:
    for r in unique:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("\nSaved to %s" % merged_path)

# ── Reset KB and reload ────────────────────────────────────────────────────
kb_dir = "/data/gt/experience_kb/data/experience_kb"
# Remove old KB files to start fresh
for fn in ["rules.json", "embeddings.npy", "faiss_index.bin", "metadata.json"]:
    fp = os.path.join(kb_dir, fn)
    if os.path.exists(fp):
        os.remove(fp)
        print("Removed old %s" % fn)

# Create fresh KB
kb = ExperienceKB(kb_dir=kb_dir)

# Validate and add
valid = []
invalid = 0
for r in unique:
    if kb._validate_rule(r):
        valid.append(r)
    else:
        invalid += 1
        print("  INVALID: %s" % r.get("title", "?"))

print("\nValid: %d, Invalid: %d" % (len(valid), invalid))

# Batch add
added = kb.add_rules_batch(valid)
kb.save()
print("Added %d rules to KB. Total: %d" % (len(added), len(kb.rules)))
print("\nDone!")
