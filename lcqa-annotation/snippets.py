"""
Stage 3: filtered JSONL -> snippet/document pairs.

This is the handoff to lcqa-synthesis, and the point of the whole pipeline.

Within a surviving document, take only the sections that pass `snippet_filters`
(the right length, high quality, low toxicity) and sample a few of them. Each
sampled section is paired with the **entire** document it came from:

    {"snippet": "<one ~512-token section>", "doc": "<the whole document>"}

A question is then written from the snippet, but the training example shows the
whole document — so answering it means locating that passage in a long context
rather than reading a short one. The document repeats in every pair, which is
why these files are large.

    python snippets.py kept.jsonl pairs.jsonl
"""

import json
import random
import sys


def passes(sections_meta, index, filters):
    """True if the section at `index` satisfies every threshold."""
    for field, bounds in filters.items():
        values = sections_meta.get(field)
        if values is None or index >= len(values):
            continue
        if "min" in bounds and values[index] < bounds["min"]:
            return False
        if "max" in bounds and values[index] > bounds["max"]:
            return False
    return True


def document_to_pairs(record, config):
    """One document -> a list of {snippet, doc} pairs, in document order."""
    sections = record.get("sections", [])
    document = record.get("document", "")
    if not sections or not document:
        return []

    eligible = [i for i in range(len(sections))
                if passes(record.get("sections_meta", {}), i, config["snippet_filters"])]
    if len(eligible) < config["min_snippets_per_doc"]:
        return []

    rng = random.Random(config["random_seed"])
    chosen = rng.sample(eligible, min(config["snippets_per_doc"], len(eligible)))
    return [{"snippet": sections[i], "doc": document} for i in sorted(chosen)]


def main(input_jsonl, output_jsonl, config_path="config.json"):
    config = json.load(open(config_path))
    documents = pairs = 0

    with open(input_jsonl) as fin, open(output_jsonl, "w") as fout:
        for line in fin:
            result = document_to_pairs(json.loads(line), config)
            if result:
                documents += 1
                pairs += len(result)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"{documents} documents -> {pairs} snippet/doc pairs -> {output_jsonl}")


if __name__ == "__main__":
    main(*sys.argv[1:])
