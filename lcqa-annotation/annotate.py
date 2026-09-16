"""
Stage 1: input parquet -> annotated JSONL.

Split each document into sections, score every section, then roll the section
scores back up to the document as avg / max / min. The document-level bucket is
the bucket of the *average score*, not the average of the section buckets.

    python annotate.py examples/traced/1-input.parquet out.jsonl

Each output line:

    {
      "document":     "<the full original text>",
      "sections":     ["<section 1>", ...],
      "document_meta":  {"avg_eli5_scores": .., "avg_eli5_scores_bucket": .., ...},
      "sections_meta":  {"token_count": [..], "eli5_scores": [..], ...},
      "prev_document_meta": {<every column the input parquet carried>}
    }
"""

import json
import sys

import pandas as pd

from split import split_document, count_tokens
from score import score_sections, to_bucket

QUALITY = ("eli5_scores", "preselect_scores", "fineweb_edu_scores")


def annotate_document(text, config, subset=None, device="cuda:0"):
    """Split and score one document."""
    splitter = config["splitter_by_subset"].get(subset, config["default_splitter"])
    max_tokens = config["section_max_tokens"]
    model = config["tokenizer_model"]

    sections = split_document(text, max_tokens, splitter, model)
    if not sections:
        return {"document": text, "sections": [], "document_meta": {}, "sections_meta": {}}

    scores = score_sections(sections, device=device)

    # Per section: how long it is, plus every score and bucket.
    sections_meta = {"token_count": [count_tokens(s, model, max_tokens) for s in sections]}
    sections_meta.update(scores)

    # Per document: aggregate each score across sections.
    document_meta = {}
    for name, values in scores.items():
        if name.endswith("_bucket"):
            continue                      # buckets are per section only
        document_meta[f"avg_{name}"] = sum(values) / len(values)
        document_meta[f"max_{name}"] = max(values)
        document_meta[f"min_{name}"] = min(values)

    # ...and bucket the aggregated quality scores against the corpus cut points.
    for name in QUALITY:
        for how in ("avg", "max", "min"):
            key = f"{how}_{name}"
            document_meta[f"{key}_bucket"] = to_bucket(document_meta[key], name[: -len("_scores")])

    return {"document": text, "sections": sections,
            "document_meta": document_meta, "sections_meta": sections_meta}


def main(input_parquet, output_jsonl, config_path="config.json", device="cuda:0"):
    config = json.load(open(config_path))
    rows = pd.read_parquet(input_parquet)
    columns = [c for c in rows.columns if c != "text"]

    with open(output_jsonl, "w") as out:
        for i, row in rows.iterrows():
            record = annotate_document(row["text"], config, row.get("subset"), device)
            record["prev_document_meta"] = {c: _plain(row[c]) for c in columns}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  [{i + 1}/{len(rows)}] {len(record['sections'])} sections")

    print(f"wrote {output_jsonl}")


def _plain(value):
    """numpy scalar -> python scalar, so json can serialise it."""
    if hasattr(value, "item"):
        value = value.item()
    return None if isinstance(value, float) and value != value else value


if __name__ == "__main__":
    main(*sys.argv[1:])
