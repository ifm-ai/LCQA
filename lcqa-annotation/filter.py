"""
Stage 2: annotated JSONL -> filtered JSONL.

Keep a document only if every threshold in config.json's `document_filters`
holds. The thresholds are bucket numbers, so "avg_fineweb_edu_scores_bucket >= 17"
reads as "its sections average into the 17th of 20 quality bands for educational
value" — and the same 17 would mean the same standing for any of the three
classifiers.

    python filter.py annotated.jsonl kept.jsonl
    python filter.py annotated.jsonl kept.jsonl rejected.jsonl   # keep the rejects too

Most documents do not survive, and that is the point: the corpus is large and
the questions we synthesise later are only as good as the passages behind them.
"""

import json
import sys


def keeps(document_meta, filters):
    """True if the document satisfies every threshold."""
    for field, bounds in filters.items():
        value = document_meta.get(field)
        if value is None:
            return False
        if "min" in bounds and value < bounds["min"]:
            return False
        if "max" in bounds and value > bounds["max"]:
            return False
    return True


def main(input_jsonl, output_jsonl, rejected_jsonl=None, config_path="config.json"):
    filters = json.load(open(config_path))["document_filters"]
    kept = total = 0
    rejects = open(rejected_jsonl, "w") if rejected_jsonl else None

    with open(input_jsonl) as fin, open(output_jsonl, "w") as fout:
        for line in fin:
            record = json.loads(line)
            total += 1

            # Document length comes from the input parquet, not from scoring,
            # so lift it into document_meta to be filtered alongside the rest.
            meta = record["document_meta"]
            if "jais_token_count" in record.get("prev_document_meta", {}):
                meta["jais_token_count"] = record["prev_document_meta"]["jais_token_count"]

            line_out = json.dumps(record, ensure_ascii=False) + "\n"
            if keeps(meta, filters):
                fout.write(line_out)
                kept += 1
            elif rejects:
                rejects.write(line_out)

    if rejects:
        rejects.close()
    print(f"kept {kept}/{total} documents -> {output_jsonl}")
    if rejected_jsonl:
        print(f"rejected {total - kept}/{total} -> {rejected_jsonl}")


if __name__ == "__main__":
    main(*sys.argv[1:])
