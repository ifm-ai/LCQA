# Example data

Real data from a real run. Two sets, answering different questions.

## `traced/` — ten documents, followed through every stage

The same ten documents at each step, so you can pick one in `1-input.parquet` and
watch what happens to it:

| File | Contents |
|---|---|
| `1-input.parquet` | 10 documents — pg19 ×3, uspto ×3, europarl ×2, freelaw ×2 |
| `2-annotated.jsonl` | the same 10, split into sections and scored |
| `3-kept.jsonl` | the 5 that clear every threshold |
| `3-rejected.jsonl` | the 5 that do not |
| `4-snippets.jsonl` | 20 snippet/doc pairs, from the kept documents only |

The split is the interesting part, so here is why each rejected document was
dropped:

| subset | length | missed |
|---|---|---|
| uspto | 8k–32k | `jais_token_count` 9,359 < 16,384 — too short, quality was fine |
| europarl | 8k–32k | `fineweb_edu` bucket 3 < 17, `preselect` bucket 6 < 10 |
| freelaw | 8k–32k | `fineweb_edu` bucket 5 < 17, `preselect` 7 < 10, and too short |
| europarl | 32k–128k | `fineweb_edu` bucket 2 < 17 |
| uspto | 32k–128k | `fineweb_edu` bucket 10 < 17 |

Three different ways to fail: too short despite good scores, poor quality despite
good length, and both at once. Parliamentary proceedings and patents are perfectly
good text — they just score low on *educational value*, which is what these
thresholds ask about.

Documents are ordered by length bucket, not by outcome, so the file does not give
away which is which. Run `filter.py` and find out.

## `input-sample.parquet` — what actually goes in

43 documents drawn at random from the input corpus, so the mix is honest:

| | 8k–32k | 32k–128k | 128k–512k |
|---|---|---|---|
| freelaw | 8 | 3 | 1 |
| uspto | 8 | 3 | 1 |
| pg19 | 7 | 2 | 1 |
| europarl | 7 | 2 | — |

Fewer long documents than short ones, deliberately: a single 128k–512k document
is around 700 KB of text, and its annotated form is over 1.5 MB.

`4-snippets.jsonl` is the largest file here because every pair repeats the full
document alongside its snippet. That redundancy is the format, not an oversight:
the snippet is what a question gets written about, the document is what the model
must search.

## Where these came from

Every example is from the `High` partition of the source corpus, which is not a
random slice of the web — it holds documents whose `final_bucket` is 19, the top
band. So these documents already passed a quality bar before this pipeline saw
them, and five of ten still get rejected here.

## Provenance and licensing

All four corpora are public-domain or freely reusable sources:

| subset | what it is | basis |
|---|---|---|
| `freelaw` | US court opinions | US judicial works — public domain |
| `uspto` | US patent documents | US government works — public domain |
| `pg19` | Project Gutenberg books published before 1919 | expired copyright — public domain |
| `europarl` | European Parliament proceedings | EU documents, reusable |

Two things to be clear about. First, the selection is **by corpus, not by
document**: the pipeline data carries no per-document license field, so there is
nothing finer to filter on. Second, these are the source corpora's terms; each
document is reproduced here unmodified apart from the removal described below.

The upstream `source` column, which held an absolute path on the cluster the
corpus was built on, has been dropped from every file here. Nothing else was
altered — the text, scores and section boundaries are exactly what the pipeline
produced.
