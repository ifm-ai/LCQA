# lcqa-annotation

Finds passages worth asking questions about, inside documents long enough to be
worth searching.

This is the first half of LCQA. It takes a corpus of long documents and produces
**snippet/document pairs**: a short high-quality passage, paired with the entire
document it came from. `lcqa-synthesis` then writes a question about the snippet
and trains on the whole document — so answering requires finding the passage in a
long context, not reading a short one.

## The idea

A 100k-token document is rarely uniformly good or uniformly junk. Scoring it as a
single unit throws away exactly the information we need, which is *where* the good
parts are. So:

```
   document
      │  split.py     cut into ~512-token sections
      ▼
   sections
      │  score.py     score every section: 3 quality classifiers + toxicity,
      ▼               each quality score mapped to a 0-19 corpus rank bucket
   scored sections
      │  annotate.py  roll section scores up to the document (avg / max / min)
      ▼
   annotated document
      │  filter.py    drop the document unless its aggregates clear the thresholds
      ▼
   surviving document                      (about half are dropped here)
      │  snippets.py  keep the best sections, pair each with the full document
      ▼
   {"snippet": ..., "doc": ...}
```

The buckets are what make a single threshold meaningful. The three quality
classifiers return incomparable numbers — two probabilities and a roughly 0–5
regression — so each raw score is mapped onto a shared 0–19 scale using cut
points taken from the corpus score distribution. A bucket is a rank rather than a
score: bucket 17 means the same standing in the corpus whichever classifier
produced it, so `>= 17` is one honest rule rather than three arbitrary ones.

## The files

| File | What it does |
|---|---|
| `config.json` | Every threshold and setting, in one place. |
| `split.py` | Document → sections. Six strategies; prose, markdown and code split differently. |
| `score.py` | Sections → quality and toxicity scores, plus the bucket mapping. |
| `annotate.py` | Stage 1 — input parquet → annotated JSONL. |
| `filter.py` | Stage 2 — split into documents that clear the thresholds and those that don't. |
| `snippets.py` | Stage 3 — surviving documents → snippet/doc pairs. |
| `buckets/` | The bucket cut points, taken from the corpus score distribution — see `buckets/README.md`. |
| `examples/` | Real data at every stage. Start here. |

Under 450 lines in total. This is written to be read, not deployed: it runs one
document at a time on one machine.

Two things the production pipeline had are not here, and both are plumbing rather
than idea. It ran on SLURM across many GPUs, with shard-and-merge for oversized
files and resume caches — none of which change a single score. And between
filtering and snippet sampling it interleaved the surviving documents from all
sources into shuffled training-order chunks; since every document is processed
independently, that changes the order records land in, never their contents. The
step that groups documents by length runs before this repo, as described under
Input.

## Running it

```bash
cd lcqa-annotation
pip install -r requirements.txt
export LCQA_MODELS_DIR=/path/to/models        # see below

python annotate.py examples/traced/1-input.parquet annotated.jsonl
python filter.py   annotated.jsonl  kept.jsonl  rejected.jsonl
python snippets.py kept.jsonl       pairs.jsonl
```

## Models

The four classifiers are not in this repo. All are public:

| Model | Source | Size |
|---|---|---|
| DCLM / ELI5 fastText | [`mlfoundations/fasttext-oh-eli5`](https://huggingface.co/mlfoundations/fasttext-oh-eli5) | 2.3 GB |
| PreSelect fastText | [`hkust-nlp/preselect-fasttext-classifier`](https://huggingface.co/hkust-nlp/preselect-fasttext-classifier) | 2.6 GB |
| FineWeb-Edu | [`HuggingFaceFW/fineweb-edu-classifier`](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) | 419 MB |
| Detoxify `multilingual` | fetched automatically on first use | ~1.5 GB |

Download the first three into one directory:

```bash
pip install huggingface_hub
export LCQA_MODELS_DIR=$PWD/models

huggingface-cli download mlfoundations/fasttext-oh-eli5 \
    openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin --local-dir "$LCQA_MODELS_DIR"
huggingface-cli download hkust-nlp/preselect-fasttext-classifier \
    PreSelect-classifier.bin --local-dir "$LCQA_MODELS_DIR"
huggingface-cli download HuggingFaceFW/fineweb-edu-classifier \
    --local-dir "$LCQA_MODELS_DIR/fineweb-edu-classifier"
```

That leaves the layout `score.py` expects:

```
models/
├── openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin
├── PreSelect-classifier.bin
└── fineweb-edu-classifier/
```

`LCQA_MODELS_DIR` defaults to `./models`, so a symlink or a directory of that
name next to the scripts also works without setting anything.

## Input

Parquet, one row per document, with a `text` column and a `subset` column naming
the source corpus. The other columns are carried through untouched into
`prev_document_meta` so nothing upstream is lost.

Documents are expected to be pre-grouped by length — the corpus this was built for
is partitioned into `8k-32k`, `32k-128k` and `128k-512k` token buckets, because
how a document should be split and how much is worth keeping both depend on how
long it is. That bucketing is upstream of this repo; `config.json`'s
`jais_token_count` threshold is the only place length enters here.

## What comes out

One JSON object per line. `document_meta` describes the whole document,
`sections_meta` has one value per section:

```jsonc
{
  "document": "<full original text>",
  "sections": ["<~512 tokens>", "..."],
  "document_meta": {
    "avg_eli5_scores": 0.62, "max_...": ..., "min_...": ...,
    "avg_eli5_scores_bucket": 14,
    "avg_toxicity_scores": 0.002
  },
  "sections_meta": {
    "token_count": [511, 498, ...],
    "eli5_scores": [...], "eli5_scores_bucket": [...],
    "toxicity_scores": [...]
  },
  "prev_document_meta": { "<every input column>": ... }
}
```

Note `avg_eli5_scores_bucket` is the bucket *of the average score*, not the
average of the section buckets — averaging rank numbers would be meaningless.

## License

Code: Apache-2.0, see `../LICENSE`. Example data: public-domain sources only, see
`examples/README.md`.
