# LCQA — Long-Context Question Answering data

Building training data that requires a model to *find* an answer in a long
document, rather than read a short one.

The trick is to write the question about a small, high-quality passage, but show
the model the entire document it came from. The answer is fully determined by the
passage, so it stays correct when the context expands — but getting to it means
searching tens of thousands of tokens.

Two halves:

| | |
|---|---|
| [`lcqa-annotation`](lcqa-annotation/) | Corpus → snippet/document pairs. Splits long documents into sections, scores every section for quality and toxicity, keeps the documents worth using, and samples the passages worth asking about. |
| `lcqa-synthesis` | Snippet/document pairs → questions and answers. |

They meet at one format, one JSON object per pair:

```json
{"snippet": "<a ~512-token passage>", "doc": "<the whole document it came from>"}
```

`lcqa-annotation` produces these; `lcqa-synthesis` consumes them.

Each directory has its own README. Start with
[`lcqa-annotation/examples`](lcqa-annotation/examples/) if you would rather read
data than code — it holds real input and output at every stage.

## License

Code is Apache-2.0 (see [LICENSE](LICENSE)). Example data comes from
public-domain sources only; provenance is documented in
[`lcqa-annotation/examples/README.md`](lcqa-annotation/examples/README.md).
