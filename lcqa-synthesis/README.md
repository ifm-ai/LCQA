# lcqa-synthesis

Turns snippet/document pairs into long-context question answering data. An LLM
(Qwen3.5-397B-A17B) reads a few passages of a document (the snippets chosen by
`lcqa-annotation`), writes a question about them and answers it from the same passages. The
training example keeps the question, the LLM's thinking and its answer, but shows the whole
document in place of the passages, so the model has to find them in a long context.

1. **Question** (`question_prompt.py`). Show the LLM the snippets of one document and ask for
   one question as JSON. Only question types whose answer is fully determined by the snippets
   are used, so it stays correct when the snippets are expanded to the whole document.
   Generations that never finish thinking, and questions that are malformed or miss a
   required field, are dropped.
2. **Answer** (`answer_prompt.py`). Lay out the question with the snippets and ask the LLM to
   answer it (the answer prompt). Lay out the question the same way with the whole document
   (the training prompt).
3. **Final** (`generate.py`). The training prompt becomes the user turn; the LLM's thinking
   and answer become the assistant turn. Generations that never finish thinking or have an
   empty answer are dropped.

## Diversity knobs

Drawn per document.

- **Question** (`question_prompt.py`): task type (17, from finding a stated fact through
  comparison, bridging an indirect reference and causal or conditional reasoning to tracking
  an entity's state and stated opinions), question format (open-ended, short answer, yes/no,
  multiple choice, true/false, fill in the blank, comparative, temporal, extractive; some task
  types allow only the formats that suit them) and difficulty (easy, medium, hard, weighted
  0.1/0.4/0.5).
- **Layout** (`answer_prompt.py`): question before or after the text, a prefix (none,
  `Question: `, `Q: `, `Query: `, `Answer this: `, `Please answer: `) and one of four framings
  (bare, `Context:`, `Based on the following information:` / `Use the following information
  to answer:`, `Reference material:`).

## Files

- `question_prompt.py`: question knobs, the prompt, and parsing the LLM's JSON.
- `answer_prompt.py`: layout knobs, and the answer and training prompts.
- `generate.py`: runs both LLM steps with vLLM and builds the training conversations,
  writing `data/questions.jsonl`, `data/answers.jsonl` and `data/final.jsonl`.
- `data/input.jsonl`: 100 documents in the output format of `lcqa-annotation/snippets.py`,
  one line per document holding a list of `{"snippet", "doc"}` pairs. The first five are the
  documents followed through `lcqa-annotation/examples/traced`; the other 95 are court
  opinions (freelaw), patents (uspto), Project Gutenberg books (pg19) and European Parliament
  proceedings (europarl) from the same corpus.
- `data/questions.jsonl`: per document, the question knobs, the prompt, the raw LLM output
  (thinking, then `</think>`, then the answer) and the parsed question (`null` if dropped).
- `data/answers.jsonl`: per question, the layout knobs, the answer prompt and the raw LLM
  output.
- `data/final.jsonl`: per question, the training conversation, with the whole document in
  the user turn and the LLM's thinking (`think`) and answer (`content`) in the assistant turn.

Rows are linked across files by `id`, the line number of the document in `input.jsonl`
(counting from 0).

`python generate.py` runs the whole pipeline on `data/input.jsonl`; it needs `vllm`,
`transformers` and 8 GPUs for the model.
