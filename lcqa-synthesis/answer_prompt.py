"""Question + text -> the prompt the LLM answers and the prompt the model is trained on.

Both are laid out the same way. The LLM answers from the snippets (answer prompt),
while the training example shows the whole document in their place (training prompt).
"""

# ---------------------------------------------------------------------------
# Diversity knobs
# ---------------------------------------------------------------------------

QUESTION_POSITIONS = ["before", "after"]
QUESTION_PREFIXES = ["", "Question: ", "Q: ", "Query: ", "Answer this: ", "Please answer: "]
STYLES = {   # style -> (question before the text, question after the text)
    "bare": ("{question}\n\n{text}",
             "{text}\n\n{question}"),
    "context_block": ("{question}\n\nContext:\n{text}",
                      "Context:\n{text}\n\n{question}"),
    "instruction": ("{question}\n\nUse the following information to answer:\n\n{text}",
                    "Based on the following information:\n\n{text}\n\n{question}"),
    "reference": ("{question}\n\nReference material:\n{text}",
                  "Reference material:\n{text}\n\n{question}"),
}


def sample_layout(rng):
    return {"question_position": rng.choice(QUESTION_POSITIONS),
            "question_prefix": rng.choice(QUESTION_PREFIXES),
            "style": rng.choice(list(STYLES))}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def lay_out(question, text, layout):
    before, after = STYLES[layout["style"]]
    template = before if layout["question_position"] == "before" else after
    return template.format(question=layout["question_prefix"] + question, text=text)


def build_prompts(question, snippets, doc, layout):
    answer_prompt = lay_out(question, "\n\n".join(snippets), layout).rstrip() + "\n"
    training_prompt = lay_out(question, doc, layout).strip()
    return answer_prompt, training_prompt
