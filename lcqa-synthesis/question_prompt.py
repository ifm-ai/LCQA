"""Snippets of one document -> prompt asking an LLM to write a question about them.

The question is written from a few passages, but the training example shows the
whole document. So only question types whose answer stays fully determined by
the passages when they are expanded to the full document are used.
"""
import json
import re

# ---------------------------------------------------------------------------
# Diversity knobs
# ---------------------------------------------------------------------------

TASK_INSTRUCTIONS = {
    # basic retrieval
    "find_information":
        "Generate a factual question where the answer is directly stated in the passages. "
        "Target a specific piece of information (name, date, number, place, etc.).",
    "attribute_extraction":
        "Generate a question asking for a specific attribute of an entity. "
        "Example: 'What is X's role?' or 'Where is X located?' or 'What is X's value?'",
    # relationships and comparisons
    "relationship_identification":
        "Generate a question about the relationship between two entities mentioned in the passages. "
        "Example: 'How are X and Y related?' or 'What connection exists between X and Y?'",
    "comparison":
        "Generate a question comparing two or more things mentioned in the passages. "
        "Example: 'How does X differ from Y?' or 'Which is larger, X or Y?'",
    # multi-hop reasoning
    "combine_information":
        "Generate a question requiring synthesis of facts from multiple passages. "
        "The answer should combine information from at least 2 different parts.",
    "bridge_reasoning":
        "Generate a question requiring resolution of an indirect reference. "
        "Example: If 'John is CEO' and 'The company was founded in 1990', ask 'When was John's company founded?'",
    # causal and logical
    "causal_reasoning":
        "Generate a question about why something happened, where the cause is stated in the passages.",
    "consequence_result":
        "Generate a question about the outcome or result of something mentioned in the passages. "
        "Example: 'What happened as a result of X?' or 'What was the outcome of X?'",
    "purpose_goal":
        "Generate a question about the purpose or goal of something in the passages. "
        "Example: 'Why was X done?' or 'What is the purpose of X?'",
    "conditional_reasoning":
        "Generate a conditional or hypothetical question based on facts in the passages. "
        "Example: 'Based on the policy, what would happen if X?'",
    # process and method
    "method_process":
        "Generate a question about how something was done or how a process works. "
        "Example: 'How was X achieved?' or 'What method was used for X?'",
    # tracking at specific points
    "entity_tracking":
        "Generate a question about an entity's state at a specific point mentioned in the text. "
        "Example: 'What was X's role during Y?' or 'Where was X when Y happened?'",
    "variable_tracking":
        "Generate a question about a value at a specific point in the narrative. "
        "Example: 'What was the price when X happened?'",
    # understanding and inference
    "definition_explanation":
        "Generate a question asking for an explanation or definition of something in context. "
        "Example: 'What does X mean in this context?' or 'How does X work according to the text?'",
    "example_identification":
        "Generate a question asking for an example that is given in the passages. "
        "Example: 'What example is provided for X?' or 'What case illustrates X?'",
    "synthesis_inference":
        "Generate a question whose answer requires inference from stated facts (not directly stated). "
        "The answer should be logically derivable from the passage content.",
    "stance_opinion":
        "Generate a question about a position or opinion explicitly stated in the passages. "
        "Example: 'What is the author's view on X?' or 'What position does X take on Y?'",
}

FORMAT_INSTRUCTIONS = {
    "open_ended": '''Format: Open-ended question.
Output JSON:
{
    "question": "Your natural question here"
}''',

    "short_answer": '''Format: Question expecting a 1-3 word answer.
Output JSON:
{
    "question": "Question expecting a brief answer"
}''',

    "yes_no": '''Format: Yes/No question with clear answer in text.
Output JSON:
{
    "question": "Yes/No question here"
}''',

    "multiple_choice": '''Format: Multiple choice with 4 options.
Create plausible distractors that require careful reading to eliminate.
Output JSON:
{
    "question": "Your question here",
    "choices": ["A) option1", "B) option2", "C) option3", "D) option4"]
}''',

    "true_false": '''Format: True/False statement.
Create a statement that requires careful reading to verify.
Output JSON:
{
    "statement": "A claim that is either true or false based on the passages"
}''',

    "fill_in_the_blank": '''Format: Fill-in-the-blank.
Output JSON:
{
    "question": "Statement with _____ for the blank to fill"
}''',

    "comparative": '''Format: Comparative question.
Output JSON:
{
    "question": "Question comparing/contrasting items from the text"
}''',

    "temporal": '''Format: Question about timing/sequence.
Output JSON:
{
    "question": "When/before/after question about events in the passages"
}''',

    "extractive": '''Format: Question asking for exact text extraction.
Output JSON:
{
    "question": "What exact phrase/term does the text use to describe X?"
}''',
}

DIFFICULTY_INSTRUCTIONS = {
    "easy": "Difficulty: EASY - Answer directly stated, similar wording to source.",
    "medium": "Difficulty: MEDIUM - Requires connecting 2-3 pieces of info, some paraphrasing.",
    "hard": "Difficulty: HARD - Requires synthesis, different wording than source, multi-step reasoning.",
}
DIFFICULTY_WEIGHTS = [0.1, 0.4, 0.5]


def sample_knobs(rng):
    task_type = rng.choice(list(TASK_INSTRUCTIONS))
    if task_type == "comparison":                  # some task/format pairings make more sense
        question_format = rng.choice(["open_ended", "comparative"])
    elif task_type in ("entity_tracking", "variable_tracking"):
        question_format = rng.choice(["open_ended", "temporal", "short_answer"])
    elif task_type == "stance_opinion":
        question_format = rng.choice(["open_ended", "yes_no"])
    else:
        question_format = rng.choice(list(FORMAT_INSTRUCTIONS))
    difficulty = rng.choices(list(DIFFICULTY_INSTRUCTIONS), weights=DIFFICULTY_WEIGHTS)[0]
    return {"task_type": task_type, "question_format": question_format, "difficulty": difficulty}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert at creating high-quality questions for training long-context language models.

Your task: Generate a single question based on the provided text passages.

Key principles:
1. The question must be answerable ONLY from the provided passages
2. The answer must be unambiguous and fully determined by the passage content
3. The question should sound natural, like something a real user would ask
4. Avoid questions about counts, lists of "all" items, or document structure
5. Do NOT generate the answer - only the question

Output your response as a JSON object (see format instructions below)."""


def build_prompt(snippets, knobs):
    passages = "\n\n".join(f"[Passage {i + 1}]\n{snippet}" for i, snippet in enumerate(snippets))
    return f"""{SYSTEM_PROMPT}

---

## Passages

These passages come from a SINGLE document.

{passages}

---

## Task

{TASK_INSTRUCTIONS[knobs["task_type"]]}

{FORMAT_INSTRUCTIONS[knobs["question_format"]]}

{DIFFICULTY_INSTRUCTIONS[knobs["difficulty"]]}

Generate now. Output ONLY valid JSON, nothing else."""


# ---------------------------------------------------------------------------
# Parse the LLM output
# ---------------------------------------------------------------------------

def parse_question(generation, question_format):
    """LLM output -> the question as a dict, or None if it is missing or malformed."""
    if "</think>" not in generation:                      # ran out of tokens while thinking
        return None
    text = generation.split("</think>", 1)[1].strip()     # the thinking may hold drafts; never parse it
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        text = code_block.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    try:
        question = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(question, dict):
        return None

    if question_format == "true_false":
        if not question.get("statement"):
            return None
    elif not question.get("question"):
        return None
    if question_format == "multiple_choice" and len(question.get("choices") or []) < 2:
        return None
    if len((question.get("question") or question.get("statement") or "").strip()) < 5:
        return None
    return question


def question_text(question, question_format):
    if question_format == "true_false":
        return f"True or False: {question['statement']}"
    if question_format == "multiple_choice":
        return question["question"] + "\n" + "\n".join(question["choices"])
    return question["question"]
