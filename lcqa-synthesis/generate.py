import json
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from answer_prompt import build_prompts, sample_layout
from question_prompt import build_prompt, parse_question, question_text, sample_knobs

MODEL = "Qwen/Qwen3.5-397B-A17B"


def write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":   # vLLM's worker processes re-import this file
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, tensor_parallel_size=8, max_model_len=48100)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=16000)
    rng = random.Random(0)

    def generate(prompts):
        chats = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        return [output.outputs[0].text for output in llm.generate(chats, sampling_params)]   # thinking, </think>, answer

    # One line per document: a few of its snippets, each paired with the whole document.
    documents = [json.loads(line) for line in open("data/input.jsonl")]

    # 1. Write a question about each document's snippets.
    questions = []
    for i, pairs in enumerate(documents):
        knobs = sample_knobs(rng)
        questions.append({"id": i, "knobs": knobs, "prompt": build_prompt([p["snippet"] for p in pairs], knobs)})
    for q, generation in zip(questions, generate([q["prompt"] for q in questions])):
        q["generation"] = generation
        q["question"] = parse_question(generation, q["knobs"]["question_format"])
    write_jsonl("data/questions.jsonl", questions)

    # 2. Answer each question from the snippets. The training prompt shows the whole document instead.
    answers, training_prompts = [], []
    for q in questions:
        if q["question"] is None:
            continue
        pairs = documents[q["id"]]
        layout = sample_layout(rng)
        answer_prompt, training_prompt = build_prompts(question_text(q["question"], q["knobs"]["question_format"]),
                                                       [p["snippet"] for p in pairs], pairs[0]["doc"], layout)
        answers.append({"id": q["id"], "layout": layout, "prompt": answer_prompt})
        training_prompts.append(training_prompt)
    for a, generation in zip(answers, generate([a["prompt"] for a in answers])):
        a["generation"] = generation
    write_jsonl("data/answers.jsonl", answers)

    # 3. Training conversation: the question with the whole document, then the LLM's thinking and answer.
    final = []
    for a, training_prompt in zip(answers, training_prompts):
        if "</think>" not in a["generation"]:                 # ran out of tokens while thinking
            continue
        think, answer = (part.strip() for part in a["generation"].split("</think>", 1))
        if not answer:
            continue
        if think.endswith("cw"):                              # Qwen3.5 often ends its thinking with a stray "cw"
            think = think[:-2]
        final.append({"id": a["id"], "conversation": [
            {"role": "user", "content": training_prompt},
            {"role": "assistant", "think": think, "content": answer},
        ]})
    write_jsonl("data/final.jsonl", final)
