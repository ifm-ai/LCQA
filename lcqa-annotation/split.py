"""
Step 1 of 3: cut a long document into sections of roughly `max_tokens`.

Why sections at all: a 100k-token document is rarely uniformly good or
uniformly junk, so judging quality on the whole document throws away the
information we actually need. Sections are the unit we score.

Two families of strategy:

  "<kind>_only"  hand the whole document to the splitter and let it decide
  "<kind>"       first cut on sentence boundaries, then merge neighbouring
                 sentences up to the token budget, and only call the splitter
                 on pieces that are still too big

where <kind> is semantic (prose), markdown (papers, books) or code.
Prose splits fine either way; structured text keeps its headings and code
blocks intact when the matching splitter is used.
"""

import tiktoken
from semantic_text_splitter import TextSplitter, MarkdownSplitter, CodeSplitter
import tree_sitter_python

SPLITTERS = ("semantic", "markdown", "code",
             "semantic_only", "markdown_only", "code_only")

_cache = {}


def _tools(model, max_tokens):
    """Build (and remember) the tokenizer and the three splitters."""
    key = (model, max_tokens)
    if key not in _cache:
        _cache[key] = {
            "encoding": tiktoken.encoding_for_model(model),
            "semantic": TextSplitter.from_tiktoken_model(model, max_tokens),
            "markdown": MarkdownSplitter.from_tiktoken_model(model, max_tokens),
            "code": CodeSplitter.from_tiktoken_model(
                tree_sitter_python.language(), model, max_tokens),
        }
    return _cache[key]


def count_tokens(text, model="gpt-3.5-turbo", max_tokens=512):
    return len(_tools(model, max_tokens)["encoding"].encode(text, disallowed_special=()))


def _sentences(text):
    """Group lines into sentences: keep appending lines until one ends in . ? !"""
    sentences, buffer = [], []
    for line in text.split("\n"):
        buffer.append(line.strip())
        if buffer[-1].rstrip().endswith((".", "?", "!")):
            sentences.append("\n".join(buffer))
            buffer = []
    if buffer:
        sentences.append("\n".join(buffer))
    return sentences


def split_document(text, max_tokens=512, splitter="semantic", model="gpt-3.5-turbo"):
    """Split `text` into a list of sections of roughly `max_tokens` tokens."""
    kind = splitter.replace("_only", "")
    chunks = _tools(model, max_tokens)[kind].chunks

    # "<kind>_only": the splitter sees the whole document.
    if splitter.endswith("_only"):
        return chunks(text)

    # Hybrid: merge sentences up to the budget, split anything still oversized.
    sections, current, current_tokens = [], [], 0
    for sentence in _sentences(text):
        n = count_tokens(sentence, model, max_tokens)

        if n > max_tokens:                       # one sentence blows the budget
            if current_tokens > max_tokens / 2:  # flush what we have first,
                sections.append("\n".join(current))
            elif current:                        # unless it is too small to stand alone
                sentence = "\n".join(current) + "\n" + sentence
            sections.extend(chunks(sentence))
            current, current_tokens = [], 0

        elif current_tokens + n > max_tokens:    # adding it would overflow
            if current_tokens > max_tokens / 2:
                sections.append("\n".join(current))
                current, current_tokens = [sentence], n
            else:
                sections.extend(chunks("\n".join(current) + "\n" + sentence))
                current, current_tokens = [], 0

        else:
            current.append(sentence)
            current_tokens += n

    if current:
        sections.append("\n".join(current))
    return sections
