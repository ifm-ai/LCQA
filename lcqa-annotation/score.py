"""
Step 2 of 3: score every section.

Four judgements per section, three about quality and one about safety:

  eli5        fastText, DCLM's "does this read like a helpful explanation"
  preselect   fastText, PreSelect's "is this worth pretraining on"
  fineweb_edu FineWeb-Edu transformer, "is this educational"
  toxicity    Detoxify multilingual, several dimensions

The three quality scores live on incomparable scales (two probabilities and a
roughly 0-5 regression), so each is also mapped to a **bucket** 0-19 against cut
points taken from the corpus score distribution (buckets/*.json). A bucket is a
rank rather than a score: the same bucket number means the same standing in the
corpus whichever classifier produced it, and that is what makes a single
threshold meaningful across all three.

Model weights are not in this repo; see the README.
"""

import bisect
import json
import os

import torch
from fasttext.FastText import _FastText
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from detoxify import Detoxify

MODELS_DIR = os.environ.get("LCQA_MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))
BUCKETS_DIR = os.path.join(os.path.dirname(__file__), "buckets")

_files = {
    "eli5": "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
    "preselect": "PreSelect-classifier.bin",
    "fineweb_edu": "fineweb-edu-classifier",
}
_buckets = {
    "eli5": "breaks_fasttext_eli5.json",
    "preselect": "breaks_fasttext_preselect.json",
    "fineweb_edu": "breaks_fineweb_edu_classifier.json",
}

_loaded = {}


def _model(name, device="cuda:0"):
    """Load a model the first time it is asked for."""
    if name not in _loaded:
        path = os.path.join(MODELS_DIR, _files.get(name, ""))
        if name in ("eli5", "preselect"):
            _loaded[name] = _FastText(model_path=path)
        elif name == "fineweb_edu":
            _loaded[name] = (
                AutoTokenizer.from_pretrained(path),
                AutoModelForSequenceClassification.from_pretrained(path).to(device),
            )
        elif name == "toxicity":
            _loaded[name] = Detoxify("multilingual", device=device)
    return _loaded[name]


def to_bucket(score, name):
    """Map a raw score to its 0-19 corpus bucket."""
    if name not in _loaded.setdefault("_breaks", {}):
        with open(os.path.join(BUCKETS_DIR, _buckets[name])) as f:
            _loaded["_breaks"][name] = json.load(f)
    return bisect.bisect_right(_loaded["_breaks"][name], score)


def _fasttext(name, sections):
    """fastText gives P(label); flip it when the label is the negative class."""
    labels, probs = _model(name).predict([s.replace("\n", " ") for s in sections])
    return [float(1.0 - p[0]) if l[0] in ("__label__0", "__label__cc") else float(p[0])
            for l, p in zip(labels, probs)]


def _fineweb_edu(sections, batch_size, device):
    tokenizer, model = _model("fineweb_edu", device)
    scores = []
    for i in range(0, len(sections), batch_size):
        batch = tokenizer(sections[i:i + batch_size], return_tensors="pt",
                          padding="longest", truncation=True).to(device)
        with torch.no_grad():
            logits = model(**batch).logits.squeeze(-1).float().cpu()
        scores.extend(logits.reshape(-1).tolist())
    return scores


def _toxicity(sections, batch_size, device):
    model, out = _model("toxicity", device), {}
    for i in range(0, len(sections), batch_size):
        for key, values in model.predict(sections[i:i + batch_size]).items():
            out.setdefault(key, []).extend(
                float(v) for v in (values if isinstance(values, list) else [values]))
    return out


def score_sections(sections, batch_size=32, device="cuda:0"):
    """Score every section. Returns {score_name: [one value per section]}."""
    scores = {
        "eli5_scores": _fasttext("eli5", sections),
        "preselect_scores": _fasttext("preselect", sections),
        "fineweb_edu_scores": _fineweb_edu(sections, batch_size, device),
    }
    for name in ("eli5", "preselect", "fineweb_edu"):
        scores[f"{name}_scores_bucket"] = [to_bucket(s, name) for s in scores[f"{name}_scores"]]
    for dimension, values in _toxicity(sections, batch_size, device).items():
        scores[f"{dimension}_scores"] = values
    return scores
