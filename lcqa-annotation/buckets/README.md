# Bucket boundaries

Three files, one per quality classifier, each holding 19 cut points. A raw score
is turned into a bucket by counting how many cut points it exceeds
(`bisect_right`), giving a bucket number from 0 to 19.

| File | Classifier | Raw scale |
|---|---|---|
| `breaks_fasttext_eli5.json` | DCLM/ELI5 fastText | probability, 0–1 |
| `breaks_fasttext_preselect.json` | PreSelect fastText | probability, 0–1 |
| `breaks_fineweb_edu_classifier.json` | FineWeb-Edu | regression, roughly 0–5 |

## Why they exist

The three classifiers return numbers that cannot be compared: two probabilities
and a regression on a different scale. A threshold of `0.7` means something
completely different to each. Mapping every raw score onto the same 0–19 scale
first makes one threshold mean one thing across all three, which is what lets
`config.json` say `>= 17` three times and have it be a single, honest rule.

## These values describe one specific corpus

The cut points are the score distribution of the corpus this pipeline was built
for. **They are not universal constants.** Applied to a different corpus they
still produce bucket numbers, but those numbers no longer mean what they claim
to — a bucket is a statement about rank *within the corpus the boundaries came
from*, not a property of the text.

If you run this pipeline on your own data and want the thresholds in
`config.json` to carry their intended meaning, recompute these files: score a
large, representative sample of your corpus with each classifier, then take 19
ascending cut points from the resulting score distribution and write them as a
plain JSON array. Evenly spaced percentiles — one every 5% — give each bucket an
equal share of the corpus and are the easiest version to reason about.

The values shipped here were produced upstream of this pipeline and are used
elsewhere in the same project, including on documents shorter than the ones
processed here, so their exact spacing is not something this repo defines.

Treat the values here as a worked reference — they make the shipped examples
reproducible and show the expected format — rather than as something to reuse
unchanged.
