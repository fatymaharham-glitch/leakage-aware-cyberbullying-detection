# Beginner project presentation guide

## One-sentence explanation

This project tests whether machine learning can suggest one of six cyberbullying-related
categories for a tweet while sending uncertain cases to a human moderator.

## What problem did we solve?

Social-media moderation involves too much text for people to read manually. We built a
research model that can help organise tweets. It does not punish users and does not prove
that cyberbullying happened.

## What data did we use?

- Main Kaggle cyberbullying dataset: 47,692 original rows and 43,826 usable model rows.
- Six approximately balanced categories.
- 8,766 rows were protected as the final test and left untouched until the end.
- Davidson external dataset: 24,783 tweets labelled hate, offensive, or neither.

Important honesty point: the main dataset licence/version still needs confirmation. The
Davidson source repository includes an MIT licence.

## What did we do?

1. Experiment 1 checked the data, duplicates, similar tweets, and protected folds.
2. Experiment 2 compared text cleaning and word/character features.
3. Experiment 3 compared models, imbalance options, contextual features, and leakage.
4. Experiment 4 tested spelling changes, confidence, errors, and a separate dataset.
5. Experiment 5 evaluated the frozen model on the protected fold and exported the demo.

## Why was Logistic Regression selected?

- Best grouped-validation macro-F1: about 87.0%.
- It returns probabilities, which support a low-confidence referral rule.
- Its word and character features are easier to explain than sentence embeddings.
- Linear SVM was very close, but the difference was not statistically clear.

## Final results to remember

| Result | Value |
|---|---:|
| Protected final-test macro-F1 | **87.4%** |
| Protected final-test accuracy | **88.6%** |
| Protected final-test rows | **8,766** |
| External binary accuracy | **79.9%** |
| External harmful-content recall | **84.7%** |
| External false-positive rate | **44.0%** |
| Low-confidence referral rate on final test | **4.4%** |

The strongest final classes were ethnicity and age. `not_cyberbullying` and
`other_cyberbullying` were hardest because their meanings overlap more.

## Robustness result

Removing punctuation caused only a small drop. Repeated-letter normalisation caused almost
no change. Heavy leetspeak and partial word masking caused large drops. This shows why a
human-review fallback is necessary for disguised text.

## External evaluation explanation

The external dataset has different labels, so we converted both datasets to harmful versus
not harmful. The model detected most harmful tweets, but its 44.0% false-positive rate was
high. This is evidence of domain shift: performance changes when the data source changes.

## Confidence and human review

The selected confidence threshold is 45%. Below that value, the demo recommends human
review. This is a practical referral rule, not proof that the probabilities are perfect.

## Identity-term result

Simple paired sentences changed predictions and confidence when identity words changed.
This is a sensitivity warning only. The dataset has no reliable demographic ground truth,
so the project does not claim full fairness.

## Main limitations

- A single tweet lacks conversation context, intent, repetition, and power imbalance.
- External labels do not exactly match the six project classes.
- Obfuscated spelling can reduce performance heavily.
- External false positives are high.
- Main dataset licence/version details remain incomplete.
- Manual judgement is still required for ambiguous cases.

## Demo steps

1. Run `make demo`.
2. Open `http://127.0.0.1:8000`.
3. Enter one short example.
4. Explain the category and confidence.
5. Show that low confidence asks for human review.
6. Repeat: this is a research suggestion, not an automatic punishment decision.

## Two-minute speaking script

“My project classifies individual tweets into six cyberbullying-related categories. I first
cleaned and checked the dataset, removed conflicting duplicates, and grouped similar tweets
to reduce data leakage. I compared several text representations and models. Combined word
and character TF-IDF with balanced Logistic Regression performed best. It achieved 87.4%
macro-F1 on a protected final test of 8,766 tweets. I also tested a separate hate and
offensive-language dataset, where accuracy was 79.9%, but the false-positive rate was high
at 44%. The model was robust to punctuation changes but weak against heavy leetspeak and
masked words. Therefore, the demo sends uncertain cases to a human moderator. The system is
only a research support tool: one tweet cannot prove a complete cyberbullying event.”

## Likely questions

**Why macro-F1?** It gives every class equal importance instead of allowing larger classes
to dominate the result.

**Why group similar tweets?** Otherwise nearly identical tweets can appear in training and
testing, making performance look better than it really is.

**Why not use the sentence-transformer model?** It scored about 82.7% macro-F1, below the
simpler TF-IDF model, and was harder to explain.

**Why not automate punishment?** False positives, missing context, obfuscation, and domain
shift make human judgement essential.

**Was the final test reused?** No. Model selection was frozen first, and fold 0 was evaluated
once. The access record is saved in `experiments/experiment_5_final_evaluation/results.json`.
