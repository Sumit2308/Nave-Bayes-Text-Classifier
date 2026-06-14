# 📰 Naïve Bayes Text Classifier — ML Project 5

A Naïve Bayes text classification model built on the
Wikipedia dataset from Project 2.

## Dataset
Reused from Project 2 — 60 Wikipedia articles across 4 categories:
- Artificial Intelligence
- Space Exploration
- Human Biology
- Famous Scientists

Columns used: `clean_text`, `label`

## Approach
TF-IDF + Naïve Bayes Pipeline

| Step | Tool |
|---|---|
| Text vectorization | TfidfVectorizer (max 5000 features, bigrams) |
| Classification | MultinomialNB, BernoulliNB, ComplementNB |
| Evaluation | Accuracy, F1, Confusion Matrix |

## Model Comparison

| Model | Accuracy | F1 Score |
|---|---|---|
| **MultinomialNB** | **1.0000** ✅ | **1.0000** |
| BernoulliNB | (your value) | (your value) |
| ComplementNB | (your value) | (your value) |

## Key Insights
- All 4 categories have very distinct vocabularies
  (quantum/physics vs cell/dna vs model/learning vs orbit/apollo)
- TF-IDF captures domain-specific terms perfectly
- MultinomialNB is best for multi-class text classification
- Laplace smoothing (alpha) had no effect here — dataset is clean and balanced
- 100% accuracy reflects highly separable categories, not overfitting

## Previous Assignment
Dataset source: [Project 2 — Text Preprocessing Pipeline](https://github.com/Sumit2308/text-preprocessing-nlp)

## Tech Stack
`Python` `scikit-learn` `NLTK` `pandas` `matplotlib` `seaborn`
