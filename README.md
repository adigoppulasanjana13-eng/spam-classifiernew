# Spam Email / SMS Classifier

A binary text classification project that detects spam vs. ham (legitimate) SMS messages
using classical NLP and machine learning techniques.

## Dataset

[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — 5,572 labeled SMS messages (4,825 ham / 747 spam).

## Approach

1. **Preprocessing**: lowercasing, punctuation/number removal, tokenization, stopword
   removal, and Porter stemming.
2. **Feature extraction**: TF-IDF vectorization with unigrams + bigrams (max 3,000 features).
3. **Modeling**: trained and compared three classifiers — Naive Bayes, Logistic Regression,
   and Linear SVM.
4. **Evaluation**: accuracy, confusion matrix, precision-recall curve, and top discriminating
   terms per class.

## Results

| Model               | Accuracy |
|---------------------|----------|
| Naive Bayes         | 97.58%   |
| Logistic Regression | 97.31%   |
| **Linear SVM**      | **98.21%** |

Linear SVM was the best-performing model on the held-out test set (20% split, stratified).

```
              precision    recall  f1-score   support

         ham       0.98      0.99      0.99       966
        spam       0.96      0.90      0.93       149

    accuracy                           0.98      1115
```

### Visualizations (in `outputs/`)
- `confusion_matrix.png` — confusion matrix for the best model
- `precision_recall_curve.png` — precision-recall curve
- `top_features.png` — top spam-indicating and ham-indicating terms

## Project Structure

```
spam-classifier/
├── data/
│   └── sms.tsv              # SMS Spam Collection dataset
├── outputs/
│   ├── model.pkl             # trained best model (Linear SVM)
│   ├── vectorizer.pkl         # fitted TF-IDF vectorizer
│   ├── results.json           # accuracy summary
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   └── top_features.png
├── spam_classifier.py        # main training & evaluation script
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python spam_classifier.py
```

This will preprocess the data, train all three models, print accuracy comparisons,
and save evaluation plots + the trained model/vectorizer to `outputs/`.

## Tech Stack

Python · Pandas · Scikit-learn · NLTK · TF-IDF · Matplotlib · Seaborn
