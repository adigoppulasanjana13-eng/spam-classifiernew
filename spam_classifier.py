"""
Spam Email / SMS Classifier
============================
Binary text classification on the SMS Spam Collection dataset (5,574 messages).

Pipeline:
    1. Load data
    2. Preprocess (lowercase, remove punctuation, tokenize, remove stopwords, Porter stemming)
    3. TF-IDF vectorization (unigrams + bigrams, max 3000 features)
    4. Train & compare Naive Bayes, Logistic Regression, Linear SVM
    5. Evaluate best model: accuracy, confusion matrix, precision-recall curve, top features
    6. Save trained model + vectorizer for reuse

Run:
    python spam_classifier.py
"""

import re
import string
import json

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("data/sms.tsv", sep="\t", header=None, names=["label", "message"])
print(f"Total messages: {len(df)}")
print(df["label"].value_counts())

# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)  # remove standalone numbers
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


print("\nPreprocessing messages (tokenization, stopword removal, stemming)...")
df["clean_message"] = df["message"].apply(preprocess)
df["label_num"] = (df["label"] == "spam").astype(int)

# ---------------------------------------------------------------------------
# 3. TF-IDF Vectorization (unigrams + bigrams, 3000 features)
# ---------------------------------------------------------------------------
print("\nVectorizing with TF-IDF (unigrams + bigrams, max_features=3000)...")
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_message"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ---------------------------------------------------------------------------
# 4. Train & compare models
# ---------------------------------------------------------------------------
print("\nTraining models...")
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
}

results = {}
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    trained_models[name] = model
    print(f"  {name:20s} accuracy: {acc * 100:.2f}%")

best_name = max(results, key=results.get)
best_model = trained_models[best_name]
print(f"\nBest model: {best_name} ({results[best_name] * 100:.2f}% accuracy)")

# ---------------------------------------------------------------------------
# 5. Evaluation: confusion matrix, classification report, precision-recall, top features
# ---------------------------------------------------------------------------
preds = best_model.predict(X_test)
print("\nClassification Report (best model):")
report = classification_report(y_test, preds, target_names=["ham", "spam"])
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title(f"Confusion Matrix - {best_name}")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=120)
plt.close()
print("Saved outputs/confusion_matrix.png")

# Precision-Recall curve (need probability/decision scores)
if hasattr(best_model, "predict_proba"):
    scores = best_model.predict_proba(X_test)[:, 1]
else:
    scores = best_model.decision_function(X_test)

precision, recall, _ = precision_recall_curve(y_test, scores)
plt.figure(figsize=(5, 4))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve - {best_name}")
plt.tight_layout()
plt.savefig("outputs/precision_recall_curve.png", dpi=120)
plt.close()
print("Saved outputs/precision_recall_curve.png")

# Top discriminating features (works directly for linear models)
feature_names = np.array(vectorizer.get_feature_names_out())
if hasattr(best_model, "coef_"):
    coefs = best_model.coef_[0]
elif hasattr(best_model, "feature_log_prob_"):
    # For Naive Bayes: difference in log-prob between spam and ham classes
    coefs = best_model.feature_log_prob_[1] - best_model.feature_log_prob_[0]
else:
    coefs = None

if coefs is not None:
    top_spam_idx = np.argsort(coefs)[-15:][::-1]
    top_ham_idx = np.argsort(coefs)[:15]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].barh(feature_names[top_spam_idx][::-1], coefs[top_spam_idx][::-1], color="crimson")
    axes[0].set_title("Top SPAM-indicating terms")
    axes[1].barh(feature_names[top_ham_idx][::-1], coefs[top_ham_idx][::-1], color="seagreen")
    axes[1].set_title("Top HAM-indicating terms")
    plt.tight_layout()
    plt.savefig("outputs/top_features.png", dpi=120)
    plt.close()
    print("Saved outputs/top_features.png")

# ---------------------------------------------------------------------------
# 6. Save model + vectorizer + results summary
# ---------------------------------------------------------------------------
joblib.dump(best_model, "outputs/model.pkl")
joblib.dump(vectorizer, "outputs/vectorizer.pkl")

summary = {
    "dataset_size": len(df),
    "train_size": X_train.shape[0],
    "test_size": X_test.shape[0],
    "model_accuracies": {k: round(v * 100, 2) for k, v in results.items()},
    "best_model": best_name,
    "best_accuracy": round(results[best_name] * 100, 2),
}
with open("outputs/results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved outputs/model.pkl, outputs/vectorizer.pkl, outputs/results.json")
print("\nDone.")
