# 1. Import library
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

import seaborn as sns
import matplotlib.pyplot as plt

# 2. Download resource NLTK
nltk.download('punkt')
nltk.download('stopwords')

# 3. Load data
df = pd.read_excel('tweets_selenium_label.xlsx')

# 4. Cleaning label
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(str).str.strip()

print("Label distribution sebelum balancing:")
print(df['label'].value_counts())

# 5. Preprocessing
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['preprocessed_text'] = df['text'].astype(str).apply(preprocess_text)

# 6. Balancing dataset
min_count = df['label'].value_counts().min()
df = df.groupby('label').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

print("Label distribution sesudah balancing:")
print(df['label'].value_counts())

# 7. Encode label
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['label'])

# 8. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['preprocessed_text'])

# 9. Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# === MODEL 1: Logistic Regression ===
print("\n==== Logistic Regression ====")
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)

y_pred_logreg = logreg_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Classification Report:\n", classification_report(y_test, y_pred_logreg, target_names=label_encoder.classes_))

# === MODEL 2: SVM ===
print("\n==== Support Vector Machine (SVM) ====")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))

# === MODEL 3: Naive Bayes ===
print("\n==== Naive Bayes ====")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb, target_names=label_encoder.classes_))

# === Predict sample text
def predict_text(text, model):
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    pred_idx = model.predict(vec)[0]
    label_name = label_encoder.inverse_transform([pred_idx])[0]
    return label_name

sample_text = "perkembangan cina sangat pesat dan sudah menyaingi amerika"

print(f"\nPrediksi untuk teks: '{sample_text}'")
print("Logistic Regression →", predict_text(sample_text, logreg_model))
print("SVM →", predict_text(sample_text, svm_model))
print("Naive Bayes →", predict_text(sample_text, nb_model))

# === Confusion Matrix SVM (contoh visualisasi 1 model)
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVM')
plt.tight_layout()
plt.show()
