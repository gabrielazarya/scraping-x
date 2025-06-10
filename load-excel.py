# 1. Import library
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# Download resource NLTK (kalau belum)
nltk.download('punkt')
nltk.download('stopwords')

# 2. Load data yang sudah dilabeli (update path sesuai file kamu)
df = pd.read_excel('tweets_selenium_label.xlsx')

# 3. Cek isi kolom
print(df.columns)

# 4. Cek data sample
print(df.head())

# 5. Cleaning label
# Hapus baris yang label-nya kosong (NaN)
df = df.dropna(subset=['label'])

# Pastikan kolom label string, strip spasi
df['label'] = df['label'].astype(str).str.strip()

# Cek jumlah per label
print("Label distribution sebelum balancing:")
print(df['label'].value_counts())

# 6. Preprocessing
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

# Apply
df['preprocessed_text'] = df['text'].astype(str).apply(preprocess_text)

# 7. (Optional) Balancing dataset
# ------ Uncomment kalau mau coba balanced ------
min_count = df['label'].value_counts().min()
df = df.groupby('label').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
print("Label distribution sesudah balancing:")
print(df['label'].value_counts())

# 8. Encode label
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['label'])

# 9. TF-IDF Vectorization
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(df['preprocessed_text'])

# 10. Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 11. Train model sederhana (contoh: Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 12. Evaluasi model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 12. Prediksi cepat untuk teks baru
def predict_text(text):
    # Preprocessing sama seperti training
    processed = preprocess_text(text)
    # Vectorize
    vectorized = vectorizer.transform([processed])
    # Predict
    label_idx = model.predict(vectorized)[0]
    label_name = label_encoder.inverse_transform([label_idx])[0]
    return label_name

# Contoh pemakaian
sample_text = "china selalu di hati"
prediction = predict_text(sample_text)
print(f"\nPrediksi untuk teks: '{sample_text}' -> {prediction}")

# 13. Visualisasi confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Hitung confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Buat plot
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()