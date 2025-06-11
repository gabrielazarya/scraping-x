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
from sklearn.utils.class_weight import compute_class_weight  # Tambahkan ini
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

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

# 6. Balancing dataset (optional, boleh pakai)
min_count = df['label'].value_counts().min()
df = df.groupby('label').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

print("Label distribution sesudah balancing:")
print(df['label'].value_counts())

# 7. Encode label
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['label'])

# 8. Tokenizing
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['preprocessed_text'])

# 9. Sequences & Padding
sequences = tokenizer.texts_to_sequences(df['preprocessed_text'])
padded_sequences = pad_sequences(sequences, padding='post', maxlen=50)

# 10. Split data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# 11. Build LSTM Model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=50))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # 3 kelas: netral, pro amerika, pro china

# 12. Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 13. Hitung class_weight (DITAMBAHKAN)
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# 14. Train model (PAKAI class_weight)
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=16,
                    validation_data=(X_test, y_test),
                    class_weight=class_weights)

# 15. Evaluasi
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Simpan model ke file .h5
model.save("lstm_sentimen_model.h5")

# Simpan tokenizer
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Simpan label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


# 16. Prediksi cepat
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_text(text):
    processed = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, padding='post', maxlen=50)
    pred_prob = model.predict(padded)
    label_idx = np.argmax(pred_prob, axis=1)[0]
    label_name = label_encoder.inverse_transform([label_idx])[0]
    return label_name

# Contoh
sample_text = "china lebih unggul dalam menghadapi perang dagang ini"
prediction = predict_text(sample_text)
print(f"\nPrediksi untuk teks: '{sample_text}' -> {prediction}")

# 17. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
