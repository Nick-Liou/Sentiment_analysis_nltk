import nltk
import random
from nltk.corpus import movie_reviews
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
import numpy as np
from typing import List, Optional, Any


# Download required NLTK data
nltk.download('movie_reviews')

# Load and shuffle documents
documents : list[tuple[list[str], str]]  = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Extract only the review texts for training Word2Vec
sentences = [list(map(str.lower, movie_reviews.words(fileid)))
             for fileid in movie_reviews.fileids()]

# Train Word2Vec model (Skip-Gram)
embedding_dim = 300
w2v_model = Word2Vec(
    sentences,
    vector_size=embedding_dim,
    window=15,
    min_count=2,
    sg=1,  # Skip-Gram
    workers=8,
    epochs=10
)

def get_vector(word: str) -> Optional[np.ndarray]:
    """Fetch vector from Word2Vec model if word exists."""
    return w2v_model.wv[word] if word in w2v_model.wv else None

def get_review_vector_avg(words: List[str]) -> np.ndarray:
    """Compute average vector for a list of words."""
    vectors = [w2v_model.wv[w] for w in map(str.lower, words) if w in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(embedding_dim, dtype=np.float32)

def get_review_vector_max(words: List[str]) -> np.ndarray:
    """Compute max-pooled vector for a list of words."""
    vectors = [w2v_model.wv[w] for w in map(str.lower, words) if w in w2v_model.wv]
    return np.max(vectors, axis=0) if vectors else np.zeros(embedding_dim, dtype=np.float32)



# Prepare data
X_avg = []
X_max = []
y = []

for words, label in documents:
    X_avg.append(get_review_vector_avg(words))
    X_max.append(get_review_vector_max(words))
    y.append(1 if label == 'pos' else 0)

# Split data
X_avg_train, X_avg_test, y_train, y_test = train_test_split(X_avg, y, test_size=0.2, random_state=42)
X_max_train, X_max_test, _, _ = train_test_split(X_max, y, test_size=0.2, random_state=42)

# Train classifiers
clf_avg = LogisticRegression(max_iter=1000)
clf_avg.fit(X_avg_train, y_train)
y_pred_avg = clf_avg.predict(X_avg_test)

clf_max = LogisticRegression(max_iter=1000)
clf_max.fit(X_max_train, y_train)
y_pred_max = clf_max.predict(X_max_test)

# Results
print("Classification Report (Average Vector):")
print(classification_report(y_test, y_pred_avg))

print("Classification Report (Max Pooling Vector):")
print(classification_report(y_test, y_pred_max))

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

scale_data = True

# Split data
X_avg_train, X_avg_test, y_train, y_test = train_test_split(X_avg, y, test_size=0.2, random_state=42)
X_max_train, X_max_test, _, _ = train_test_split(X_max, y, test_size=0.2, random_state=42)

if scale_data:
    clf_avg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf_max = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
else:
    clf_avg = LogisticRegression(max_iter=1000)
    clf_max = LogisticRegression(max_iter=1000)

# Train classifiers
clf_avg.fit(X_avg_train, y_train)
y_pred_avg = clf_avg.predict(X_avg_test)

clf_max.fit(X_max_train, y_train)
y_pred_max = clf_max.predict(X_max_test)

# Results
print("Classification Report (Average Vector):")
print(classification_report(y_test, y_pred_avg))

print("Classification Report (Max Pooling Vector):")
print(classification_report(y_test, y_pred_max))

import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def build_dense_model(input_dim: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(input_dim,)),             # Cleanly define input shape
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')         # Binary classification
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Convert lists to NumPy arrays
X_avg_train_np = np.array(X_avg_train)
X_avg_test_np = np.array(X_avg_test)
X_max_train_np = np.array(X_max_train)
X_max_test_np = np.array(X_max_test)
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

print("Train labels:", np.unique(y_train_np, return_counts=True))
print("Test labels:", np.unique(y_test_np, return_counts=True))

if scale_data:
  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()
  X_avg_train_np_scaled = scaler.fit_transform(X_avg_train_np)
  X_avg_test_np_scaled = scaler.transform(X_avg_test_np)
  print("Data Scaled!")
else:
  X_avg_train_np_scaled = X_avg_train_np
  X_avg_test_np_scaled = X_avg_test_np


# Train on average vectors
print("Training model on average vectors...")
model_avg = build_dense_model(embedding_dim)
# model_avg.fit(X_avg_train_np, y_train_np, epochs=10, batch_size=32, validation_split=0.1)
model_avg.fit(
    X_avg_train_np_scaled,
    y_train_np,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)


# Evaluate
print("Evaluating model on average vectors...")
loss, accuracy_avg = model_avg.evaluate(X_avg_test_np_scaled, y_test_np)
print(f"Test Accuracy (Average Vector): {accuracy_avg:.4f}")

if scale_data:
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_max_train_np_scaled = scaler.fit_transform(X_max_train_np)
  X_max_test_np_scaled = scaler.transform(X_max_test_np)
else:
  X_max_train_np_scaled = X_max_train_np
  X_max_test_np_scaled = X_max_test_np



# Train on max-pooled vectors
print("Training model on max-pooled vectors...")
model_max = build_dense_model(embedding_dim)
model_max.fit(
    X_max_train_np_scaled,
    y_train_np,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
print("Evaluating model on max-pooled vectors...")
loss, accuracy_max = model_max.evaluate(X_max_test_np_scaled, y_test_np)
print(f"Test Accuracy (Max Pooling Vector): {accuracy_max:.4f}")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout

def build_sequence_model(seq_len: int, embedding_dim: int) -> tf.keras.Model:
    inputs = Input(shape=(seq_len, embedding_dim))
    x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def review_to_sequences(words: List[str], seq_len: int = 10) -> List[np.ndarray]:
    """Convert a list of words into non-overlapping sequences of word vectors."""
    vectors = [w2v_model.wv[w.lower()] for w in words if w.lower() in w2v_model.wv]
    sequences = []
    for i in range(0, len(vectors) - seq_len + 1, seq_len):
        seq = vectors[i:i + seq_len]
        sequences.append(np.stack(seq))  # shape: (seq_len, embedding_dim)
    return sequences

sequence_length = 15


# Shuffle and split into train/test by review (not sequences)
train_reviews, test_reviews = train_test_split(documents, test_size=0.2, random_state=42)
# Sequence training data from train_reviews
X_seq, y_seq = [], []

for words, label in train_reviews:
    sequences = review_to_sequences(words, seq_len=sequence_length)
    X_seq.extend(sequences)
    y_seq.extend([1 if label == 'pos' else 0] * len(sequences))

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

model_seq = build_sequence_model(sequence_length, embedding_dim)
model_seq.fit(
    X_seq_train, y_seq_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

loss, acc = model_seq.evaluate(X_seq_test, y_seq_test)
print(f"Sequence Model Accuracy: {acc:.4f}")

def predict_review(words: List[str], model, seq_len: int = 10) -> float:
    """Predict review sentiment by averaging predictions over all sequences."""
    sequences = review_to_sequences(words, seq_len)
    if not sequences:
        return 0.5  # neutral prediction
    sequences_np = np.array(sequences)
    preds = model.predict(sequences_np, verbose=0)
    return float(np.mean(preds))  # average prediction across sequences

sample_review_words = movie_reviews.words(movie_reviews.fileids('pos')[0])
score = predict_review(words, model_seq, seq_len=sequence_length)
print(f"Predicted sentiment score: {score:.3f} → {'Positive' if score >= 0.5 else 'Negative'}")

# Run review-level evaluation
correct = 0
for words, label in test_reviews:
    score = predict_review(words, model_seq, seq_len=sequence_length)
    pred_label = 'pos' if score >= 0.5 else 'neg'
    if pred_label == label:
        correct += 1

review_accuracy = correct / len(test_reviews)
print(f"Review-level accuracy (Case C): {review_accuracy:.4f}")