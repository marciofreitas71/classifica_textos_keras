import pandas as pd
import numpy as np
import nltk
import json
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar e pré-processar os dados
def load_data():
    # Carregar os dados do arquivo Excel
    df = pd.read_excel('datasets/dataset_combinado.xlsx')

    # Extrair textos, rótulos e processos
    texts = df['TEXTO'].tolist()
    labels = df['CLASSE'].tolist()

    return texts, labels

def preprocess_text(texts):
    # Tokenização usando NLTK
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]

    # Remoção de stopwords e pontuações
    stop_words = set(stopwords.words('portuguese'))
    preprocessed_texts = [
        [word for word in text if word.isalnum() and word not in stop_words]
        for text in tokenized_texts
    ]

    return preprocessed_texts

# Treinamento do modelo CNN
def train_model(texts, labels):
    # Criação de um vocabulário usando NLTK para contar as palavras
    all_words = [word for text in texts for word in text]
    word_counts = Counter(all_words)
    word_index = {word: index + 1 for index, (word, _) in enumerate(word_counts.most_common())}

    # Salvar o vocabulário em um arquivo JSON
    with open('vocabulario/word_index.json', 'w') as f:
        json.dump(word_index, f)

    # Tokenização e criação das sequências
    sequences = [[word_index[word] for word in text if word in word_index] for text in texts]

    # Padding para ter sequências de igual tamanho
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    # Salvar o comprimento máximo das sequências em um arquivo JSON
    with open('vocabulario/max_length.json', 'w') as f:
        json.dump(max_length, f)

   # Codificação dos rótulos
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    # Mapear classes para índices
    label_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}

    # Salvar o mapeamento em um arquivo JSON
    with open('vocabulario/label_encoder.json', 'w') as f:
        json.dump(label_mapping, f)

    # Divisão em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    # Construção do modelo CNN
    embedding_dim = 100
    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Treinamento do modelo
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=2, batch_size=100,
        callbacks=[
            ModelCheckpoint(filepath='modelo/modelo_cnn_h5.keras', monitor='val_accuracy', save_best_only=True),
            EarlyStopping(monitor='val_accuracy', patience=4)
        ]
    )

    # Avaliação do modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Previsões do modelo
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Decodificação dos rótulos
    decoded_labels = label_encoder.inverse_transform(y_test)
    decoded_predictions = label_encoder.inverse_transform(y_pred)

    # Relatório de classificação
    print(classification_report(decoded_labels, decoded_predictions))

    # Matriz de confusão
    cm = confusion_matrix(decoded_labels, decoded_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return model

# Carregar dados
texts, labels = load_data()

# Pré-processamento do texto
preprocessed_texts = preprocess_text(texts)

# Treinamento do modelo
model = train_model(preprocessed_texts, labels)
