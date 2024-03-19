import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from nltk.tokenize import word_tokenize
import nltk
import cProfile
nltk.download('punkt')  # Baixar os recursos necessários para o tokenizador do NLTK

# Carregar o dataset
dataset = pd.read_excel('datasets/dataset_combinado.xlsx')

# Dividir os dados em features (X) e target (y)
X = dataset['TEXTO'].astype(str)
y = pd.get_dummies(dataset['CLASSE']).values

# Tokenização dos textos usando NLTK
max_features = 5000
X_tokenized = X.apply(word_tokenize)  # Tokenizar cada texto

# Converter tokens em sequências de números
word_index = {word: idx + 1 for idx, word in enumerate(set(word for sublist in X_tokenized for word in sublist))}
X_sequences = X_tokenized.apply(lambda tokens: [word_index[token] for token in tokens])

# Pad sequences
X_padded = pad_sequences(X_sequences, maxlen=100)  # Defina o tamanho máximo da sequência, se necessário

# Divisão em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Definição do modelo LSTM
embedding_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embedding_dim))  # Remova input_length daqui
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))
adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Checkpoint para salvar os pesos do modelo
filepath = "weights_best.hdf5.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Treinamento do modelo
batch_size = 32
epochs = 50
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

# Avaliação do modelo
scores = model.evaluate(X_test, y_test, verbose=0)
print("Acurácia do modelo: %.2f%%" % (scores[1]*100))

# Gerar previsões
y_pred = model.predict(X_test)
y_pred_classes = [np.argmax(y) for y in y_pred]
y_test_classes = [np.argmax(y) for y in y_test]

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Classe 1', 'Classe 2', 'Classe 3', 'Classe 4'], yticklabels=['Classe 1', 'Classe 2', 'Classe 3', 'Classe 4'])
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Real')
plt.title('Matriz de Confusão')
plt.show()

# Salvando o modelo
model.save('modelo_classificador.h5')
