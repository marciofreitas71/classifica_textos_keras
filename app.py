from flask import Flask, request, jsonify
from keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

app = Flask(__name__)

# Carregar o modelo treinado
model = load_model('modelo/modelo_cnn_h5.keras')

# Carregar o vocabulário
with open('vocabulario/word_index.json', 'r') as f:
    word_index = json.load(f)

# Carregar o comprimento máximo das sequências
with open('vocabulario/max_length.json', 'r') as f:
    max_length = json.load(f)

# Função de pré-processamento
def preprocess_text(text):
    stop_words = set(stopwords.words('portuguese'))
    tokenized_text = word_tokenize(text.lower())
    preprocessed_text = [word for word in tokenized_text if word.isalnum() and word not in stop_words]
    return preprocessed_text

# Rota para classificação de texto
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    texts = data['texts']
    preprocessed_texts = [preprocess_text(text) for text in texts]
    sequences = [[word_index[word] for word in text if word in word_index] for text in preprocessed_texts]
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    predictions = model.predict(padded_sequences)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Carregar o label_encoder
    with open('vocabulario/label_encoder.json', 'r') as f:
        label_encoder = json.load(f)
    
    # Mapeamento inverso para obter o nome da classe
    inverse_label_mapping = {v: k for k, v in label_encoder.items()}
    class_names = [inverse_label_mapping[label] for label in predicted_labels]
    
    return jsonify({'predictions': class_names})

if __name__ == '__main__':
    app.run(debug=True)
