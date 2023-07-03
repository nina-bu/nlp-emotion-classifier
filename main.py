import numpy as np
import re
import emoji
import contractions
import joblib
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

LSTM_MODEL_PATH = 'models/classifierLSTM.h5'
NB_MODEL_PATH = 'models/classifierNB.pkl'
RF_MODEL_PATH = 'models/classifierRF.pkl'
TFIDF_VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
CLASSIFIER_DATA_PATH = 'models/classifier_data.pkl'

def normalize_text(text):
    text = emoji.demojize(text)
    text = re.sub(r':[a-z_]+:', '', text) 
    text = re.sub(r'[^\w\s]', '', text)
    text = contractions.fix(text)

    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    normalized_text = ' '.join(tokens)

    return normalized_text

def preprocess_tfidf_text(text): 
    normalized_text = normalize_text(text)
    return tfidf_vectorizer.transform([normalized_text])

def preprocess_lstm_text(text):
    normalized_text = normalize_text(text)
    tokenized_text = tokenizer.texts_to_sequences([normalized_text])
    padded_text = pad_sequences(tokenized_text, maxlen=max_length)
    return padded_text

def predict_emotion(input_text):   
    tfidf_vectorized_text = preprocess_tfidf_text(input_text)
    lstm_vectorized_text = preprocess_lstm_text(input_text)

    nb_prediction = classifierNB.predict(tfidf_vectorized_text)

    rf_prediction = classifierRF.predict(tfidf_vectorized_text)

    lstm_prediction_idx = np.argmax(modelLSTM.predict(lstm_vectorized_text), axis=-1)    
    lstm_prediction = label_encoder.inverse_transform(lstm_prediction_idx)

    return nb_prediction, rf_prediction, lstm_prediction

if __name__ == '__main__':
    classifier_data = joblib.load(CLASSIFIER_DATA_PATH)
    tokenizer = classifier_data['tokenizer']
    embedding_matrix = classifier_data['embedding_matrix']
    max_length = classifier_data['max_length']
    label_encoder = classifier_data['label_encoder']

    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    classifierNB = joblib.load(NB_MODEL_PATH)
    classifierRF = joblib.load(RF_MODEL_PATH)
    modelLSTM = load_model(LSTM_MODEL_PATH)

    print("=== Emotion Classifier ===")
    print("Enter text to predict emotion (press 'x' to exit):")
    while True:
        text = input()
        if text == 'x':
            break
        emotions = predict_emotion(text)
        nb_prediction = emotions[0]
        rf_prediction = emotions[1]
        lstm_prediction = emotions[2]
        
        print("NB Prediction:", nb_prediction)
        print("RF Prediction:", rf_prediction)
        print("LSTM Prediction:", lstm_prediction)
        print()

