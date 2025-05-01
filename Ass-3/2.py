import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, Attention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import re
import unicodedata
import os
import zipfile
import requests

# Reduce these parameters to fit your memory constraints
NUM_SAMPLES = 10000  # Reduced from 30000
MAX_VOCAB_SIZE = 5000  # Reduced from 10000
EMBEDDING_DIM = 128  # Reduced from 256
LSTM_UNITS = 256  # Reduced from 512
BATCH_SIZE = 32  # Reduced from 64
EPOCHS = 20  # Reduced from 30
MAX_SEQ_LENGTH = 15  # Reduced from 20

DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip'
DATA_FILE = 'fra-eng.zip'
TEXT_FILE = 'fra.txt'

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) 
                  if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w

def download_and_extract_data():
    if not os.path.exists(DATA_FILE):
        print("Downloading dataset...")
        response = requests.get(DATA_URL, stream=True)
        with open(DATA_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    if not os.path.exists(TEXT_FILE):
        print("Extracting dataset...")
        with zipfile.ZipFile(DATA_FILE, 'r') as zip_ref:
            zip_ref.extractall()

def load_dataset(path, num_samples):
    eng_texts = []
    fra_texts = []
    with open(path, encoding='UTF-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                eng = preprocess_sentence(parts[0])
                fra = preprocess_sentence(parts[1])
                eng_texts.append(eng)
                fra_texts.append(fra)
    return eng_texts, fra_texts

def tokenize_pad(texts, tokenizer=None, is_target=False):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='', oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    max_len = max(len(s) for s in sequences) if is_target else MAX_SEQ_LENGTH
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return padded, tokenizer, max_len

def prepare_data():
    download_and_extract_data()
    
    if not os.path.exists(TEXT_FILE):
        raise FileNotFoundError(f"Could not find {TEXT_FILE} after download and extraction")
    
    eng_texts, fra_texts = load_dataset(TEXT_FILE, NUM_SAMPLES)
    
    input_padded, input_tokenizer, input_max_len = tokenize_pad(eng_texts)
    target_padded, target_tokenizer, target_max_len = tokenize_pad(fra_texts, is_target=True)
    
    input_train, input_val, target_train, target_val = train_test_split(
        input_padded, target_padded, test_size=0.2, random_state=42)
    
    return (input_tokenizer, target_tokenizer, input_max_len, target_max_len, 
            input_train, input_val, target_train, target_val)

def build_model(input_vocab_size, target_vocab_size, input_max_len, target_max_len):
    # Encoder
    encoder_inputs = Input(shape=(input_max_len,))
    encoder_embedding = Embedding(input_vocab_size, EMBEDDING_DIM, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(target_max_len-1,))
    decoder_embedding = Embedding(target_vocab_size, EMBEDDING_DIM, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    # Simplified attention
    attention = tf.keras.layers.Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
    attention = tf.keras.layers.Activation('softmax')(attention)
    context = tf.keras.layers.Dot(axes=[2, 1])([attention, encoder_outputs])
    decoder_concat = Concatenate(axis=-1)([decoder_outputs, context])
    
    decoder_dense = Dense(target_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def translate(sentence, model, input_tokenizer, target_tokenizer, input_max_len, target_max_len):
    sentence = preprocess_sentence(sentence)
    seq = input_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=input_max_len, padding='post')
    
    # Initialize decoder input with start token
    target_seq = np.zeros((1, target_max_len-1))
    target_seq[0, 0] = target_tokenizer.word_index['<start>']
    
    # Predict one word at a time
    translated = []
    for i in range(1, target_max_len-1):
        output = model.predict([seq, target_seq], verbose=0)
        sampled_token_index = np.argmax(output[0, i-1, :])
        sampled_word = target_tokenizer.index_word.get(sampled_token_index, '')
        
        if sampled_word == '<end>' or i == target_max_len-2:
            break
            
        translated.append(sampled_word)
        target_seq[0, i] = sampled_token_index
    
    return ' '.join(translated)

def main():
    try:
        (input_tokenizer, target_tokenizer, input_max_len, target_max_len,
         input_train, input_val, target_train, target_val) = prepare_data()
        
        input_vocab_size = len(input_tokenizer.word_index) + 1
        target_vocab_size = len(target_tokenizer.word_index) + 1
        
        # Prepare decoder inputs and outputs
        decoder_input_train = target_train[:, :-1]
        decoder_output_train = target_train[:, 1:]
        decoder_input_val = target_val[:, :-1]
        decoder_output_val = target_val[:, 1:]
        
        model = build_model(input_vocab_size, target_vocab_size, input_max_len, target_max_len)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        print("Training model...")
        model.fit(
            [input_train, decoder_input_train],
            np.expand_dims(decoder_output_train, -1),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=([input_val, decoder_input_val], np.expand_dims(decoder_output_val, -1)),
            callbacks=[early_stopping]
        )
        
        # Test translations
        test_sentences = [
            "Hello",
            "How are you?",
            "What is your name?",
            "I love programming",
            "Goodbye"
        ]
        
        for sentence in test_sentences:
            translation = translate(sentence, model, input_tokenizer, target_tokenizer, 
                                  input_max_len, target_max_len)
            print(f"\nEnglish: {sentence}")
            print(f"French: {translation}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    # Disable oneDNN optimizations if needed
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()