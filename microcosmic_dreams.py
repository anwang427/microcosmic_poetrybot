import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras import backend as K

#load chinese poetry corpus
text=(open("/Users/Anna/poetry_bot/poetry_corpus.txt").read())
text=text.lower()

#map all unique characters and words to a number for simpler processing. This is a 
#character level mapping; switch to word-level mapping in later iterations for improved grammer.
words = sorted(list(set(text)))
n_to_word = {n:word for n, word in enumerate(words)}
word_to_n = {word:n for n, word in enumerate(words)}

#preprocessing of data

X = [] #training array
Y = [] #target array
length = len(text)
seq_length = 100 #number of word to consider before predicting a word
for i in range(0, length-seq_length, 1):
    sequence = text[i:i + seq_length]
    label =text[i + seq_length]
    X.append([word_to_n[word] for word in sequence])
    Y.append(word_to_n[label])

#reshape array into format accepted by LSTM (number_of_sequences, length_of_sequence, number_of_features)
X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(words))
Y_modified = np_utils.to_categorical(Y)

#modelling
model = Sequential()
model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_modified, Y_modified, epochs=2, batch_size=5000) #batch size is the number of samples in each batch propogated through the network
model.save_weights('/Users/Anna/poetry_bot/models/deep_modal.h5')

#model.load_model('/Users/Anna/poetry_bot/models/deep_modal.h5')

#poetry text generation
string_mapped = X[99]
full_string = [n_to_word[value] for value in string_mapped]

for i in range(400):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(words))
    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_word[value] for value in string_mapped]
    full_string.append(n_to_word[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

txt=""
for word in full_string:
    txt = txt+word
txt