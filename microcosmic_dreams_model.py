#import classes and functions used to train the model
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K

#load chinese poetry corpus ASCII text into memory and convert to lowercase
raw_text= open("poetry_corpus.txt", "r").read()
raw_text=raw_text.lower()

#map all unique characters and words to an integer for simpler processing. This is a 
#character level mapping; switch to word-level mapping in later iterations for improved grammer.
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars)) #create set of all distinct characters

#preprocessing of data which has been loaded and mapped
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

#reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
#normalize
X = X / float(n_vocab)
#hot encode the output variable
y = np_utils.to_categorical(dataY)

#define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
