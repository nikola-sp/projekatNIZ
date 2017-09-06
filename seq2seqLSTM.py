
# coding: utf-8

# In[31]:

import numpy as np
import sys
import argparse
import re
from nltk import FreqDist
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Embedding, RepeatVector, Dense, TimeDistributed, Flatten
from keras import initializers
from keras import optimizers
from keras.utils import get_file
from keras.models import model_from_json

from sklearn import model_selection


# In[23]:

#create_vocab
#-creates a vocabulary of vocab_size most common words from source if vocab_size>0, without words longer than max-len
def create_vocab(source, vocab_size, name):
    print("Creating vocabulary")
    flat_source = [val for sublist in source for val in sublist]
    fdist=FreqDist(flat_source)
    most_common = fdist.most_common(vocab_size-1)
    
    dest=[]
    for word, apperances in most_common:
        dest.append(word)
    dest.append('UNK')
    dest.insert(0, '<PAD>')   #0 is used for simple masking
    
    print("Saving vocabulary")
    with open(name, "wb") as fp:
        pickle.dump(dest, fp)
        
    return dest
    


# In[ ]:

def translate(text, model):
    print('not implemented')
    


# In[40]:

#learn
#-creates and traines LSTM neural nework of layer_cell_num layers, using X and Y datasets
def learning(X_file, Y_file, vocab, layer_cell_num):
    X_raw = open(X_file, 'r')
    Y_raw = open(Y_file, 'r')
    
    #parsing input and output files
    X=[]
    Y=[]
    for line in X_raw:
        X.append(re.compile('\w+').findall(line))
    X_raw.close()
    print("Input sample size:"+str(len(X)))
    
    for line in Y_raw:
        Y.append(re.compile('\w+').findall(line))
    Y_raw.close()
    print("Output sample size (has to match input):"+str(len(Y)))
    
    #due to fixed size input to keras RNN (this can be overriden, but is irrelevant in this case), padding is necessary
    X_max_len=max(len(sublist) for sublist in X)
    Y_max_len=max(len(sublist) for sublist in Y)
    
    print("Max sentence size:"+str(X_max_len)+" "+str(Y_max_len))
    
    #create or read a vocabulary
    X_vocab=[]
    Y_vocab=[]
    if vocab>0:
        X_vocab = create_vocab(X, 2*vocab, 'vocabularySrc.txt') #160000
        Y_vocab = create_vocab(Y, vocab,'vocabularyDest.txt')   #80000
    else:
        print("Opening vocabulary")
        with open("vocabularySrc.txt", "rb") as fp:
            X_vocab = pickle.load(fp)
        with open("vocabularyDest.txt", "rb") as fp:
            Y_vocab = pickle.load(fp)
            
    #replace words with vocabulary location (if not found, word will be replaced with "UNK")
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_vocab:
                X[i][j]=X_vocab.index(word)
            else:
                X[i][j]=X_vocab.index('UNK')
    for i, sentence in enumerate(Y):
        for j, word in enumerate(sentence):
            if word in Y_vocab:
                Y[i][j]=Y_vocab.index(word)
            else:
                Y[i][j]=Y_vocab.index('UNK')
    print("Words are replaced by numbers in vocabulary")            
    
    #flip input sentences
    for sentence in X:
        rsent = list(reversed(sentence))
        sentence = rsent
        
    #add padding with zeros to achive a fixed size input
    X = pad_sequences(X, maxlen=X_max_len, dtype='int32', value=0)
    Y = pad_sequences(Y, maxlen=Y_max_len, dtype='int32', value=0)
        
        
    #LSTM NETWORK MODEL
    #encoder network
    print("Creating model")
    model=Sequential()
    
    model.add(Embedding(2*vocab, layer_cell_num, input_length = X_max_len, mask_zero = True))  #embedding of words in 1000 D space and pading elimination
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None)))
    model.add(RepeatVector(Y_max_len))
              
   
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True)) 
    model.add(Flatten())
    model.add(TimeDistributed(Dense(vocab)))
    print(model.summary())

    sgd = optimizers.SGD(lr=0.7, clipnorm=5)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    
    print("Training and testing model")
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.33)
    
    model.fit(X_train, Y_train, epochs=7, batch_size=128)
    
    model.evaluate(X_test, Y_test)
    
    return model
    


# In[39]:

def save_model(model):

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

