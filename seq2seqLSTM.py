
# coding: utf-8

# In[1]:

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

from sklearn.cross_validation import train_test_split
from gensim.models import word2vec
from gensim.models import KeyedVectors


# In[3]:

def translate(text_file):
    text_raw = open(text_file, 'r')
    text=[]
    for line in text_raw:
        text.append(re.compile('\w+').findall(line))
    text_raw.close()
    print("Input text size:"+str(len(text)))
    
    for sentence in text:
        sentence.append("<EOS>")        
    
    word_vectors_X = KeyedVectors.load_word2vec_format('xWordVectors.txt', binary=True)
    word_vectors_Y = KeyedVectors.load_word2vec_format('yWordVectors.txt', binary=True)
        
    print("Vocabulary loaded")  
          
    #LSTM NETWORK MODEL
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
                   
    #replace words with vocabulary vector (if not found, word will be replaced with "UNK")
    layer_cell_num=loaded_model.layers[0].input_shape[2]
    for i, sentence in enumerate(text):
        for j, word in enumerate(sentence):
            if word in word_vectors_X.vocab:
                text[i][j]=word_vectors_X.word_vec(word)
            else:
                text[i][j]=np.ones(layer_cell_num)
                
    print("Words are replaced by vectors in vocabulary")            
    
    #flip input sentences
    for sentence in text:
        sentence.reverse() 
    
    #padding and triming
    X_max_len=loaded_model.layers[0].input_shape[1]
    for sentence in text:
        dif = X_max_len - len(sentence)
        if dif<0:
            del sentence[(X_max_len-1):]
            sentence.append("<EOS>")
        while dif>0 :
            sentence.append(np.zeros(layer_cell_num))
            dif=dif-1  
            
    
    preds = loaded_model.predict(text)
    
    for i, sentence in enumerate(preds):
        for j, word in enumerate(sentence):
                print(word_vectors_Y.similar_by_vector(preds[i][j], topn=1))
                


# In[1]:

#learn
#-creates and traines LSTM neural nework of layer_cell_num layers, using X and Y datasets
def learning(X_file, Y_file, vocab, layer_cell_num):
    X_raw = open(X_file, 'r')
    Y_raw = open(Y_file, 'r')
    
    #parsing input and output files
    X=[]
    Y=[]
    for line in X_raw:
        X.append((re.compile('\w+').findall(line)))
    X_raw.close()
    print("Input sample size:"+str(len(X)))
    
    for line in Y_raw:
        Y.append((re.compile('\w+').findall(line)))
    Y_raw.close()
    print("Output sample size (has to match input):"+str(len(Y)))
    
    #due to fixed size input to keras RNN (this can be overriden, but is irrelevant in this case), padding is necessary
    X_max_len=max(len(sublist) for sublist in X)+1
    Y_max_len=max(len(sublist) for sublist in Y)+1
    
    for sentence in X:
        sentence.append("<EOS>")
    for sentence in Y:
        sentence.append("<EOS>")
        
    
    if vocab>0:
        print("Creating vocabulary")
        model_X = word2vec.Word2Vec(X, size=layer_cell_num, max_vocab_size =2*vocab)
        model_Y = word2vec.Word2Vec(Y, size=layer_cell_num, max_vocab_size =vocab)
        word_vectors_X = model_X.wv
        word_vectors_Y = model_Y.wv
        word_vectors_X.save_word2vec_format('xWordVectors.txt', binary=True)
        word_vectors_Y.save_word2vec_format('yWordVectors.txt', binary=True)
        del model_X
        del model_Y
    else:
        word_vectors_X = KeyedVectors.load_word2vec_format('xWordVectors.txt', binary=True)
        word_vectors_Y = KeyedVectors.load_word2vec_format('yWordVectors.txt', binary=True)
        
    print("Vocabulary created")
            
    #replace words with vocabulary vector (if not found, word will be replaced with "UNK")
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in word_vectors_X.vocab:
                X[i][j]=word_vectors_X.word_vec(word)
            else:
                X[i][j]=np.ones(layer_cell_num)
    for i, sentence in enumerate(Y):
        for j, word in enumerate(sentence):
            if word in word_vectors_Y.vocab:
                Y[i][j]=word_vectors_Y.word_vec(word)
            else:
                Y[i][j]=np.ones(layer_cell_num)
                
    print("Words are replaced by vectors in vocabulary")            
    
    #flip input sentences
    for sentence in X:
        sentence.reverse()   
    
    #add padding with zeros to achive a fixed size input
    
    for sentence in X:
        dif = X_max_len - len(sentence)
        while dif>0 :
            sentence.append(np.zeros(layer_cell_num))
            dif=dif-1
            
    for sentence in Y:
        dif = Y_max_len - len(sentence)
        while dif>0 :
            sentence.append(np.zeros(layer_cell_num))
            dif=dif-1        
        
    #LSTM NETWORK MODEL
    #encoder network
    print("Creating model, input shape: " + str(X_max_len) + ", "+ str(layer_cell_num))
    model=Sequential()
    
    model.add(LSTM(layer_cell_num, input_shape=(X_max_len, layer_cell_num), activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None)))
    model.add(RepeatVector(Y_max_len))
              
   
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    model.add(LSTM(layer_cell_num, activation='sigmoid', recurrent_activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), recurrent_initializer=initializers.RandomUniform(minval=-0.08, maxval=0.08, seed=None), return_sequences=True))
    print(model.summary())

    sgd = optimizers.SGD(lr=0.7, clipnorm=5)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    
    X_array=np.asarray(X)
    
    print("Training and testing model")
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
    
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

