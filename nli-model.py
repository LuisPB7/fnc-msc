#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# imports #
from numpy.random import seed
seed(0)
import os
import csv
import sys
import math
import random
import time
import numpy as np
import itertools
#import resource
import collections
import unicodedata
import gensim
import pickle
import keras
from nltk import tokenize
from sklearn import preprocessing
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Masking ,Flatten, Lambda ,Reshape, Merge,merge,GlobalAveragePooling1D, Convolution2D, MaxPooling2D,Dense, Dropout, Embedding, LSTM, GRU, Recurrent, Bidirectional, Layer, GlobalMaxPooling1D, Input, Permute, Highway, TimeDistributed
from keras import backend as K
from keras import initializations
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from my_layers import SelfAttLayer, AttentionWithContext

# Some hyperparameters #
hidden_units = 150
max_seq_len = 50
word_embeddings_dim = 100
max_words = 200000
########################

print("Opening features")
with open('features.pkl', 'rb') as inpFeat:
    overlapFeatures_fnc = pickle.load(inpFeat)
    refutingFeatures_fnc = pickle.load(inpFeat)
    polarityFeatures_fnc = pickle.load(inpFeat)
    handFeatures_fnc = pickle.load(inpFeat)
    overlapFeatures_fnc_test = pickle.load(inpFeat)
    refutingFeatures_fnc_test = pickle.load(inpFeat)
    polarityFeatures_fnc_test = pickle.load(inpFeat)
    handFeatures_fnc_test = pickle.load(inpFeat)
    overlapFeatures_nli = pickle.load(inpFeat)
    refutingFeatures_nli = pickle.load(inpFeat)
    polarityFeatures_nli = pickle.load(inpFeat)
    handFeatures_nli = pickle.load(inpFeat)
    overlapFeatures_nli_test = pickle.load(inpFeat)
    refutingFeatures_nli_test = pickle.load(inpFeat)
    polarityFeatures_nli_test = pickle.load(inpFeat)
    handFeatures_nli_test = pickle.load(inpFeat)
    overlapFeatures_matched_test = pickle.load(inpFeat)
    refutingFeatures_matched_test = pickle.load(inpFeat)
    polarityFeatures_matched_test = pickle.load(inpFeat)
    handFeatures_matched_test = pickle.load(inpFeat)
    overlapFeatures_mismatched_test = pickle.load(inpFeat)
    refutingFeatures_mismatched_test = pickle.load(inpFeat)
    polarityFeatures_mismatched_test = pickle.load(inpFeat)
    handFeatures_mismatched_test = pickle.load(inpFeat)
    overlapFeatures_fnc_two = pickle.load(inpFeat)
    refutingFeatures_fnc_two = pickle.load(inpFeat)
    polarityFeatures_fnc_two = pickle.load(inpFeat)
    handFeatures_fnc_two = pickle.load(inpFeat)
    overlapFeatures_fnc_two_test = pickle.load(inpFeat)
    refutingFeatures_fnc_two_test = pickle.load(inpFeat)
    polarityFeatures_fnc_two_test = pickle.load(inpFeat)
    handFeatures_fnc_two_test = pickle.load(inpFeat)
    bleu_nli = pickle.load(inpFeat)
    bleu_nli_test = pickle.load(inpFeat)
    bleu_matched = pickle.load(inpFeat)
    bleu_mismatched = pickle.load(inpFeat)
    rouge_nli = pickle.load(inpFeat)
    rouge_nli_test = pickle.load(inpFeat)
    rouge_matched = pickle.load(inpFeat)
    rouge_mismatched = pickle.load(inpFeat)
    bleu_fnc = pickle.load(inpFeat)
    bleu_fnc_test = pickle.load(inpFeat)
    bleu_two_sentences = pickle.load(inpFeat)
    bleu_two_sentences_test = pickle.load(inpFeat)
    rouge_fnc = pickle.load(inpFeat)
    rouge_fnc_test = pickle.load(inpFeat)
    rouge_two_sentences = pickle.load(inpFeat)
    rouge_two_sentences_test = pickle.load(inpFeat)


del overlapFeatures_fnc, refutingFeatures_fnc, polarityFeatures_fnc, handFeatures_fnc, \
    overlapFeatures_fnc_test, refutingFeatures_fnc_test, polarityFeatures_fnc_test, overlapFeatures_fnc_two, refutingFeatures_fnc_two,\
    polarityFeatures_fnc_two, handFeatures_fnc_two, overlapFeatures_fnc_two_test, polarityFeatures_fnc_two_test, handFeatures_fnc_two_test,\
    bleu_fnc, bleu_fnc_test, bleu_two_sentences_test, rouge_fnc, rouge_fnc_test, rouge_two_sentences, rouge_two_sentences_test

print("Opening variables")
with open('variables.pkl', 'rb') as inp:
    embedding_weights = pickle.load(inp)
    X1 = pickle.load(inp)
    X2 = pickle.load(inp)
    Y = pickle.load(inp)
    X1_test = pickle.load(inp)
    X2_test = pickle.load(inp)
    Y_test = pickle.load(inp)
    X1_nli = pickle.load(inp)
    X2_nli = pickle.load(inp)
    Y_nli = pickle.load(inp)
    X1_test_nli = pickle.load(inp)
    X2_test_nli = pickle.load(inp)
    Y_test_nli = pickle.load(inp)
    X1_test_matched = pickle.load(inp)
    X2_test_matched = pickle.load(inp)
    Y_test_matched = pickle.load(inp)
    X1_test_mismatched = pickle.load(inp)
    X2_test_mismatched = pickle.load(inp)
    Y_test_mismatched = pickle.load(inp)
    X2_two_sentences = pickle.load(inp)
    X2_test_two_sentences = pickle.load(inp)
    tokenizer = pickle.load(inp)

    
del X1, X2, Y, X1_test, X2_test, Y_test, X2_two_sentences, X2_test_two_sentences

print("Opening similarities")
with open('similarity.pkl', 'rb') as inpSim:
    cosFeatures = pickle.load(inpSim)
    cosFeatures_test = pickle.load(inpSim)
    cosFeatures_nli = pickle.load(inpSim)
    cosFeatures_nli_test = pickle.load(inpSim)
    cosFeatures_matched = pickle.load(inpSim)
    cosFeatures_mismatched = pickle.load(inpSim)

cosFeatures_nli = np.array(cosFeatures_nli)
cosFeatures_nli_test = np.array(cosFeatures_nli_test)
cosFeatures_matched = np.array(cosFeatures_matched)
cosFeatures_mismatched = np.array(cosFeatures_mismatched)
    
del cosFeatures, cosFeatures_test

########################## Definir o modelo ##################################### 

# Definir algumas camadas do modelo #

early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
embedding_layer = Embedding( embedding_weights.shape[0], embedding_weights.shape[1], input_length=max_seq_len, weights=[embedding_weights], trainable=False )
transform_layer = TimeDistributed(Dense( word_embeddings_dim, activation='tanh', name='transform' ))
gru1 = GRU(hidden_units, consume_less='gpu', return_sequences=True, name='gru1' )
gru2 = GRU(hidden_units, consume_less='gpu', return_sequences=True, name='gru2') 
gru1 = Bidirectional(gru1, name='bigru1')
gru2 = Bidirectional(gru2, name='bigru2')

#####################################

# Definir os inputs do modelo #

input_premisse = Input(shape=(max_seq_len,))
input_hyp = Input(shape=(max_seq_len,))
input_overlap = Input(shape=(1,))
input_refuting = Input(shape=(15,))
input_polarity = Input(shape=(2,))
input_hand = Input(shape=(26,))
input_sim = Input(shape=(1,))
input_bleu = Input(shape=(1,))
input_rouge = Input(shape=(3,))

###############################

# Definir o sentence encoder #

mask = Masking(mask_value=0, input_shape=(max_seq_len,))(input_premisse)
embed = embedding_layer(mask)
g1 = gru1(embed)
g1 = merge([embed, g1], mode='concat')
g1 = Dropout(0.1)(g1)
g2 = gru2(g1)
g2 = merge([g1, g2], mode='concat')
g2 = Dropout(0.1)(g2)
att = TimeDistributed(Dense(hidden_units))(g2)
att = SelfAttLayer(input_context, name='attention')(att)
SentenceEncoder = Model([input_premisse, input_context], att)

##############################

# Combinar as duas representações #

premisse_representation = SentenceEncoder([input_premisse, drop_premisse])
hyp_representation = SentenceEncoder([input_hyp, drop_hyp])
concat = merge([premisse_representation, hyp_representation], mode='concat')
mul = merge([premisse_representation, hyp_representation], mode='mul')
dif = merge([premisse_representation, hyp_representation], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0])
final_merge = merge([concat, mul, dif, input_overlap, input_refuting, input_polarity, input_hand, input_sim, input_bleu, input_rouge], mode='concat')
drop3 = Dropout(0.1)(final_merge)
dense1 = Dense(hidden_units*2, activation='relu', name='dense1')(drop3)
drop4 = Dropout(0.1)(dense1)
concat_model = Model([input_premisse, input_hyp, input_overlap, input_refuting, input_polarity, input_hand, input_sim, input_bleu, input_rouge], drop4)
dense2 = Dense(3, activation='softmax')(drop4)
final_model = Model([input_premisse, input_hyp, input_overlap, input_refuting, input_polarity, input_hand, input_sim, input_bleu, input_rouge], dense2)

###################################

final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
concat_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

final_model.fit([X1_nli, X2_nli, overlapFeatures_nli, refutingFeatures_nli, polarityFeatures_nli, handFeatures_nli, cosFeatures_nli, bleu_nli, rouge_nli], \
                Y_nli, validation_data=([X1_test_nli, X2_test_nli, overlapFeatures_nli_test, refutingFeatures_nli_test, polarityFeatures_nli_test, handFeatures_nli_test, cosFeatures_nli_test, bleu_nli_test, rouge_nli_test], Y_test_nli), \
                callbacks=[early_stop], nb_epoch=100, batch_size=64)

final_model.save("snli-pooling.h5")
concat_model.save("concat_snli_pooling.h5")

score, acc = final_model.evaluate([X1_test_nli, X2_test_nli, overlapFeatures_nli_test, refutingFeatures_nli_test, polarityFeatures_nli_test, handFeatures_nli_test, cosFeatures_nli_test, bleu_nli_test, rouge_nli_test], Y_test_nli, batch_size=64)

print("Accuracy on SNLI test set is " + str(acc))

score, acc = final_model.evaluate([X1_test_matched, X2_test_matched, overlapFeatures_matched_test, refutingFeatures_matched_test, polarityFeatures_matched_test, handFeatures_matched_test, cosFeatures_matched, bleu_matched, rouge_matched], \
                                  Y_test_matched, batch_size=64)

print("Accuracy on MultiNLI matched test set is " + str(acc))

score, acc = final_model.evaluate([X1_test_mismatched, X2_test_mismatched, overlapFeatures_mismatched_test, refutingFeatures_mismatched_test, polarityFeatures_mismatched_test, handFeatures_mismatched_test, cosFeatures_mismatched, bleu_mismatched, rouge_mismatched], Y_test_mismatched, batch_size=64)

print("Accuracy on MultiNLI mismatched test set is " + str(acc))


