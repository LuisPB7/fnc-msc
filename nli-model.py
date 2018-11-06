#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# imports #
from numpy.random import seed
seed(0)
import numpy as np
import pickle
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Masking, concatenate, multiply, subtract, Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, GlobalMaxPooling1D, Input
from keras.optimizers import Adam
from my_layers import SelfAttLayer

# Some hyperparameters #
hidden_units = 300
max_seq_len = 50
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

Y_nli = np.array(Y_nli)
Y_test_nli = np.array(Y_test_nli)
Y_test_matched = np.array(Y_test_matched)
Y_test_mismatched = np.array(Y_test_mismatched)

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

with open('cider_nli.pkl', 'rb') as ciderFile:
    cider_nli_train = pickle.load(ciderFile, encoding='latin1')
    cider_snli_test = pickle.load(ciderFile, encoding='latin1')
    cider_matched = pickle.load(ciderFile, encoding='latin1')
    cider_mismatched = pickle.load(ciderFile, encoding='latin1')

cider_nli_train = np.array(cider_nli_train)
cider_snli_test = np.array(cider_snli_test)
cider_matched = np.array(cider_matched)
cider_mismatched = np.array(cider_mismatched)

########################## Definir o modelo ##################################### 

# Define some model layers #

early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1, restore_best_weights=True)
embedding_layer = Embedding( embedding_weights.shape[0], embedding_weights.shape[1], input_length=max_seq_len, weights=[embedding_weights], trainable=False )
lstm1 = LSTM(hidden_units, implementation=2, return_sequences=True, name='lstm1' )
lstm1 = Bidirectional(lstm1, name='bilstm1')

#####################################

# Define the inputs for the model #

input_premisse = Input(shape=(max_seq_len,))
input_hyp = Input(shape=(max_seq_len,))
input_overlap = Input(shape=(1,))
input_refuting = Input(shape=(15,))
input_polarity = Input(shape=(2,))
input_hand = Input(shape=(26,))
input_sim = Input(shape=(1,))
input_bleu = Input(shape=(1,))
input_rouge = Input(shape=(3,))
input_cider = Input(shape=(1,))

###############################

# Define the sentence encoder #

mask = Masking(mask_value=0, input_shape=(max_seq_len,))(input_premisse)
embed = embedding_layer(mask)
l1 = lstm1(embed)
drop1 = Dropout(0.1)(l1)
maxim = GlobalMaxPooling1D()(drop1)
att = SelfAttLayer()(drop1)
out = concatenate([maxim, att])
SentenceEncoder = Model(input_premisse, maxim, name='SentenceEncoder')

##############################

# Combining the representations #

premisse_representation = SentenceEncoder(input_premisse)
hyp_representation = SentenceEncoder(input_hyp)
concat = concatenate([premisse_representation, hyp_representation])
mul = multiply([premisse_representation, hyp_representation])
dif = subtract([premisse_representation, hyp_representation])
final_merge = concatenate([concat, mul, dif, input_overlap, input_refuting, input_polarity, input_hand, input_sim, input_bleu, input_rouge, input_cider])
drop3 = Dropout(0.1)(final_merge)
dense1 = Dense(hidden_units*2, activation='relu', name='dense1')(drop3)
drop4 = Dropout(0.1)(dense1)
dense2 = Dense(hidden_units, activation='relu', name='dense2')(drop4)
drop5 = Dropout(0.1)(dense2)
concat_model = Model([input_premisse, input_hyp, input_overlap, input_refuting, input_polarity, input_hand, input_sim, input_bleu, input_rouge, input_cider], drop5)
dense3 = Dense(3, activation='softmax')(drop5)
final_model = Model([input_premisse, input_hyp, input_overlap, input_refuting, input_polarity, input_hand, input_sim, input_bleu, input_rouge, input_cider], dense3)

###################################

final_model.compile(optimizer=Adam(amsgrad=True), loss='categorical_crossentropy', metrics=['accuracy'])

final_model.summary()

final_model.fit([X1_nli, X2_nli, overlapFeatures_nli, refutingFeatures_nli, polarityFeatures_nli, handFeatures_nli, cosFeatures_nli, bleu_nli, rouge_nli, cider_nli_train], \
                Y_nli, validation_data=([X1_test_nli, X2_test_nli, overlapFeatures_nli_test, refutingFeatures_nli_test, polarityFeatures_nli_test, handFeatures_nli_test, cosFeatures_nli_test, bleu_nli_test, rouge_nli_test, cider_snli_test], Y_test_nli), \
                callbacks=[early_stop], epochs=30, batch_size=64)

final_model.save_weights('nli-weights.h5')

# Testing on SNLI, MultiNLI Matched & Mismatched #

final_model.load_weights('nli-weights.h5')

score, acc = final_model.evaluate([X1_test_nli, X2_test_nli, overlapFeatures_nli_test, refutingFeatures_nli_test, polarityFeatures_nli_test, handFeatures_nli_test, cosFeatures_nli_test, bleu_nli_test, rouge_nli_test, cider_snli_test], Y_test_nli, batch_size=64)

print("Best accuracy on SNLI test set is " + str(acc))

score, acc = final_model.evaluate([X1_test_matched, X2_test_matched, overlapFeatures_matched_test, refutingFeatures_matched_test, polarityFeatures_matched_test, handFeatures_matched_test, cosFeatures_matched, bleu_matched, rouge_matched, cider_matched], \
                                  Y_test_matched, batch_size=64)

print("Accuracy on MultiNLI matched test set is " + str(acc))

score, acc = final_model.evaluate([X1_test_mismatched, X2_test_mismatched, overlapFeatures_mismatched_test, refutingFeatures_mismatched_test, polarityFeatures_mismatched_test, handFeatures_mismatched_test, cosFeatures_mismatched, bleu_mismatched, rouge_mismatched, cider_mismatched], Y_test_mismatched, batch_size=64)

print("Accuracy on MultiNLI mismatched test set is " + str(acc))
