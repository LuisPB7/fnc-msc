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
from keras.models import Model, Sequential, load_model
from keras.layers import Masking , Lambda , Merge,merge, Dense, Dropout, Embedding, GlobalAveragePooling1D, LSTM, GRU, Recurrent, Bidirectional, Layer, GlobalMaxPooling1D, Input, Permute, Highway, TimeDistributed
from keras import backend as K
from keras import initializations
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import cPickle

from my_layers2 import SelfAttLayer, weightedAccCallback, AlignmentAttentionLayer
from score import score_submission, print_confusion_matrix, report_score

# Some hyperparameters #
hidden_units = 150
max_seq_len = 50
max_seqs = 30
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

del overlapFeatures_nli, refutingFeatures_nli, polarityFeatures_nli, handFeatures_nli, overlapFeatures_nli_test , refutingFeatures_nli_test, \
    polarityFeatures_nli_test, handFeatures_nli_test, overlapFeatures_matched_test, refutingFeatures_matched_test, polarityFeatures_matched_test, \
    handFeatures_matched_test, overlapFeatures_mismatched_test, refutingFeatures_mismatched_test, polarityFeatures_mismatched_test, \
    handFeatures_mismatched_test, bleu_nli, bleu_nli_test, bleu_matched, bleu_mismatched, rouge_nli, rouge_nli_test, rouge_matched, rouge_mismatched

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

    
del X1_nli, X2_nli, Y_nli, X1_test_nli, X2_test_nli, Y_test_nli, X1_test_matched, X2_test_matched, Y_test_matched, X1_test_mismatched, \
    X2_test_mismatched, Y_test_mismatched

print("Opening similarities")
with open('similarity.pkl', 'rb') as inpSim:
    cosFeatures = pickle.load(inpSim)
    cosFeatures_test = pickle.load(inpSim)
    cosFeatures_nli = pickle.load(inpSim)
    cosFeatures_nli_test = pickle.load(inpSim)
    cosFeatures_matched = pickle.load(inpSim)
    cosFeatures_mismatched = pickle.load(inpSim)

cosFeatures = np.array(cosFeatures)
cosFeatures_test = np.array(cosFeatures_test)

cosFeatures_fnc = []
cosFeatures_two = []
for feat in cosFeatures:
    cosFeatures_fnc += [feat[0]]
    cosFeatures_two += [feat[1]]
cosFeatures_fnc = np.array(cosFeatures_fnc)
cosFeatures_two = np.array(cosFeatures_two)

print(cosFeatures_fnc.shape)
print(cosFeatures_two.shape)

cosFeatures_fnc_test = []
cosFeatures_two_test = []
for feat in cosFeatures_test:
    cosFeatures_fnc_test += [feat[0]]
    cosFeatures_two_test += [feat[1]]
cosFeatures_fnc_test = np.array(cosFeatures_fnc_test)
cosFeatures_two_test = np.array(cosFeatures_two_test)

del cosFeatures_nli, cosFeatures_nli_test, cosFeatures_matched, cosFeatures_mismatched

with open("train.basic.pkl", "rb") as countsTrain:
    names = cPickle.load(countsTrain)
    talos_counts_train = cPickle.load(countsTrain)

with open("test.basic.pkl", "rb") as countsTest:
    names = cPickle.load(countsTest)
    talos_counts_test = cPickle.load(countsTest)


with open("train.sim.tfidf.pkl", "rb") as tfidfSim_train:
    talos_tfidfsim_train = cPickle.load(tfidfSim_train)
    

with open("test.sim.tfidf.pkl", "rb") as tfidfSim_test:
    talos_tfidfsim_test = cPickle.load(tfidfSim_test)


with open("train.headline.svd.pkl", "rb") as svdHealine_train:
    talos_svdHeadline_train = cPickle.load(svdHealine_train)

with open("test.headline.svd.pkl", "rb") as svdHealine_test:
    talos_svdHeadline_test = cPickle.load(svdHealine_test)
    
print(type(talos_svdHeadline_test))

with open("train.body.svd.pkl", "rb") as svdBody_train:
    talos_svdBody_train = cPickle.load(svdBody_train)
    
with open("test.body.svd.pkl", "rb") as svdBody_test:
    talos_svdBody_test = cPickle.load(svdBody_test)
    
with open("train.sim.svd.pkl", "rb") as svdSim_train:
    talos_svdsim_train = cPickle.load(svdSim_train)
    
with open("test.sim.svd.pkl", "rb") as svdSim_test:
    talos_svdsim_test = cPickle.load(svdSim_test)
    
with open("train.headline.word2vec.pkl", "rb") as w2vHealine_train:
    talos_w2vHeadline_train = cPickle.load(w2vHealine_train)
    
with open("test.headline.word2vec.pkl", "rb") as w2vHealine_test:
    talos_w2vHeadline_test = cPickle.load(w2vHealine_test)
    
with open("train.body.word2vec.pkl", "rb") as w2vBody_train:
    talos_w2vBody_train = cPickle.load(w2vBody_train)
    
with open("test.body.word2vec.pkl", "rb") as w2vBody_test:
    talos_w2vBody_test = cPickle.load(w2vBody_test)
    
with open("train.sim.word2vec.pkl", "rb") as w2vSim_train:
    talos_w2vsim_train = cPickle.load(w2vSim_train)
    
with open("test.sim.word2vec.pkl", "rb") as w2vSim_test:
    talos_w2vsim_test = cPickle.load(w2vSim_test)
    
with open("train.headline.senti.pkl", "rb") as sentiHealine_train:
    talos_sentiHeadline_train = cPickle.load(sentiHealine_train)

with open("test.headline.senti.pkl", "rb") as sentiHealine_test:
    talos_sentiHeadline_test = cPickle.load(sentiHealine_test)
    
with open("train.body.senti.pkl", "rb") as sentiBody_train:
    talos_sentiBody_train = cPickle.load(sentiBody_train)
    
with open("test.body.senti.pkl", "rb") as sentiBody_test:
    talos_sentiBody_test = cPickle.load(sentiBody_test)
    
M = 5
nb_epoch = T = 100
alpha_zero = 0.001
model_prefix = 'Model_'
snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)

########################## Definir o modelo ##################################### 

# Definir algumas camadas do modelo #

early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
weightedAccuracy = weightedAccCallback(X1_test, X2_test, Y_test, overlapFeatures_fnc_test, refutingFeatures_fnc_test, polarityFeatures_fnc_test, handFeatures_fnc_test,  \
                                       cosFeatures_fnc_test,cosFeatures_two_test, bleu_fnc_test, rouge_fnc_test, X2_test_two_sentences, overlapFeatures_fnc_two_test, \
                                       refutingFeatures_fnc_two_test, polarityFeatures_fnc_two_test, handFeatures_fnc_two_test, \
                                       bleu_two_sentences_test, rouge_two_sentences_test, talos_counts_test, talos_tfidfsim_test, talos_svdHeadline_test, \
                                       talos_svdBody_test, talos_svdsim_test,talos_w2vHeadline_test, talos_w2vBody_test, talos_w2vsim_test, talos_sentiHeadline_test, talos_sentiBody_test)

embedding_layer = Embedding( embedding_weights.shape[0], embedding_weights.shape[1], input_length=max_seq_len, weights=[embedding_weights], trainable=False )
transform_layer = TimeDistributed(Dense( word_embeddings_dim, activation='relu', name='transform' ))
gru1 = GRU(hidden_units, consume_less='gpu', return_sequences=True, name='gru1' )
gru2 = GRU(hidden_units, consume_less='gpu', return_sequences=True, name='gru2') 
gru1 = Bidirectional(gru1, name='bigru1')
gru2 = Bidirectional(gru2, name='bigru2')
right_branch_gru1 = GRU(hidden_units, consume_less='gpu', return_sequences=True )
right_branch_gru2 = GRU(hidden_units, consume_less='gpu', return_sequences=True ) 
right_branch_gru1 = Bidirectional(right_branch_gru1)
right_branch_gru2 = Bidirectional(right_branch_gru2)

snli_model = load_model("snli-weights.h5", custom_objects={'SelfAttLayer':SelfAttLayer})
layer_dict = dict([(layer.name, layer) for layer in snli_model.layers])

concat_model = load_model("concat_snli.h5", custom_objects={'SelfAttLayer':SelfAttLayer})

#####################################

# Definir os inputs do modelo #

input_headline = Input(shape=(max_seq_len,))
input_two = Input(shape=(max_seq_len,))
input_body = Input(shape=(max_seqs, max_seq_len,))
input_overlap = Input(shape=(1,))
input_overlap_two = Input(shape=(1,))
input_refuting = Input(shape=(15,))
input_refuting_two = Input(shape=(15,))
input_polarity = Input(shape=(2,))
input_polarity_two = Input(shape=(2,))
input_hand = Input(shape=(26,))
input_hand_two = Input(shape=(26,))
input_sim = Input(shape=(1,))
input_sim_two = Input(shape=(1,))
input_bleu = Input(shape=(1,))
input_bleu_two = Input(shape=(1,))
input_rouge = Input(shape=(3,))
input_rouge_two = Input(shape=(3,))

input_talos_count = Input(shape=(41,))
input_talos_tfidfsim = Input(shape=(1,))
input_talos_headline_svd = Input(shape=(50,))
input_talos_body_svd = Input(shape=(50,))
input_talos_svdsim = Input(shape=(1,))
input_talos_headline_w2v = Input(shape=(300,))
input_talos_body_w2v = Input(shape=(300,))
input_talos_w2vsim = Input(shape=(1,))
input_talos_headline_senti = Input(shape=(4,))
input_talos_body_senti = Input(shape=(4,))


###############################

# Definir o sentence encoder #

mask = Masking(mask_value=0, input_shape=(max_seq_len,))(input_headline)
embed = embedding_layer(mask)
g1 = gru1(embed)
g1 = merge([embed, g1], mode='concat')
g1 = Dropout(0.01)(g1)
g2 = gru2(g1)
g2 = merge([g1, g2], mode='concat')
g2 = Dropout(0.01)(g2)
att = TimeDistributed(Dense(hidden_units))(g2)
att = SelfAttLayer(name='attention')(att)
HeadlineEncoder = Model(input_headline, att, name='HeadlineEncoder')

HeadlineEncoder.load_weights("snli-weights.h5", by_name=True)

##############################

# Definir o document encoder #

body_sentence = TimeDistributed(HeadlineEncoder)(input_body)
body_g1 = right_branch_gru1(body_sentence)
body_g1 = merge([body_sentence, body_g1], mode='concat')
body_g1 = Dropout(0.01)(body_g1)
body_g2 = right_branch_gru2(body_g1)
body_g2 = merge([body_g1,body_g2], mode='concat')
body_g2 = Dropout(0.01)(body_g2)
body_att = TimeDistributed(Dense(hidden_units))(body_g2)
body_att = SelfAttLayer()(body_att)
DocumentEncoder = Model(input_body, body_att, name='DocumentEncoder')

##############################

# Combinar as representações #

headline_representation = HeadlineEncoder(input_headline)
document_representation = DocumentEncoder(input_body)
two_sentences_align = concat_model([input_headline, input_two, input_overlap_two, input_refuting_two, input_polarity_two, input_hand_two, \
                                    input_sim_two, input_bleu_two, input_rouge_two])
concat = merge([headline_representation, document_representation], mode='concat')
mul = merge([headline_representation, document_representation], mode='mul')
dif = merge([headline_representation, document_representation], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0])
final_merge = merge([concat, mul, dif, input_overlap, input_refuting, input_polarity, input_hand, input_sim, input_bleu, input_rouge, input_talos_count, input_talos_tfidfsim, input_talos_headline_svd, input_talos_body_svd,\
                     input_talos_svdsim, input_talos_headline_w2v, input_talos_body_w2v, input_talos_w2vsim, input_talos_headline_senti, input_talos_body_senti], mode='concat')
drop3 = Dropout(0.01)(final_merge)
dense1 = Dense(hidden_units*2, activation='relu', name='dense1', weights=layer_dict['dense1'].get_weights())(drop3)
drop4 = Dropout(0.01)(dense1)
dense2 = Dense(hidden_units, activation='relu', name='dense2', weights=layer_dict['dense2'].get_weights())(drop4)
drop5 = Dropout(0.01)(dense2)

concat_final = merge([drop5, two_sentences_align, input_talos_count, input_talos_tfidfsim, input_talos_headline_svd, input_talos_body_svd, \
                     input_talos_svdsim, input_talos_headline_w2v, input_talos_body_w2v, input_talos_w2vsim, \
                     input_talos_headline_senti, input_talos_body_senti], mode='concat')
drop6 = Dropout(0.01)(concat_final)
dense3 = Dense(4, activation='softmax')(drop6)
final_model = Model([input_headline, input_body,input_overlap, input_refuting, input_polarity, input_hand, \
                     input_sim, input_sim_two, input_bleu, input_rouge, input_two, input_overlap_two, input_refuting_two, input_polarity_two, input_hand_two, \
                     input_bleu_two, input_rouge_two, input_talos_count, input_talos_tfidfsim, input_talos_headline_svd, input_talos_body_svd, \
                     input_talos_svdsim, input_talos_headline_w2v, input_talos_body_w2v, input_talos_w2vsim, \
                     input_talos_headline_senti, input_talos_body_senti], dense3)
#######################################################################################


final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

final_model.fit([X1, X2, overlapFeatures_fnc, refutingFeatures_fnc, polarityFeatures_fnc, handFeatures_fnc,  \
                 cosFeatures_fnc,cosFeatures_two, bleu_fnc, rouge_fnc, X2_two_sentences, overlapFeatures_fnc_two, refutingFeatures_fnc_two, \
                 polarityFeatures_fnc_two, handFeatures_fnc_two, bleu_two_sentences, rouge_two_sentences, \
                 talos_counts_train, talos_tfidfsim_train, talos_svdHeadline_train, talos_svdBody_train, talos_svdsim_train, \
                 talos_w2vHeadline_train, talos_w2vBody_train, talos_w2vsim_train, talos_sentiHeadline_train, talos_sentiBody_train], Y,\
                validation_data=([X1_test, X2_test, overlapFeatures_fnc_test, refutingFeatures_fnc_test, polarityFeatures_fnc_test, handFeatures_fnc_test,  \
                                  cosFeatures_fnc_test,cosFeatures_two_test, bleu_fnc_test, rouge_fnc_test, X2_test_two_sentences, overlapFeatures_fnc_two_test, \
                                  refutingFeatures_fnc_two_test, polarityFeatures_fnc_two_test, handFeatures_fnc_two_test, \
                                  bleu_two_sentences_test, rouge_two_sentences_test, talos_counts_test, talos_tfidfsim_test, talos_svdHeadline_test, talos_svdBody_test, talos_svdsim_test, \
                                  talos_w2vHeadline_test, talos_w2vBody_test, talos_w2vsim_test, talos_sentiHeadline_test, talos_sentiBody_test], Y_test), \
                callbacks=[weightedAccuracy]+snapshot.get_callbacks(model_prefix=model_prefix), batch_size=64, nb_epoch=100, class_weight={0:1, 1:3, 2:3, 3:3})


final_model.save("fnc-weights.h5")

#######################################################################################

# Testar o modelo no test set do FNC #

test_outputs = []
test_predictions = []
labels = ['unrelated', 'agree', 'disagree', 'discuss']
for target in Y_test:
    test_outputs += [labels[target.tolist().index(1)]]
aux = final_model.predict([X1_test, X2_test, overlapFeatures_fnc_test, refutingFeatures_fnc_test, polarityFeatures_fnc_test, handFeatures_fnc_test,  \
                                  cosFeatures_fnc_test,cosFeatures_two_test, bleu_fnc_test, rouge_fnc_test, X2_test_two_sentences, overlapFeatures_fnc_two_test, \
                                  refutingFeatures_fnc_two_test, polarityFeatures_fnc_two_test, handFeatures_fnc_two_test, \
                                  bleu_two_sentences_test, rouge_two_sentences_test, talos_counts_test, talos_tfidfsim_test, talos_svdHeadline_test, talos_svdBody_test, talos_svdsim_test, \
                                  talos_w2vHeadline_test, talos_w2vBody_test, talos_w2vsim_test, talos_sentiHeadline_test, talos_sentiBody_test])
for prediction in aux:
    test_predictions += [labels[prediction.argmax()]]
report_score(test_outputs, test_predictions)

######################################
