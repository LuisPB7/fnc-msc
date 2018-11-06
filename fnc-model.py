#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# imports #
from numpy.random import seed
seed(0)
import numpy as np
import pickle
import keras
from keras.models import Model, load_model
from keras.layers import Masking, Dense, concatenate, multiply, subtract, Dropout, Embedding, LSTM, GRU, Bidirectional, GlobalMaxPooling1D, Input, TimeDistributed
from keras.optimizers import Adam

from my_layers import SelfAttLayer, weightedAccCallback
from score import score_submission, print_confusion_matrix, report_score

# Some hyperparameters #
hidden_units = 300
max_seq_len = 50
max_seqs = 30
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

cosFeatures_fnc_test = []
cosFeatures_two_test = []
for feat in cosFeatures_test:
    cosFeatures_fnc_test += [feat[0]]
    cosFeatures_two_test += [feat[1]]
cosFeatures_fnc_test = np.array(cosFeatures_fnc_test)
cosFeatures_two_test = np.array(cosFeatures_two_test)

del cosFeatures_nli, cosFeatures_nli_test, cosFeatures_matched, cosFeatures_mismatched

with open("cider_fnc.pkl", "rb") as ciderFile:
    cider_fnc_train = pickle.load(ciderFile, encoding='latin1')
    cider_fnc_test = pickle.load(ciderFile, encoding='latin1')
    cider_two_train = pickle.load(ciderFile, encoding='latin1')
    cider_two_test = pickle.load(ciderFile, encoding='latin1')

import pickle as cPickle

with open("talos/fnc-1/tree_model/train.basic.pkl", "rb") as countsTrain:
    names = cPickle.load(countsTrain)
    talos_counts_train = cPickle.load(countsTrain, encoding='latin1')

with open("talos/fnc-1/tree_model/test.basic.pkl", "rb") as countsTest:
    names = cPickle.load(countsTest)
    talos_counts_test = cPickle.load(countsTest, encoding='latin1')

with open("talos/fnc-1/tree_model/train.sim.tfidf.pkl", "rb") as tfidfSim_train:
    talos_tfidfsim_train = cPickle.load(tfidfSim_train, encoding='latin1')

with open("talos/fnc-1/tree_model/test.sim.tfidf.pkl", "rb") as tfidfSim_test:
    talos_tfidfsim_test = cPickle.load(tfidfSim_test, encoding='latin1')

with open("talos/fnc-1/tree_model/train.headline.svd.pkl", "rb") as svdHealine_train:
    talos_svdHeadline_train = cPickle.load(svdHealine_train, encoding='latin1')

with open("talos/fnc-1/tree_model/test.headline.svd.pkl", "rb") as svdHealine_test:
    talos_svdHeadline_test = cPickle.load(svdHealine_test, encoding='latin1')

with open("talos/fnc-1/tree_model/train.body.svd.pkl", "rb") as svdBody_train:
    talos_svdBody_train = cPickle.load(svdBody_train, encoding='latin1')

with open("talos/fnc-1/tree_model/test.body.svd.pkl", "rb") as svdBody_test:
    talos_svdBody_test = cPickle.load(svdBody_test, encoding='latin1')

with open("talos/fnc-1/tree_model/train.sim.svd.pkl", "rb") as svdSim_train:
    talos_svdsim_train = cPickle.load(svdSim_train, encoding='latin1')

with open("talos/fnc-1/tree_model/test.sim.svd.pkl", "rb") as svdSim_test:
    talos_svdsim_test = cPickle.load(svdSim_test, encoding='latin1')

with open("talos/fnc-1/tree_model/train.headline.word2vec.pkl", "rb") as w2vHealine_train:
    talos_w2vHeadline_train = cPickle.load(w2vHealine_train, encoding='latin1')

with open("talos/fnc-1/tree_model/test.headline.word2vec.pkl", "rb") as w2vHealine_test:
    talos_w2vHeadline_test = cPickle.load(w2vHealine_test, encoding='latin1')

with open("talos/fnc-1/tree_model/train.body.word2vec.pkl", "rb") as w2vBody_train:
    talos_w2vBody_train = cPickle.load(w2vBody_train, encoding='latin1')

with open("talos/fnc-1/tree_model/test.body.word2vec.pkl", "rb") as w2vBody_test:
    talos_w2vBody_test = cPickle.load(w2vBody_test, encoding='latin1')

with open("talos/fnc-1/tree_model/train.sim.word2vec.pkl", "rb") as w2vSim_train:
    talos_w2vsim_train = cPickle.load(w2vSim_train, encoding='latin1')

with open("talos/fnc-1/tree_model/test.sim.word2vec.pkl", "rb") as w2vSim_test:
    talos_w2vsim_test = cPickle.load(w2vSim_test, encoding='latin1')

with open("talos/fnc-1/tree_model/train.headline.senti.pkl", "rb") as sentiHealine_train:
    talos_sentiHeadline_train = cPickle.load(sentiHealine_train, encoding='latin1')

with open("talos/fnc-1/tree_model/test.headline.senti.pkl", "rb") as sentiHealine_test:
    talos_sentiHeadline_test = cPickle.load(sentiHealine_test, encoding='latin1')

with open("talos/fnc-1/tree_model/train.body.senti.pkl", "rb") as sentiBody_train:
    talos_sentiBody_train = cPickle.load(sentiBody_train, encoding='latin1')

with open("talos/fnc-1/tree_model/test.body.senti.pkl", "rb") as sentiBody_test:
    talos_sentiBody_test = cPickle.load(sentiBody_test, encoding='latin1')

########################## Definir o modelo ##################################### 

# Define some model layers #

early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1, restore_best_weights=True)
weightedAccuracy = weightedAccCallback(X1_test, X2_test, Y_test, overlapFeatures_fnc_test, refutingFeatures_fnc_test, polarityFeatures_fnc_test, handFeatures_fnc_test,  \
                                       cosFeatures_fnc_test,cosFeatures_two_test, bleu_fnc_test, rouge_fnc_test,cider_fnc_test, X2_test_two_sentences, overlapFeatures_fnc_two_test, \
                                       refutingFeatures_fnc_two_test, polarityFeatures_fnc_two_test, handFeatures_fnc_two_test, \
                                       bleu_two_sentences_test, rouge_two_sentences_test, cider_two_test, talos_counts_test, talos_tfidfsim_test, talos_svdHeadline_test, \
                                       talos_svdBody_test, talos_svdsim_test,talos_w2vHeadline_test, talos_w2vBody_test, talos_w2vsim_test, talos_sentiHeadline_test, talos_sentiBody_test)

embedding_layer = Embedding( embedding_weights.shape[0], embedding_weights.shape[1], input_length=max_seq_len, weights=[embedding_weights], trainable=False )
lstm1 = LSTM(hidden_units, implementation=2, return_sequences=True, name='lstm1' )
lstm1 = Bidirectional(lstm1, name='bilstm1')
right_branch_lstm1 = LSTM(hidden_units, implementation=2, return_sequences=True )
right_branch_lstm1 = Bidirectional(right_branch_lstm1)

#####################################

# Define the inputs for the model #

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
input_cider = Input(shape=(1,))
input_cider_two = Input(shape=(1,))

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

# Define the sentence encoder #

mask = Masking(mask_value=0, input_shape=(max_seq_len,))(input_headline)
embed = embedding_layer(mask)
l1 = lstm1(embed)
drop1 = Dropout(0.1)(l1)
maxim = GlobalMaxPooling1D()(drop1)
att = SelfAttLayer(name='attention')(drop1)
out = concatenate([maxim, att])
HeadlineEncoder = Model(input_headline, maxim, name='HeadlineEncoder')

HeadlineEncoder.set_weights(layer_dict['SentenceEncoder'].get_weights())

##############################

# Define the document encoder #

body_sentence = TimeDistributed(HeadlineEncoder)(input_body)
body_g1 = right_branch_lstm1(body_sentence)
body_g1 = Dropout(0.1)(body_g1)
body_maxim = GlobalMaxPooling1D()(body_g1)
body_att = SelfAttLayer()(body_g1)
body_out = concatenate([body_maxim, body_att])
DocumentEncoder = Model(input_body, body_maxim, name='DocumentEncoder')

##############################

# Combining both representations #

headline_representation = HeadlineEncoder(input_headline)
document_representation = DocumentEncoder(input_body)

# Match between headline and first two sentences from body #

two_sentences_representation = HeadlineEncoder(input_two)
concat_two = concatenate([headline_representation, two_sentences_representation])
mul_two = multiply([headline_representation, two_sentences_representation])
dif_two = subtract([headline_representation, two_sentences_representation])
final_merge_two = concatenate([concat_two, mul_two, dif_two, input_overlap_two, input_refuting_two, input_polarity_two, input_hand_two, \
                               input_sim_two, input_bleu_two, input_rouge_two, input_cider_two])
drop3_two = Dropout(0.1)(final_merge_two)
dense1_two = Dense(hidden_units*2, activation='relu', weights=layer_dict['dense1'].get_weights())(drop3_two)
drop4_two = Dropout(0.1)(dense1_two)
dense2_two = Dense(hidden_units, activation='relu',weights=layer_dict['dense2'].get_weights())(drop4_two)
match = Dropout(0.1)(dense2_two)

#####################################################

concat = concatenate([headline_representation, document_representation])
mul = multiply([headline_representation, document_representation])
dif = subtract([headline_representation, document_representation])
final_merge = concatenate([concat, mul, dif, input_overlap, input_refuting, input_polarity, input_hand, input_sim, input_bleu, input_rouge, input_cider])
drop3 = Dropout(0.1)(final_merge)
dense1 = Dense(hidden_units*2, activation='relu', name='dense1', weights=layer_dict['dense1'].get_weights())(drop3)
drop4 = Dropout(0.1)(dense1)
dense2 = Dense(hidden_units, activation='relu', name='dense2', weights=layer_dict['dense2'].get_weights())(drop4)
drop5 = Dropout(0.1)(dense2)
concat_final = concatenate([drop5,match,input_talos_count, input_talos_tfidfsim, input_talos_headline_svd, input_talos_body_svd, \
                     input_talos_svdsim, input_talos_headline_w2v, input_talos_body_w2v, input_talos_w2vsim, \
                     input_talos_headline_senti, input_talos_body_senti])
drop6 = Dropout(0.1)(concat_final)
dense3 = Dense(4, activation='softmax')(drop6)
final_model = Model([input_headline, input_body,input_overlap, input_refuting, input_polarity, input_hand, \
                     input_sim, input_sim_two, input_bleu, input_rouge,input_cider, input_two, input_overlap_two, input_refuting_two, input_polarity_two, input_hand_two, \
                     input_bleu_two, input_rouge_two, input_cider_two, input_talos_count, input_talos_tfidfsim, input_talos_headline_svd, input_talos_body_svd, \
                     input_talos_svdsim, input_talos_headline_w2v, input_talos_body_w2v, input_talos_w2vsim, \
                     input_talos_headline_senti, input_talos_body_senti], dense3)
#######################################################################################

final_model.summary()

final_model.compile(optimizer=Adam(amsgrad=True), loss='categorical_crossentropy', metrics=['accuracy'])

final_model.fit([X1, X2, overlapFeatures_fnc, refutingFeatures_fnc, polarityFeatures_fnc, handFeatures_fnc,  \
                 cosFeatures_fnc,cosFeatures_two, bleu_fnc, rouge_fnc, cider_fnc_train, X2_two_sentences, overlapFeatures_fnc_two, refutingFeatures_fnc_two, \
                 polarityFeatures_fnc_two, handFeatures_fnc_two, bleu_two_sentences, rouge_two_sentences,cider_two_train, \
                 talos_counts_train, talos_tfidfsim_train, talos_svdHeadline_train, talos_svdBody_train, talos_svdsim_train, \
                 talos_w2vHeadline_train, talos_w2vBody_train, talos_w2vsim_train, talos_sentiHeadline_train, talos_sentiBody_train], Y,\
                validation_data=([X1_test, X2_test, overlapFeatures_fnc_test, refutingFeatures_fnc_test, polarityFeatures_fnc_test, handFeatures_fnc_test,  \
                                  cosFeatures_fnc_test,cosFeatures_two_test, bleu_fnc_test, rouge_fnc_test, cider_fnc_test, X2_test_two_sentences, overlapFeatures_fnc_two_test, \
                                  refutingFeatures_fnc_two_test, polarityFeatures_fnc_two_test, handFeatures_fnc_two_test, \
                                  bleu_two_sentences_test, rouge_two_sentences_test, cider_two_test, talos_counts_test, talos_tfidfsim_test, talos_svdHeadline_test, talos_svdBody_test, talos_svdsim_test, \
                                  talos_w2vHeadline_test, talos_w2vBody_test, talos_w2vsim_test, talos_sentiHeadline_test, talos_sentiBody_test], Y_test), \
                callbacks=[weightedAccuracy, early_stop], batch_size=64, epochs=100, class_weight={0:1, 1:3, 2:3, 3:3})

final_model.save_weights("fnc-weights.h5")

#######################################################################################

final_model.load_weights("fnc-weights.h5")

# Test the model on the FNC Test Set #

test_outputs = []
test_predictions = []
labels = ['unrelated', 'agree', 'disagree', 'discuss']
for target in Y_test:
    test_outputs += [labels[target.tolist().index(1)]]

aux = final_model.predict([X1_test, X2_test, overlapFeatures_fnc_test, refutingFeatures_fnc_test, polarityFeatures_fnc_test, handFeatures_fnc_test,  \
                                  cosFeatures_fnc_test, cosFeatures_two_test, bleu_fnc_test, rouge_fnc_test, cider_fnc_test, \
                                  X2_test_two_sentences, overlapFeatures_fnc_two_test, refutingFeatures_fnc_two_test, polarityFeatures_fnc_two_test, handFeatures_fnc_two_test, \
                                  bleu_two_sentences_test, rouge_two_sentences_test, cider_two_test, talos_counts_test, talos_tfidfsim_test, talos_svdHeadline_test, talos_svdBody_test, talos_svdsim_test, \
                                  talos_w2vHeadline_test, talos_w2vBody_test, talos_w2vsim_test, talos_sentiHeadline_test, talos_sentiBody_test])
preds = []
for prediction in aux:
    pred = prediction.argmax()
    if pred == 0:
        one_hot = [1,0,0,0]
    if pred == 1:
        one_hot = [0,1,0,0]
    if pred == 2:
        one_hot = [0,0,1,0]
    if pred == 3:
        one_hot = [0,0,0,1]
    preds += [one_hot]
    test_predictions += [labels[prediction.argmax()]]
report_score(test_outputs, test_predictions)

######################################
