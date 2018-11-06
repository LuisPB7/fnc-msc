#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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
import collections
import unicodedata
import keras
from sklearn import preprocessing
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Masking , Lambda , Dense, Dropout, Embedding, LSTM, GRU, Recurrent, Bidirectional, Layer, GlobalMaxPooling1D, Input, Permute, Highway, TimeDistributed
from keras import backend as K
from keras import initializers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from score import report_score

max_seq_len = 50
hidden_units = 300

class GlobalMaxPooling1DMasked(GlobalMaxPooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(GlobalMaxPooling1DMasked, self).__init__(**kwargs)
    def build(self, input_shape): super(GlobalMaxPooling1DMasked, self).build(input_shape)
    def call(self, x, mask=None): return super(GlobalMaxPooling1DMasked, self).call(x)

class SelfAttLayer(Layer):
    def __init__(self, **kwargs):
        self.attention = None
        self.init = initializers.get('normal')
        self.supports_masking = True
        super(SelfAttLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='kernel', shape=(input_shape[-1],), initializer='normal', trainable=True)
        super(SelfAttLayer, self).build(input_shape)
    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        #eij = K.tanh(K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1))
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        #weights = ai/K.expand_dims(K.sum(ai, axis=1), 1)
        weighted_input = x*weights.dimshuffle(0,1,'x')
        #weighted_input = x*K.expand_dims(weights,2)
        self.attention = weights
        return K.sum(weighted_input, axis=1)
        #return weights
    def get_output_shape_for(self, input_shape): return (input_shape[0], input_shape[-1])
    #def get_output_shape_for(self, input_shape): return (input_shape[0], input_shape[-2], 1)
    def compute_output_shape(self, input_shape): return self.get_output_shape_for(input_shape)
    
class MatrixMultiplication(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MatrixMultiplication, self).__init__(**kwargs)
    def build(self, input_shape):
        super(MatrixMultiplication, self).build(input_shape)
    def call(self, inputs, mask=None):
        a = inputs[0]
        b = inputs[1]
        return K.dot(a,b)
    def get_output_shape_for(self, input_shapes): return (input_shapes[0][-2], input_shapes[1][-1])
    def compute_output_shape(self, input_shapes): return self.get_output_shape_for(input_shapes)

class weightedAccCallback(keras.callbacks.Callback):
    
    def __init__(self, X1, X2, Y, overlapFeatures_fnc, refutingFeatures_fnc, polarityFeatures_fnc, handFeatures_fnc,  \
                 cosFeatures,cosFeatures_two, bleu_fnc, rouge_fnc, cider_fnc_test, X2_two_sentences, overlapFeatures_fnc_two, refutingFeatures_fnc_two, \
                 polarityFeatures_fnc_two, handFeatures_fnc_two, bleu_two_sentences, rouge_two_sentences, cider_two_test, talos_counts_test, \
                 talos_tfidfsim_test, talos_svdHeadline_test, talos_svdBody_test, talos_svdsim_test, \
                 talos_w2vHeadline_test, talos_w2vBody_test, talos_w2vsim_test, talos_sentiHeadline_test, talos_sentiBody_test ):
        self.X1 = X1 
        self.X2 = X2
        self.Y = Y
        self.overlapFeatures_fnc = overlapFeatures_fnc
        self.refutingFeatures_fnc = refutingFeatures_fnc 
        self.polarityFeatures_fnc = polarityFeatures_fnc 
        self.handFeatures_fnc = handFeatures_fnc
        self.cosFeatures = cosFeatures
        self.cosFeatures_two = cosFeatures_two
        self.bleu_fnc = bleu_fnc
        self.rouge_fnc = rouge_fnc
        self.X2_two_sentences = X2_two_sentences
        self.overlapFeatures_fnc_two = overlapFeatures_fnc_two
        self.refutingFeatures_fnc_two = refutingFeatures_fnc_two
        self.polarityFeatures_fnc_two = polarityFeatures_fnc_two
        self.handFeatures_fnc_two = handFeatures_fnc_two
        self.bleu_two_sentences = bleu_two_sentences
        self.rouge_two_sentences = rouge_two_sentences
        self.talos_counts_test = talos_counts_test 
        self.talos_tfidfsim_test = talos_tfidfsim_test 
        self.talos_svdHeadline_test = talos_svdHeadline_test
        self.talos_svdBody_test = talos_svdBody_test
        self.talos_svdsim_test = talos_svdsim_test
        self.talos_w2vHeadline_test = talos_w2vHeadline_test
        self.talos_w2vBody_test = talos_w2vBody_test
        self.talos_w2vsim_test = talos_w2vsim_test
        self.talos_sentiHeadline_test = talos_sentiHeadline_test
        self.talos_sentiBody_test = talos_sentiBody_test
        self.cider_fnc_test = cider_fnc_test
        self.cider_two_test = cider_two_test
        self.best_weighted_accuracy = [0.0, 0.0, 0.0, 0.0, 0.0]
        
    def on_epoch_end(self, epoch, logs={}):
        test_outputs = []
        test_predictions = []
        labels = ['unrelated', 'agree', 'disagree', 'discuss']
        for target in self.Y:
            test_outputs += [labels[target.tolist().index(1)]]
        aux = self.model.predict([self.X1, self.X2, self.overlapFeatures_fnc, self.refutingFeatures_fnc, self.polarityFeatures_fnc,\
        self.handFeatures_fnc, self.cosFeatures, self.cosFeatures_two, self.bleu_fnc, self.rouge_fnc, self.cider_fnc_test, \
        self.X2_two_sentences, self.overlapFeatures_fnc_two, self.refutingFeatures_fnc_two, self.polarityFeatures_fnc_two, self.handFeatures_fnc_two,\
        self.bleu_two_sentences, self.rouge_two_sentences, self.cider_two_test, self.talos_counts_test, self.talos_tfidfsim_test, \
        self.talos_svdHeadline_test,  self.talos_svdBody_test , self.talos_svdsim_test, self.talos_w2vHeadline_test , self.talos_w2vBody_test, \
        self.talos_w2vsim_test ,self.talos_sentiHeadline_test ,  self.talos_sentiBody_test], batch_size=64)
        for prediction in aux:
            test_predictions += [labels[prediction.argmax()]]
        score = report_score(test_outputs, test_predictions, matrix=False)
            
def swish(x):
    return x * K.sigmoid(x) 
