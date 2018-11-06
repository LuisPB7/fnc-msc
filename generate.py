#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import csv
import sys
import math
import random
import time
import numpy as np
import itertools
import collections
import pickle
import unicodedata
#import gensim
from nltk import tokenize
from sklearn import preprocessing
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from feature_engineering import polarity_features, refuting_features, word_overlap_features, hand_features
#from gensim.parsing.preprocessing import strip_non_alphanum
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from rouge import Rouge
import jellyfish
import string
import nltk
from nltk.translate.bleu_score import sentence_bleu
import unidecode
rouge = Rouge()
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

abbr_list = ["n't","'d","'ll","'s","'m","'ve","'re"]

max_seq_len = 50
max_seqs = 30
max_words = 200000

def closest_word(originalWord, embeddings):
    words = list(embeddings.keys())
    currentClosest = words[0]
    for word in words:
        if jellyfish.jaro_winkler(originalWord, word) > jellyfish.jaro_winkler(originalWord, currentClosest):
            currentClosest = word
    print("Closest word to " + originalWord +" is " + currentClosest)
    return embeddings[currentClosest]

def remove_parenthesis(sent):
    return ' '.join(sent.replace('(', ' ').replace(')', ' ').replace('.', '').split()).lower()

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def clean_fnc(s):
    s = unidecode.unidecode(s) # for correct tokenization
    tokens = word_tokenize(s)
    for i, tok in enumerate(tokens):
        if tok not in abbr_list:
            tokens[i] = clean(tok)
    return ' '.join(list(filter(lambda x: x != '', tokens))).lower()
    

print ("Reading training FNC data...")
X1 = [ ] # X1 vai conter lista de | headline |
X2 = [ ] # X2 vai conter lista de bodies
Y = [ ] # Y vai conter lista de one-hot vectors em relação às 4 possíveis classes
csv.field_size_limit(1000000000)
aux_dict = dict()
for row in csv.reader( open('../fnc-1/train_bodies.csv', encoding="utf8"), delimiter=',', quotechar='"' ): aux_dict[row[0]] = row[1]
with open('../fnc-1/train_stances.csv', encoding="utf8") as csvfile:
    reader = csv.reader( csvfile, delimiter=',', quotechar='"' )
    for row in reader:
        s1 = '| ' + clean_fnc(row[0]) + ' |'
        s2 = unidecode.unidecode(aux_dict[row[1]])
        if row[2] == "unrelated" : Y.append( [1,0,0,0] )
        elif row[2] == "agree" : Y.append( [0,1,0,0] )
        elif row[2] == "disagree" : Y.append( [0,0,1,0] )
        elif row[2] == "discuss" : Y.append( [0,0,0,1] )
        else: continue
        X1.append( s1 )
        X2.append( s2 )
Y = np.array( Y )

print(X1[500])

# Exatamente a mesma coisa do que em cima, mas com o dataset de teste
print ("Reading test FNC data...")
X1_test = [ ]
X2_test = [ ]
Y_test = [ ]
csv.field_size_limit(1000000000)
aux_dict = dict()
for row in csv.reader( open('../fnc-1/competition_test_bodies.csv', encoding="utf8"), delimiter=',', quotechar='"' ): aux_dict[row[0]] = row[1]
with open('../fnc-1/competition_test_stances.csv', encoding="utf8") as csvfile:
    reader = csv.reader( csvfile, delimiter=',', quotechar='"' )
    for row in reader:
        s1 = '| ' + clean_fnc(row[0]) + ' |'
        s2 = unidecode.unidecode(aux_dict[row[1]]) # body
        if row[2] == "unrelated" : Y_test.append( [1,0,0,0] )
        elif row[2] == "agree" : Y_test.append( [0,1,0,0] )
        elif row[2] == "disagree" : Y_test.append( [0,0,1,0] )
        elif row[2] == "discuss" : Y_test.append( [0,0,0,1] )
        else: continue
        X1_test.append( s1 )
        X2_test.append( s2 )
Y_test = np.array( Y_test )

###################### NLI data ########################

print ("Reading training SNLI data...")
X1_nli = [ ]
X2_nli = [ ]
Y_nli = [ ]
csv.field_size_limit(100000000)

with open('../snli_1.0/snli_1.0_train.txt') as csvfile:
    reader = csv.reader( csvfile, delimiter='\t' )
    for row in reader:
        s1 = '| ' + clean_fnc(row[1]) + ' |'
        s2 = '| ' + clean_fnc(row[2]) + ' |'
        if row[0] == "neutral" : Y_nli.append( [1,0,0] )
        elif row[0] == "entailment" : Y_nli.append( [0,1,0] )
        elif row[0] == "contradiction" : Y_nli.append( [0,0,1] )
        else: continue
        X1_nli.append( s1 )
        X2_nli.append( s2 )
        

print(X1_nli[0])
print(X1_nli[1])


print("Now reading train MultiNLI data")
with open('../multinli_1.0/multinli_1.0_train.txt', encoding='utf-8') as csvfile:
    reader = csv.reader( csvfile, delimiter='\t' )
    for row in reader:
        s1 = '| ' + clean_fnc(row[1]) + ' |'
        s2 = '| ' + clean_fnc(row[2]) + ' |'
        if row[0] == "neutral" : Y_nli.append( [1,0,0] )
        elif row[0] == "entailment" : Y_nli.append( [0,1,0] )
        elif row[0] == "contradiction" : Y_nli.append( [0,0,1] )
        else: continue
        X1_nli.append( s1 )
        X2_nli.append( s2 )

    
print ("Reading test SNLI data...")
X1_test_nli = [ ]
X2_test_nli = [ ]
Y_test_nli = [ ]
csv.field_size_limit(100000000)
with open('../snli_1.0/snli_1.0_test.txt') as csvfile:
    reader = csv.reader( csvfile, delimiter='\t' )
    for row in reader:
        s1 = '| ' + clean_fnc(row[1]) + ' |'
        s2 = '| ' + clean_fnc(row[2]) + ' |'
        if row[0] == "neutral" : Y_test_nli.append( [1,0,0] )
        elif row[0] == "entailment" : Y_test_nli.append( [0,1,0] )
        elif row[0] == "contradiction" : Y_test_nli.append( [0,0,1] )
        else: continue
        X1_test_nli.append( s1 )
        X2_test_nli.append( s2 )
        
        
print ("Reading matched MultiNLI test data...")
X1_test_matched = [ ]
X2_test_matched = [ ]
Y_test_matched = [ ]
csv.field_size_limit(100000000)
with open('../multinli_1.0/multinli_1.0_dev_matched.txt', encoding='utf-8') as csvfile:
    reader = csv.reader( csvfile, delimiter='\t' )
    for row in reader:
        s1 = '| ' + clean_fnc(row[1]) + ' |'
        s2 = '| ' + clean_fnc(row[2]) + ' |'
        if row[0] == "neutral" : Y_test_matched.append( [1,0,0] )
        elif row[0] == "entailment" : Y_test_matched.append( [0,1,0] )
        elif row[0] == "contradiction" : Y_test_matched.append( [0,0,1] )
        else: continue
        X1_test_matched.append( s1 )
        X2_test_matched.append( s2 )
        
        
print ("Reading mismatched MultiNLI test data...")
X1_test_mismatched = [ ]
X2_test_mismatched = [ ]
Y_test_mismatched = [ ]
csv.field_size_limit(100000000)
with open('../multinli_1.0/multinli_1.0_dev_mismatched.txt', encoding='utf-8') as csvfile:
    reader = csv.reader( csvfile, delimiter='\t' )
    for row in reader:
        s1 = '| ' + clean_fnc(row[1]) + ' |'
        s2 = '| ' + clean_fnc(row[2]) + ' |'
        if row[0] == "neutral" : Y_test_mismatched.append( [1,0,0] )
        elif row[0] == "entailment" : Y_test_mismatched.append( [0,1,0] )
        elif row[0] == "contradiction" : Y_test_mismatched.append( [0,0,1] )
        else: continue
        X1_test_mismatched.append( s1 )
        X2_test_mismatched.append( s2 )

########################################################

print("Creating two sentences...")
    
# Considerar as duas frases do dataset FNC #

X2_two_sentences = []
X2_test_two_sentences = []

for document in X2:
    sentences = sent_tokenize(document)
    try:
        X2_two_sentences += ['| ' + clean_fnc(sentences[0]) + ' ' + clean_fnc(sentences[1]) + ' |']
    except:
        X2_two_sentences += ['| ' + clean_fnc(sentences[0]) + ' |']

for document in X2_test:
    sentences = sent_tokenize(document)
    try:
        X2_test_two_sentences += ['| ' + clean_fnc(sentences[0]) + ' ' + clean_fnc(sentences[1]) + ' |']
    except:
        X2_test_two_sentences += ['| ' + clean_fnc(sentences[0]) + ' |']

############################################

print(X1_nli[500])
print(X2_nli[500])
print(X1[150])
print(X2[150])
print(X1_test[200])
print(X2_test[200])
print(X2_two_sentences[400])
print(X2_test_two_sentences[400])

"""
print("Generating baseline features...")

# Features com base na baseline #

overlapFeatures_fnc = np.array(word_overlap_features(X1, X2))
refutingFeatures_fnc = np.array(refuting_features(X1, X2))
polarityFeatures_fnc = np.array(polarity_features(X1, X2))
handFeatures_fnc = np.array(hand_features(X1, X2))

overlapFeatures_fnc_test = np.array(word_overlap_features(X1_test, X2_test))
refutingFeatures_fnc_test = np.array(refuting_features(X1_test, X2_test))
polarityFeatures_fnc_test = np.array(polarity_features(X1_test, X2_test))
handFeatures_fnc_test = np.array(hand_features(X1_test, X2_test))

overlapFeatures_nli = np.array(word_overlap_features(X1_nli, X2_nli))
refutingFeatures_nli = np.array(refuting_features(X1_nli, X2_nli))
polarityFeatures_nli = np.array(polarity_features(X1_nli, X2_nli))
handFeatures_nli = np.array(hand_features(X1_nli, X2_nli))

overlapFeatures_nli_test = np.array(word_overlap_features(X1_test_nli, X2_test_nli))
refutingFeatures_nli_test = np.array(refuting_features(X1_test_nli, X2_test_nli))
polarityFeatures_nli_test = np.array(polarity_features(X1_test_nli, X2_test_nli))
handFeatures_nli_test = np.array(hand_features(X1_test_nli, X2_test_nli))

overlapFeatures_matched_test = np.array(word_overlap_features(X1_test_matched, X2_test_matched))
refutingFeatures_matched_test = np.array(refuting_features(X1_test_matched, X2_test_matched))
polarityFeatures_matched_test = np.array(polarity_features(X1_test_matched, X2_test_matched))
handFeatures_matched_test = np.array(hand_features(X1_test_matched, X2_test_matched))

overlapFeatures_mismatched_test = np.array(word_overlap_features(X1_test_mismatched, X2_test_mismatched))
refutingFeatures_mismatched_test = np.array(refuting_features(X1_test_mismatched, X2_test_mismatched))
polarityFeatures_mismatched_test = np.array(polarity_features(X1_test_mismatched, X2_test_mismatched))
handFeatures_mismatched_test = np.array(hand_features(X1_test_mismatched, X2_test_mismatched))

overlapFeatures_fnc_two = np.array(word_overlap_features(X1, X2_two_sentences))
refutingFeatures_fnc_two = np.array(refuting_features(X1, X2_two_sentences))
polarityFeatures_fnc_two = np.array(polarity_features(X1, X2_two_sentences))
handFeatures_fnc_two = np.array(hand_features(X1, X2_two_sentences))

overlapFeatures_fnc_two_test = np.array(word_overlap_features(X1_test, X2_test_two_sentences))
refutingFeatures_fnc_two_test = np.array(refuting_features(X1_test, X2_test_two_sentences))
polarityFeatures_fnc_two_test = np.array(polarity_features(X1_test, X2_test_two_sentences))
handFeatures_fnc_two_test = np.array(hand_features(X1_test, X2_test_two_sentences))

print("Generating MT features...")

# Features com base no BLEU, ROUGE, METEOR, CIDER e SPICE #

bleu_nli = []
for i in range(len(X1_nli)):
    bleu_nli += [ sentence_bleu([ word_tokenize(X1_nli[i]) ], word_tokenize(X2_nli[i]) ) ]
bleu_nli = np.array(bleu_nli)

bleu_nli_test = []
for i in range(len(X1_test_nli)):
    bleu_nli_test += [ sentence_bleu([ word_tokenize(X1_test_nli[i]) ], word_tokenize(X2_test_nli[i]) ) ]
bleu_nli_test = np.array(bleu_nli_test)

bleu_matched = []
for i in range(len(X1_test_matched)):
    bleu_matched += [ sentence_bleu([ word_tokenize(X1_test_matched[i]) ], word_tokenize(X2_test_matched[i]) ) ]
bleu_matched = np.array(bleu_matched)

bleu_mismatched = []
for i in range(len(X1_test_mismatched)):
    bleu_mismatched += [ sentence_bleu([ word_tokenize(X1_test_mismatched[i]) ], word_tokenize(X2_test_mismatched[i]) ) ]
bleu_mismatched = np.array(bleu_mismatched)

rouge_nli = []
for i in range(len(X1_nli)):
    rouge_values = []
    scores = rouge.get_scores(X1_nli[i], X2_nli[i])
    rouge_values += [scores[0]['rouge-1']['f']]
    rouge_values += [scores[0]['rouge-2']['f']]
    rouge_values += [scores[0]['rouge-l']['f']]
    rouge_nli += [rouge_values]
rouge_nli = np.array(rouge_nli)
    
rouge_nli_test = []
for i in range(len(X1_test_nli)):
    rouge_values = []
    scores = rouge.get_scores(X1_test_nli[i], X2_test_nli[i])
    rouge_values += [scores[0]['rouge-1']['f']]
    rouge_values += [scores[0]['rouge-2']['f']]
    rouge_values += [scores[0]['rouge-l']['f']]
    rouge_nli_test += [rouge_values]
rouge_nli_test = np.array(rouge_nli_test)
    
rouge_matched = []
for i in range(len(X1_test_matched)):
    rouge_values = []
    scores = rouge.get_scores(X1_test_matched[i], X2_test_matched[i])
    rouge_values += [scores[0]['rouge-1']['f']]
    rouge_values += [scores[0]['rouge-2']['f']]
    rouge_values += [scores[0]['rouge-l']['f']]
    rouge_matched += [rouge_values]
rouge_matched = np.array(rouge_matched)

rouge_mismatched = []
for i in range(len(X1_test_mismatched)):
    rouge_values = []
    scores = rouge.get_scores(X1_test_mismatched[i], X2_test_mismatched[i])
    rouge_values += [scores[0]['rouge-1']['f']]
    rouge_values += [scores[0]['rouge-2']['f']]
    rouge_values += [scores[0]['rouge-l']['f']]
    rouge_mismatched += [rouge_values]
rouge_mismatched = np.array(rouge_mismatched)

bleu_fnc = []
for i in range(len(X1)):
    split_doc = sent_tokenize(X2[i])
    for j in range(len(split_doc)):
        split_doc[j] = word_tokenize(clean_fnc(split_doc[j]))
    bleu_fnc += [ sentence_bleu(split_doc, word_tokenize(X1[i])) ]
bleu_fnc = np.array(bleu_fnc)

bleu_fnc_test = []
for i in range(len(X1_test)):
    split_doc = sent_tokenize(X2_test[i])
    for j in range(len(split_doc)):
        split_doc[j] = word_tokenize(clean_fnc(split_doc[j]))
    bleu_fnc_test += [ sentence_bleu(split_doc, word_tokenize(X1_test[i]) ) ]
bleu_fnc_test = np.array(bleu_fnc_test)

bleu_two_sentences = []
for i in range(len(X1)):
    bleu_two_sentences += [ sentence_bleu(word_tokenize(clean_fnc(X2_two_sentences[i])), \
                                          word_tokenize(X1[i]) ) ]
bleu_two_sentences = np.array(bleu_two_sentences)

bleu_two_sentences_test = []
for i in range(len(X1_test)):
    bleu_two_sentences_test += [ sentence_bleu(word_tokenize(clean_fnc(X2_test_two_sentences[i])), \
                                          word_tokenize(X1_test[i]) ) ]
bleu_two_sentences_test = np.array(bleu_two_sentences_test)

rouge_fnc = []
fails = 0
for i in range(len(X1)):
    rouge_values = []
    try:
        scores = rouge.get_scores(clean(X2[i]), clean(X1[i]))
        rouge_values += [scores[0]['rouge-1']['f']]
        rouge_values += [scores[0]['rouge-2']['f']]
        rouge_values += [scores[0]['rouge-l']['f']]
    except:
        rouge_values = [0,0,0]
        fails += 1
    rouge_fnc += [rouge_values]
print("ROUGE FNC: {} fails".format(fails))
rouge_fnc = np.array(rouge_fnc)

rouge_fnc_test = []
fails = 0
for i in range(len(X1_test)):
    rouge_values = []
    try:
        scores = rouge.get_scores(clean(X2_test[i]), clean(X1_test[i]))
        rouge_values += [scores[0]['rouge-1']['f']]
        rouge_values += [scores[0]['rouge-2']['f']]
        rouge_values += [scores[0]['rouge-l']['f']]
    except:
        fails += 1
        rouge_values = [0,0,0]
    rouge_fnc_test += [rouge_values]
print("ROUGE FNC TEST: {} fails".format(fails))
rouge_fnc_test = np.array(rouge_fnc_test)

rouge_two_sentences = []
for i in range(len(X1)):
    rouge_values = []
    scores = rouge.get_scores(clean(X2_two_sentences[i]), clean(X1[i]))
    rouge_values += [scores[0]['rouge-1']['f']]
    rouge_values += [scores[0]['rouge-2']['f']]
    rouge_values += [scores[0]['rouge-l']['f']]
    rouge_two_sentences += [rouge_values]
rouge_two_sentences = np.array(rouge_two_sentences)

rouge_two_sentences_test = []
for i in range(len(X1_test)):
    rouge_values = []
    scores = rouge.get_scores(clean(X2_test_two_sentences[i]), clean(X1_test[i]))
    rouge_values += [scores[0]['rouge-1']['f']]
    rouge_values += [scores[0]['rouge-2']['f']]
    rouge_values += [scores[0]['rouge-l']['f']]
    rouge_two_sentences_test += [rouge_values]
rouge_two_sentences_test = np.array(rouge_two_sentences_test)


#####################################


# 48
with open('features.pkl', 'wb') as output:
    pickle.dump(overlapFeatures_fnc, output, 2)
    pickle.dump(refutingFeatures_fnc, output, 2)
    pickle.dump(polarityFeatures_fnc, output, 2)
    pickle.dump(handFeatures_fnc, output, 2)
    pickle.dump(overlapFeatures_fnc_test, output, 2)
    pickle.dump(refutingFeatures_fnc_test, output, 2)
    pickle.dump(polarityFeatures_fnc_test, output, 2)
    pickle.dump(handFeatures_fnc_test, output, 2)
    pickle.dump(overlapFeatures_nli, output, 2)
    pickle.dump(refutingFeatures_nli, output, 2)
    pickle.dump(polarityFeatures_nli, output, 2)
    pickle.dump(handFeatures_nli, output, 2)
    pickle.dump(overlapFeatures_nli_test, output, 2)
    pickle.dump(refutingFeatures_nli_test, output, 2)
    pickle.dump(polarityFeatures_nli_test, output, 2)
    pickle.dump(handFeatures_nli_test, output, 2)
    pickle.dump(overlapFeatures_matched_test, output, 2)
    pickle.dump(refutingFeatures_matched_test, output, 2)
    pickle.dump(polarityFeatures_matched_test, output, 2)
    pickle.dump(handFeatures_matched_test, output, 2)
    pickle.dump(overlapFeatures_mismatched_test, output, 2)
    pickle.dump(refutingFeatures_mismatched_test, output, 2)
    pickle.dump(polarityFeatures_mismatched_test, output, 2)
    pickle.dump(handFeatures_mismatched_test, output, 2)
    pickle.dump(overlapFeatures_fnc_two, output, 2)
    pickle.dump(refutingFeatures_fnc_two, output, 2)
    pickle.dump(polarityFeatures_fnc_two, output, 2)
    pickle.dump(handFeatures_fnc_two, output, 2)
    pickle.dump(overlapFeatures_fnc_two_test, output, 2)
    pickle.dump(refutingFeatures_fnc_two_test, output, 2)
    pickle.dump(polarityFeatures_fnc_two_test, output, 2)
    pickle.dump(handFeatures_fnc_two_test, output, 2)
    pickle.dump(bleu_nli, output, 2)
    pickle.dump(bleu_nli_test, output, 2)
    pickle.dump(bleu_matched, output, 2)
    pickle.dump(bleu_mismatched, output, 2)
    pickle.dump(rouge_nli, output, 2)
    pickle.dump(rouge_nli_test, output, 2)
    pickle.dump(rouge_matched, output, 2)
    pickle.dump(rouge_mismatched, output, 2)
    pickle.dump(bleu_fnc, output, 2)
    pickle.dump(bleu_fnc_test, output, 2)
    pickle.dump(bleu_two_sentences, output, 2)
    pickle.dump(bleu_two_sentences_test, output, 2)
    pickle.dump(rouge_fnc, output, 2)
    pickle.dump(rouge_fnc_test, output, 2)
    pickle.dump(rouge_two_sentences, output, 2)
    pickle.dump(rouge_two_sentences_test, output, 2)
"""

###################

print ("Reading word embeddings...")
embeddings = dict( ) # Embeddings são representadas por um dicionário com pares palavra: vetor embedding
f = open('../glove.42B.300d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings[word] = coefs
f.close()
embeddings_dim = len( embeddings['the'] ) # embeddings_dim deverá ser 300?!??!?!?!!

print ("Generating token-based representations for contexts...")
tokenizer = Tokenizer( nb_words=max_words , lower=True , split=" ") # max_words = 100.000, lower=True significa meter o texto em lowercase; ou seja, só vamos ter 100.000 palavras diferentes?

X2_clean = list(map(lambda x:clean_fnc(x), X2))
X2_test_clean = list(map(lambda x: clean_fnc(x), X2_test))

tokenizer.fit_on_texts( X1 + X2_clean + X1_test + X2_test_clean + X1_nli + X2_nli + X1_test_nli + X2_test_nli + X1_test_matched + X2_test_matched + X1_test_mismatched + X2_test_mismatched + X2_two_sentences + X2_test_two_sentences ) # X1 + X2 = juntar vetor dos | headlines | com vetor dos bodies de treino
X1 = sequence.pad_sequences( tokenizer.texts_to_sequences( X1 ) , maxlen=max_seq_len ) # Converter headlines para números e applicar zero padding, i.e. forçar todos os headlines a terem 30 tokens
X1_test = sequence.pad_sequences( tokenizer.texts_to_sequences( X1_test ) , maxlen=max_seq_len ) # Mesma coisa para headlines de teste

X1_nli = sequence.pad_sequences( tokenizer.texts_to_sequences( X1_nli ) , maxlen=max_seq_len )
X2_nli = sequence.pad_sequences( tokenizer.texts_to_sequences( X2_nli ) , maxlen=max_seq_len )
X1_test_nli = sequence.pad_sequences( tokenizer.texts_to_sequences( X1_test_nli ) , maxlen=max_seq_len )
X2_test_nli = sequence.pad_sequences( tokenizer.texts_to_sequences( X2_test_nli ) , maxlen=max_seq_len )
X1_test_matched = sequence.pad_sequences( tokenizer.texts_to_sequences( X1_test_matched ) , maxlen=max_seq_len )
X2_test_matched = sequence.pad_sequences( tokenizer.texts_to_sequences( X2_test_matched ) , maxlen=max_seq_len )
X1_test_mismatched = sequence.pad_sequences( tokenizer.texts_to_sequences( X1_test_mismatched ) , maxlen=max_seq_len )
X2_test_mismatched = sequence.pad_sequences( tokenizer.texts_to_sequences( X2_test_mismatched ) , maxlen=max_seq_len )
X2_two_sentences = sequence.pad_sequences( tokenizer.texts_to_sequences( X2_two_sentences ) , maxlen=max_seq_len )
X2_test_two_sentences = sequence.pad_sequences( tokenizer.texts_to_sequences( X2_test_two_sentences ) , maxlen=max_seq_len )

data_aux = np.zeros( ( len(X2) , max_seqs , max_seq_len ) ) # len(X2) = numero total de bodies do dataset, max_seq_len = 30, max_seqs = 15
for i, sentences in enumerate(X2):
    sentences = sent_tokenize( sentences ) # sentences é agora uma lista de frases em vez de um único body
    sentences = list(map(lambda x: '| ' + clean_fnc(x) + ' |', sentences))
    aux = [ ]
    for j, sent in enumerate(sentences): # Só consideramos as max_seqs primeiras frases do body!
        if j < max_seqs: data_aux[i,j] = sequence.pad_sequences( tokenizer.texts_to_sequences( [ sent ] ) , maxlen=max_seq_len )[0] # data_aux vai ficar a conter para cada i (body), para cada j(frase desse body), um vetor dessa frase
X1 = np.asarray( X1 )
X2 = np.asarray( data_aux ) # Os nossos bodies de treino são agora a tal matriz com cada body, em que cada body tem cada frase, e cada frase é um vetor de números (com padding)
data_aux = np.zeros( ( len(X2_test) , max_seqs , max_seq_len ) ) # Fazer exatamente a mesma coisa para os bodies de treino
for i, sentences in enumerate(X2_test):
    sentences = sent_tokenize( sentences )
    sentences = list(map(lambda x: '| ' + clean_fnc(x) + ' |', sentences))
    aux = [ ]
    for j, sent in enumerate(sentences):
        if j < max_seqs: data_aux[i,j] = sequence.pad_sequences( tokenizer.texts_to_sequences( [ sent ] ) , maxlen=max_seq_len )[0]
X1_test = np.asarray( X1_test )
X2_test = np.asarray( data_aux )
embedding_weights = np.zeros( ( max_words , embeddings_dim ) )
total = 0
success = 0
fail = 0
for word,index in tokenizer.word_index.items():
    print(index)
    if index < max_words: # Só vamos ter max_words palavras
        total += 1
        try: 
            embedding_weights[index,:] = embeddings[word] # Em vez de embeddings estava embeddings_en;
            success += 1
        except:
            print("Failed word: {}".format(word))
            embedding_weights[index,:] = closest_word(word, embeddings)
            fail += 1
            
print("Total of " + str(total) + " words, and " + str(fail) + " were not known, while " + str(success) +" were. That's " + str(fail*100/success) + " %")
embedding_weights = np.array( embedding_weights ) # embedding_weights vão ter os embeddings das 200.000 palavras em 100/300 dimensões

with open('variables.pkl', 'wb') as output:
    pickle.dump(embedding_weights, output, 2)
    pickle.dump(X1, output, 2)
    pickle.dump(X2, output, 2)
    pickle.dump(Y, output, 2)
    pickle.dump(X1_test, output, 2)
    pickle.dump(X2_test, output, 2)
    pickle.dump(Y_test, output, 2)
    pickle.dump(X1_nli, output, 2)
    pickle.dump(X2_nli, output, 2)
    pickle.dump(Y_nli, output, 2)
    pickle.dump(X1_test_nli, output, 2)
    pickle.dump(X2_test_nli, output, 2)
    pickle.dump(Y_test_nli, output, 2)
    pickle.dump(X1_test_matched, output, 2)
    pickle.dump(X2_test_matched, output, 2)
    pickle.dump(Y_test_matched, output, 2)
    pickle.dump(X1_test_mismatched, output, 2)
    pickle.dump(X2_test_mismatched, output, 2)
    pickle.dump(Y_test_mismatched, output, 2)
    pickle.dump(X2_two_sentences, output, 2)
    pickle.dump(X2_test_two_sentences, output, 2)
    pickle.dump(tokenizer, output, 2)

