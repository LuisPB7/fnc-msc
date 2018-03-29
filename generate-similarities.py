# Initialize logging.
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import csv
import unicodedata
import gensim
import numpy as np
from gensim.parsing.preprocessing import strip_non_alphanum
    
print ("Reading training FNC data...")
X1 = [ ] # X1 vai conter lista de | headline |
X2 = [ ] # X2 vai conter lista de bodies
Y = [ ] # Y vai conter lista de one-hot vectors em relacao as 4 possiveis classes
csv.field_size_limit(1000000000)
aux_dict = dict()
for row in csv.reader( open('fnc-1/train_bodies.csv', encoding="utf8"), delimiter=',', quotechar='"' ): aux_dict[row[0]] = row[1]
with open('fnc-1/train_stances.csv', encoding="utf8") as csvfile:
    reader = csv.reader( csvfile, delimiter=',', quotechar='"' )
    for row in reader:
        s1 = unicodedata.normalize('NFKD', (gensim.utils.any2unicode(u'| ' +  row[0] + u' |'))) 
        s2 = unicodedata.normalize('NFKD', ( gensim.utils.any2unicode( aux_dict[row[1]] )))
        if row[2] == "unrelated" : Y.append( [1,0,0,0] )
        elif row[2] == "agree" : Y.append( [0,1,0,0] )
        elif row[2] == "disagree" : Y.append( [0,0,1,0] )
        elif row[2] == "discuss" : Y.append( [0,0,0,1] )
        else: continue
        X1.append( s1 )
        X2.append( s2 )
Y = np.array( Y )

# Exatamente a mesma coisa do que em cima, mas com o dataset de teste
print ("Reading test FNC data...")
X1_test = [ ]
X2_test = [ ]
Y_test = [ ]
csv.field_size_limit(1000000000)
aux_dict = dict()
for row in csv.reader( open('fnc-1/competition_test_bodies.csv', encoding="utf8"), delimiter=',', quotechar='"' ): aux_dict[row[0]] = row[1]
with open('fnc-1/competition_test_stances.csv', encoding="utf8") as csvfile:
    reader = csv.reader( csvfile, delimiter=',', quotechar='"' )
    for row in reader:
        s1 = unicodedata.normalize('NFKD', ( gensim.utils.any2unicode(u'| '+  row[0] + u' |'))) # | headline |
        s2 = unicodedata.normalize('NFKD', ( gensim.utils.any2unicode( aux_dict[row[1]]) ) ) # body
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

with open('snli_1.0/snli_1.0_train.txt') as csvfile:
    reader = csv.reader( csvfile, delimiter='\t' )
    for row in reader:
        s1 = unicodedata.normalize('NFKD', ( u'| ' + gensim.utils.any2unicode( strip_non_alphanum(row[5]) ) + u' |') )
        s2 = unicodedata.normalize('NFKD', ( u'| ' + gensim.utils.any2unicode( strip_non_alphanum(row[6]) ) + u' |') )
        if row[0] == "neutral" : Y_nli.append( [1,0,0] )
        elif row[0] == "entailment" : Y_nli.append( [0,1,0] )
        elif row[0] == "contradiction" : Y_nli.append( [0,0,1] )
        else: continue
        X1_nli.append( s1 )
        X2_nli.append( s2 )

print("Now reading train MultiNLI data")
with open('multinli_1.0/multinli_1.0_train.txt', encoding='utf-8') as csvfile:
    reader = csv.reader( csvfile, delimiter='\t' )
    for row in reader:
        s1 = unicodedata.normalize('NFKD', ( u'| ' + gensim.utils.any2unicode( strip_non_alphanum(row[5]) ) + u' |') )
        s2 = unicodedata.normalize('NFKD', ( u'| ' + gensim.utils.any2unicode( strip_non_alphanum(row[6]) ) + u' |') )
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
with open('snli_1.0/snli_1.0_test.txt') as csvfile:
    reader = csv.reader( csvfile, delimiter='\t' )
    for row in reader:
        s1 = unicodedata.normalize('NFKD', ( u'| ' + gensim.utils.any2unicode( strip_non_alphanum(row[5]) ) + u' |') )
        s2 = unicodedata.normalize('NFKD', ( u'| ' + gensim.utils.any2unicode( strip_non_alphanum(row[6]) ) + u' |') )
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
with open('multinli_1.0/multinli_1.0_dev_matched.txt', encoding='utf-8') as csvfile:
    reader = csv.reader( csvfile, delimiter='\t' )
    for row in reader:
        s1 = unicodedata.normalize('NFKD', ( u'| ' + gensim.utils.any2unicode( strip_non_alphanum(row[5]) ) + u' |') )
        s2 = unicodedata.normalize('NFKD', ( u'| ' + gensim.utils.any2unicode( strip_non_alphanum(row[6]) ) + u' |') )
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
with open('multinli_1.0/multinli_1.0_dev_mismatched.txt', encoding='utf-8') as csvfile:
    reader = csv.reader( csvfile, delimiter='\t' )
    for row in reader:
        s1 = unicodedata.normalize('NFKD', ( u'| ' + gensim.utils.any2unicode( strip_non_alphanum(row[5]) ) + u' |') )
        s2 = unicodedata.normalize('NFKD', ( u'| ' + gensim.utils.any2unicode( strip_non_alphanum(row[6]) ) + u' |') )
        if row[0] == "neutral" : Y_test_mismatched.append( [1,0,0] )
        elif row[0] == "entailment" : Y_test_mismatched.append( [0,1,0] )
        elif row[0] == "contradiction" : Y_test_mismatched.append( [0,0,1] )
        else: continue
        X1_test_mismatched.append( s1 )
        X2_test_mismatched.append( s2 )

########################################################
        
# Tornar X1, X2, X1_test e X2_test sem non alpha chars #

for i in range(len(X1)):
    X1[i] = u'| ' + strip_non_alphanum(X1[i]) + u' |'
    
for i in range(len(X1_test)):
    X1_test[i] = u'| ' + strip_non_alphanum(X1_test[i]) + u' |'
    
for i in range(len(X2)):
    current_doc = X2[i]
    sentences = current_doc.split('\n')
    new_doc = ''
    for l in range(len(sentences)):
        sentences[l] = strip_non_alphanum(sentences[l])
    for l in range(len(sentences)):
        if sentences[l] != '':
            new_doc += sentences[l] + '.\n'
    X2[i] = new_doc

for i in range(len(X2_test)):
    current_doc = X2_test[i]
    sentences = current_doc.split('\n')
    new_doc = ''
    for l in range(len(sentences)):
        sentences[l] = strip_non_alphanum(sentences[l])
    for l in range(len(sentences)):
        if sentences[l] != '':
            new_doc += sentences[l]  + '.\n'
    X2_test[i] = new_doc
    

###############################################
    
# Considerar as duas frases do dataset FNC #

X2_two_sentences = []
X2_test_two_sentences = []

for document in X2:
    sentences = document.split('\n')
    X2_two_sentences += ['| ' + strip_non_alphanum(sentences[0]) + ' ' + strip_non_alphanum(sentences[1]) + ' |']
for document in X2_test:
    sentences = document.split('\n')
    X2_test_two_sentences += ['| ' + strip_non_alphanum(sentences[0]) + ' ' + strip_non_alphanum(sentences[1]) + ' |']

############################################
    
"""X1 = np.array(X1)
X2 = np.array(X2)
X1_test = np.array(X1_test)
X2_test = np.array(X2_test)
X1_nli = np.array(X1_nli)
X2_nli = np.array(X2_nli)
X1_test_nli = np.array(X1_test_nli)
X2_test_nli = np.array(X2_test_nli)
X2_two_sentences = np.array(X2_two_sentences)
X2_test_two_sentences = np.array(X2_test_two_sentences)
X1_test_matched = np.array(X1_test_matched)
X2_test_matched = np.array(X2_test_matched)
X1_test_mismatched = np.array(X1_test_mismatched)
X2_test_mismatched = np.array(X2_test_mismatched)"""

full_corpus = X1+X2+X1_test+X2_test+X1_nli+X2_nli+X1_test_nli+X2_test_nli+X2_two_sentences+X2_test_two_sentences+X1_test_matched+X2_test_matched+X1_test_mismatched+X2_test_mismatched

new_X1 = []
new_X2 = []
new_X1_test = []
new_X2_test = []
new_X1_nli = []
new_X2_nli = []
new_X1_test_nli = []
new_X2_test_nli = []
new_X2_two_sentences = []
new_X2_test_two_sentences = []
new_X1_test_matched = []
new_X2_test_matched = []
new_X1_test_mismatched = []
new_X2_test_mismatched = []


for element in X1:
    new_X1 += [element.split()]
    
for element in X2:
    new_X2 += [element.split()]
    
for element in X1_test:
    new_X1_test += [element.split()]
    
for element in X2_test:
    new_X2_test += [element.split()]
    
for element in X1_nli:
    new_X1_nli += [element.split()]
    
for element in X2_nli:
    new_X2_nli += [element.split()]
    
for element in X1_test_nli:
    new_X1_test_nli += [element.split()]
    
for element in X2_test_nli:
    new_X2_test_nli += [element.split()]
    
for element in X2_two_sentences:
    new_X2_two_sentences += [element.split()]
    
for element in X2_test_two_sentences:
    new_X2_test_two_sentences += [element.split()]
    
for element in X1_test_matched:
    new_X1_test_matched += [element.split()]
    
for element in X2_test_matched:
    new_X2_test_matched += [element.split()]
    
for element in X1_test_mismatched:
    new_X1_test_mismatched += [element.split()]
    
for element in X2_test_mismatched:
    new_X2_test_mismatched += [element.split()]
      
corpus = new_X1+new_X2+new_X1_test+new_X2_test+new_X1_nli+new_X2_nli+new_X1_test_nli+new_X2_test_nli+new_X2_two_sentences+new_X2_test_two_sentences+new_X1_test_matched+new_X2_test_matched+new_X1_test_mismatched+new_X2_test_mismatched


from gensim import corpora
import gensim.downloader as api

w2v_model = api.load("glove-wiki-gigaword-300")

dictio = corpora.Dictionary(corpus)
matrix = w2v_model.similarity_matrix(dictio)

X1 = [dictio.doc2bow(document) for document in new_X1]
X2 = [dictio.doc2bow(document) for document in new_X2]
X1_test = [dictio.doc2bow(document) for document in new_X1_test]
X2_test = [dictio.doc2bow(document) for document in new_X2_test]
X1_nli = [dictio.doc2bow(document) for document in new_X1_nli]
X2_nli = [dictio.doc2bow(document) for document in new_X2_nli]
X1_test_nli = [dictio.doc2bow(document) for document in new_X1_test_nli]
X2_test_nli = [dictio.doc2bow(document) for document in new_X2_test_nli]
X2_two_sentences = [dictio.doc2bow(document) for document in new_X2_two_sentences]
X2_test_two_sentences = [dictio.doc2bow(document) for document in new_X2_test_two_sentences]
X1_test_matched = [dictio.doc2bow(document) for document in new_X1_test_matched]
X2_test_matched = [dictio.doc2bow(document) for document in new_X2_test_matched]
X1_test_mismatched = [dictio.doc2bow(document) for document in new_X1_test_mismatched]
X2_test_mismatched = [dictio.doc2bow(document) for document in new_X2_test_mismatched]


print(X2[3])


from gensim.matutils import softcossim

cosFeatures = []
for i in range(len(X1)):
    cosFeatures += [[softcossim(X1[i], X2[i], matrix), softcossim(X1[i], X2_two_sentences[i], matrix)]]
    
cosFeatures_test = []
for i in range(len(X1_test)):
    cosFeatures_test += [[softcossim(X1_test[i], X2_test[i], matrix), softcossim(X1_test[i], X2_test_two_sentences[i], matrix)]]
    
cosFeatures_nli = []
for i in range(len(X1_nli)):
    cosFeatures_nli += [[softcossim(X1_nli[i], X2_nli[i], matrix)]]
    
cosFeatures_nli_test = []
for i in range(len(X1_test_nli)):
    cosFeatures_nli_test += [[softcossim(X1_test_nli[i], X2_test_nli[i], matrix)]]
    
cosFeatures_matched = []
for i in range(len(X1_test_matched)):
    cosFeatures_matched += [[softcossim(X1_test_matched[i], X2_test_matched[i], matrix)]]
    
cosFeatures_mismatched = []
for i in range(len(X1_test_mismatched)):
    cosFeatures_mismatched += [[softcossim(X1_test_mismatched[i], X2_test_mismatched[i], matrix)]]
    
import pickle

with open('similarity.pkl', 'wb') as output:
    pickle.dump(cosFeatures, output, 2)
    pickle.dump(cosFeatures_test, output, 2)
    pickle.dump(cosFeatures_nli, output, 2)
    pickle.dump(cosFeatures_nli_test, output, 2)
    pickle.dump(cosFeatures_matched, output, 2)
    pickle.dump(cosFeatures_mismatched, output, 2)