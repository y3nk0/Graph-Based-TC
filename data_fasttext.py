
from sklearn import metrics

import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn import svm
from sklearn.metrics import accuracy_score
from nltk.stem import *
from nltk.corpus import stopwords
import time
import re
import os.path
import math
import codecs

# path = 'webkb/'
# other_path = 'WEBKB'

# path = 'amazon/'
# other_path = 'amazon'

# path = 'subject/'
# other_path = 'subject'

# path = 'imdb/'
# other_path = 'IMDB'

# path = '20newsgroup/'
# other_path = '20NG'

path = 'reuters/'
other_path = 'reuters'

words_frequency = {}

clean_train_documents = []
y_train = []

y_test = []
clean_test_documents = []

## Open the file with read only permission

f = codecs.open(path+'data/my_'+other_path+'_train.txt', "r", encoding="utf-8")
train = [x.strip('\n') for x in f.readlines()]
f.close()

num_documents = len(train)

for i in range( 0, num_documents ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    line = train[i].split('\t')

    if line[1].split(" ")>1:
        y_train.append(line[0])

        for n, w in enumerate( line[1].split(' ') ):
            if w not in words_frequency:
                words_frequency[w] = 1
            else:
                words_frequency[w] = words_frequency[w] + 1

        clean_train_documents.append( line[1] )


## Open the file with read only permit
f = codecs.open(path+'data/my_'+other_path+'_test.txt', "r", encoding="utf-8")
test = [x.strip('\n') for x in f.readlines()]
f.close()

num_test_documents = len(test)

for i in range( 0, num_test_documents ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    line = test[i].split('\t')

    if line[1].split(" ")>1:
        y_test.append(line[0])
        clean_test_documents.append( line[1] )


f = codecs.open(path+'data/my_'+other_path+'_train_FASTTEXT.txt', "w", encoding="utf-8")
for i, doc in enumerate(clean_train_documents):
    s = doc.split(" ")
    if len(set(s))>1:
        f.write("__label__"+str(y_train[i])+" "+doc+"\n")

f.close()

f = codecs.open(path+'data/my_'+other_path+'_test_FASTTEXT.txt', "w", encoding="utf-8")
for i, doc in enumerate(clean_test_documents):
    s = doc.split(" ")
    if len(set(s))>1:
        f.write("__label__"+str(y_test[i])+" "+doc+"\n")

f.close()
