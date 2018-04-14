import networkx as nx
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
import string
from sys import maxint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
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
import sys
import scipy
sys.path.append('../../code')
from MyGraph import *
from library import *
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV


# Now, we can call these functions to create average vectors for each paragraph. The following operations will take a few minutes:

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

b = 0.20
unique_words = []
bigrams = []
trigrams = []
terms = []

ngrams_par = 1
idf_bool = False
freq_par = "not-binary"

bag_of_words = "MY TF-IDF"
classifier_par = "svm"


words_frequency = {}

y_train = []
clean_train_documents = []
categories = ['course','faculty','project','student']

clean_test_documents = []
y_test = []

if os.path.exists("data/my_WEBKB_train.txt"):
    ## Open the file with read only permi

    f = codecs.open('data/my_WEBKB_train_VOCAB.txt', "r", encoding="utf-8")
    unique_words = [x.strip('\n') for x in f.readlines()]
    f.close()

    f = codecs.open('data/my_WEBKB_train.txt', "r", encoding="utf-8")
    train = [x.strip('\n') for x in f.readlines()]
    f.close()

    num_documents = len(train)

    for i in xrange( 0, num_documents ):
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

    # unique_words = list(words_frequency.keys())
    unique_words = [k for (k,v) in words_frequency.items() if v>1]

    ## Open the file with read only permit
    f = codecs.open('data/my_WEBKB_test.txt', "r", encoding="utf-8")
    test = [x.strip('\n') for x in f.readlines()]
    f.close()

    num_test_documents = len(test)

    for i in xrange( 0, num_test_documents ):
        # Call our function for each one, and add the result to the list of
        # clean reviews
        line = test[i].split('\t')

        if line[1].split(" ")>1:
            y_test.append(line[0])
            clean_test_documents.append( line[1] )

else:
    train_data = []
    y_train_all = []
    paths=['cornell','misc','texas','washington','wisconsin']
    for path in paths:
        print path
        path_pos = 'data/course/'+path+'/*'

        files=glob.glob(path_pos)
        for file in files:
            f=open(file, 'r')
            train_data.append( f.read() )
            y_train_all.append(0)
            f.close()

    	path_neg = 'data/faculty/'+path+'/*'

    	files=glob.glob(path_neg)
    	for file in files:
    		f=open(file, 'r')
    		train_data.append( f.read() )
    		y_train_all.append(1)
    		f.close()

        path_pos = 'data/project/'+path+'/*'

        files=glob.glob(path_pos)
        for file in files:
            f=open(file, 'r')
            train_data.append( f.read() )
            y_train_all.append(2)
            f.close()

        path_neg = 'data/student/'+path+'/*'

        files=glob.glob(path_neg)
        for file in files:
            f=open(file, 'r')
            train_data.append( f.read() )
            y_train_all.append(3)
            f.close()


    train_data, test_data, y_train_all, y_test_all = train_test_split(train_data,y_train_all,test_size=0.33,random_state=42)

    train_data_NEW = []
    for t in train_data:
        train_data_NEW.append(remove_tags(t))
    unique_words = findVocab(train_data_NEW)
    train_data = parseXmlStopStemRem(train_data_NEW,unique_words,bigrams,trigrams,True)

    test_data_NEW = []
    for t in test_data:
        test_data_NEW.append(remove_tags(t))
    test_data = parseXmlStopStemRem(test_data_NEW,unique_words,bigrams,trigrams,False)

    f = codecs.open('data/my_WEBKB_train_VOCAB.txt', "w", encoding="utf-8")
    for item in unique_words:
        f.write("%s\n" % item)
    f.close()

    f = codecs.open('data/my_WEBKB_train.txt', "w", encoding="utf-8")
    for i, doc in enumerate(train_data):
        s = doc.split(" ")
        if len(set(s))>1:
            clean_train_documents.append(doc)
            f.write(str(categories[y_train_all[i]])+"\t"+doc+"\n")
            y_train.append(y_train_all[i])

    f.close()

    f = codecs.open('data/my_WEBKB_test.txt', "w", encoding="utf-8")
    for i, doc in enumerate(test_data):
        s = doc.split(" ")
        if len(set(s))>1:
            f.write(str(categories[y_test_all[i]])+"\t"+doc+"\n")
            clean_test_documents.append(doc)
            y_test.append(y_test_all[i])

    f.close()

num_documents = len(clean_train_documents)
print "Length of train data:"+str(num_documents)

num_test_documents = len(clean_test_documents)
print "Length of test data:"+str(num_test_documents)

# model = word2vec.Word2Vec.load_word2vec_format('../word2vec/GoogleNews-vectors-negative300.bin',binary=True)
model = word2vec.Word2Vec(size=300,min_count=1)
train_sentences = []
for doc in clean_train_documents:
    train_sentences.append(doc.split())

model.build_vocab(train_sentences)
# model.train(sentences)
print len(model.wv.vocab)

## use pretrained word vectors
model.intersect_word2vec_format('../word2vec/GoogleNews-vectors-negative300.bin',binary=True)
# model.train(train_sentences)

num_features = 300
trainDataVecs = getAvgFeatureVecs( train_sentences, model, num_features )

if classifier_par=="svm":
    svc = svm.LinearSVC()
    parameters = [{'C':[0.01,0.1,1,10,100,1000]}]
    clf = GridSearchCV(svc, parameters,n_jobs=-1,cv=10)
elif classifier_par=="log":
    clf = SGDClassifier(loss="log")

X = trainDataVecs
rowsX,colsX = X.shape
Y = y_train

classLabels = np.unique(Y) # different class labels on the dataset
classNum = len(classLabels) # number of classes
print "Number of classes:"+str(classNum)

classes_in_integers = np.zeros(rowsX)
for i in range(rowsX):
	for j in range(classNum):
		if classLabels[j]==Y[i]:
			classes_in_integers[i] = j

y_train = classes_in_integers
print classes_in_integers
print "y.shape:"+str(y_train.shape)

print "Training the classifier..."
start = time.time()

# This may take a few minutes to run
forest = clf.fit( trainDataVecs, y_train )
pred_train = forest.predict(trainDataVecs)

end = time.time()
elapsed = end - start
print "Total time:"+str(elapsed)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable

text_file = open("output_w2vec.txt","w")

# training score
score = accuracy_score(y_train, pred_train)
#score = metrics.f1_score(y_test, pred_test, pos_label=list(set(y_test)))
acc = "Accuracy in training set:"+str(score)
print acc
mac = "Macro:"+str(metrics.precision_recall_fscore_support(y_train, pred_train, average='macro'))
print mac
mic = "Micro:"+str(metrics.precision_recall_fscore_support(y_train, pred_train, average='micro'))
print mic

# met = metrics.classification_report(y_train, pred_train, target_names=categories,digits=4)
# print met

text_file.write("\n"+"Shape of features:"+str(trainDataVecs.shape)+"\n")
text_file.write(acc+"\n"+mac+"\n"+mic+"\n")

end = time.time()
elapsed = end - start
print "Total time:"+str(elapsed)

test_sentences = []
for doc in clean_test_documents:
    test_sentences.append(doc.split())

testDataVecs = getAvgFeatureVecs( test_sentences, model, num_features )

rowsX,colsX = testDataVecs.shape
print testDataVecs.shape

classLabels_test = np.unique(y_test) # different class labels on the dataset
classNum_test = len(classLabels_test) # number of classes
print "Number of classes:"+str(classNum_test)

classes_in_integers_test = np.zeros(rowsX)
for i in range(rowsX):
	for j in range(classNum_test):
		if classLabels_test[j]==y_test[i]:
			classes_in_integers_test[i] = j

y_test = classes_in_integers_test

pred_test = forest.predict(testDataVecs)

# testing score
score = accuracy_score(y_test, pred_test)
#score = metrics.f1_score(y_test, pred_test, pos_label=list(set(y_test)))
acc = "Accuracy in testing set:"+str(score)
print acc
mac = "Macro test:"+str(metrics.precision_recall_fscore_support(y_test, pred_test, average='macro'))
print mac
mic = "Micro test:"+str(metrics.precision_recall_fscore_support(y_test, pred_test, average='micro'))
print mic

met = metrics.classification_report(y_test, pred_test, target_names=categories, digits=4)
print met

## Statistical Significance
base_test = []
with open("tfidf_predictions_False.txt","r") as f:
    for line in f:
        base_test.append(int(line))

n_trials = np.sum(base_test!=pred_test)
n_succ = 0
p = 0.05
for count_elem,y_elem in enumerate(pred_test):
    if y_elem==y_test[count_elem] and y_test[count_elem]!=base_test[count_elem]:
        n_succ+=1

p_value = scipy.stats.binom_test(n_succ,n_trials)
sign_bool = p_value < p
print "Significance:"+str(p_value)+" "+str(sign_bool)

text_file.write("Features shape:"+str(testDataVecs.shape)+"\n")
text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+met+"\n\n")
text_file.write("Significance:"+str(p_value)+" "+str(sign_bool)+"\n")

text_file.close()

np.savetxt("w2vec_predictions.txt",pred_test,fmt='%i')
