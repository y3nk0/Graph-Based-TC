from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import svm
from lxml import etree
from nltk.stem import *
from nltk.corpus import stopwords
import re
import os.path
import string
import math
import time
from sklearn.pipeline import Pipeline
import scipy
import codecs
import sys
sys.path.append('../../code')
from MyGraph import *
from library import *
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

b = 0.20
unique_words = []
bigrams = []
trigrams = []
terms = []

ngrams_pars = [1]
idf_bool = False
freq_par = "not-binary"

unique_words = []
bigrams = []
trigrams = []
words_frequency = {}
y_train = []
clean_train_documents = []

y_test = []
clean_test_documents = []

categories = ['subjective','objective']

if os.path.exists("data/my_subject_train.txt"):
    ## Open the file with read only permi

    f = codecs.open('data/my_subject_train_VOCAB.txt', "r")
    unique_words = [x.strip('\n') for x in f.readlines()]
    f.close()

    f = codecs.open('data/my_subject_train.txt', "r")
    train = [x.strip('\n') for x in f.readlines()]
    f.close()

    num_documents = len(train)

    for i in xrange( 0, num_documents ):
        # Call our function for each one, and add the result to the list of
        # clean reviews
        line = train[i].split('\t')

        y_train.append(line[0])
        clean_train_documents.append( line[1] )

	f = codecs.open('data/my_subject_test.txt', "r")
    test = [x.strip('\n') for x in f.readlines()]
    f.close()

    num_test_documents = len(test)

    for i in xrange( 0, num_test_documents ):
        # Call our function for each one, and add the result to the list of
        # clean reviews
		line = test[i].split('\t')

		y_test.append(line[0])
		clean_test_documents.append( line[1] )

else:

    train_data = []
    y_train_all = []

    test_data = []
    y_test_all = []

    f = codecs.open('data/rotten_imdb/plot.tok.gt9.5000', "r")
    train = [x.strip('\n') for x in f.readlines()]
    f.close()

    for i, doc in enumerate(train):
        train_data.append(doc)
        y_train_all.append(0)

    f = codecs.open('data/rotten_imdb/quote.tok.gt9.5000', "r")
    train = [x.strip('\n') for x in f.readlines()]
    f.close()

    for i, doc in enumerate(train):
        train_data.append(doc)
        y_train_all.append(1)

    train_data, test_data, y_train_all, y_test_all = train_test_split(train_data, y_train_all, test_size=0.33, random_state=4)
	# train_data = np.array(train_data)
	# y_train_all = np.array(y_train_all)
	#
	# sss = StratifiedShuffleSplit(y_train_all, 1, test_size=0.2,random_state=2)
	#
	# for train_index, test_index in sss:
	# 	# print("TRAIN:", train_index, "DEV:", dev_index)
	# 	clean_train_documents, clean_test_documents = train_data[train_index], train_data[test_index]
	# 	y_train, y_test = y_train_all[train_index], y_train_all[test_index]

    unique_words = findVocab(train_data)

    f = codecs.open('data/my_subject_train_VOCAB.txt', "w")
    for item in unique_words:
        f.write("%s\n" % item)
    f.close()

    train_data = parseXmlStopStemRem(train_data,unique_words,bigrams,trigrams,True)
    test_data = parseXmlStopStemRem(test_data,unique_words,bigrams,trigrams,False)

    f = codecs.open('data/my_subject_train.txt', "w")
    for i, doc in enumerate(train_data):
        s = doc.split(" ")
        if len(set(s))>1:
            f.write(str(categories[y_train_all[i]])+"\t"+doc+"\n")
            clean_train_documents.append(doc)
            y_train.append(y_train_all[i])

    f.close()

    f = codecs.open('data/my_subject_test.txt', "w")
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
# if (not os.path.isfile("reuters_gow_train.txt")):
#     print ("Creating the graph of words..."),
#     features = createGraphFeatures(num_documents,clean_train_documents,unique_words,sliding_window)
#     np.savetxt("reuters_gow_train.txt", features, fmt='%i')
#     print "\t Done!"
# else:
#     print ("Loading the graph of words..."),
#     features = np.loadtxt("reuters_gow_train.txt")
#     print "\t Done!"

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
print "Creating the bag of words..."

# MY TF-IDF
print "Number of unique_words:"+str(len(unique_words))

features = np.zeros((num_documents,len(unique_words)))

term_num_docs = {}

totalLen = 0

for i in range( 0,num_documents ):
    #dG = nx.Graph()
    found_unique_words = []
    wordList1 = clean_train_documents[i].split(None)
    wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]

    docLen = len(wordList2)
    totalLen += docLen

    bigram=""
    prev_unigram = ""
    prev_bigram = ""
    # for k, word in enumerate(wordList2):
    for k, word in enumerate(terms):
        if word in clean_train_documents[i]:

            if word not in found_unique_words:
                found_unique_words.append(word)
                if word not in term_num_docs:
                    term_num_docs[word] = 1
                else:
                    term_num_docs[word] += 1

            prev_unigram = word
            prev_bigram = bigram

avgLen = float(totalLen)/num_documents
print "Average document length:"+str(avgLen)
idf_col = {}
for x in term_num_docs:
    icf_col[x] = math.log10((float(num_documents)+1.0) / (term_num_docs[x]))

blen = len(unique_words)-1
tlen = len(bigrams)-1

for i in range( 0,num_documents ):
    tf = dict()
    wordList1 = clean_train_documents[i].split(None)
    wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
    docLen = len(wordList2)

    for k, word in enumerate(wordList2):
    # for k, word in enumerate(terms):
        # use frequencies
        tf[word] = wordList2.count(word)

    for k, g in enumerate(tf):
        # Degree centrality (local feature)

        if g in unique_words:
            #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
            #tf_g = 1+math.log(1+math.log(tf[g]))
            # tf_g = tf[g]
            if idf_bool:
                #features[i,unique_words.index(g)] = (float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))) * idf_col[g]
                features[i,unique_words.index(g)] = float(tf[g]) * idf_col[g]
            else:
                #features[i,unique_words.index(g)] = float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))
                features[i,unique_words.index(g)] = float(tf[g])


print "Training the classifier..."
start = time.time()

# Initialize a Random Forest classifier with 100 trees
#clf = RandomForestClassifier(n_estimators = 100)
#clf = AdaBoostClassifier(n_estimators=100)
# clf = svm.SVC(kernel="linear",probability=True)

svc = svm.LinearSVC()
parameters = [{'C':[0.01,0.1,1,10,100,1000]}]
clf = GridSearchCV(svc, parameters,n_jobs=-1,cv=10)

X = features
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

y = classes_in_integers
print classes_in_integers
print "y.shape:"+str(y.shape)
# cv = StratifiedKFold(y, n_folds=5)

#
# This may take a few minutes to run
forest = clf.fit( features, y )

text_file = open("output_tf_icf_"+str(idf_bool)+".txt","w")

# training score
pred_train = forest.predict(features)
score = accuracy_score(y_train, pred_train)
#score = metrics.f1_score(y_test, pred_test, pos_label=list(set(y_test)))
acc = "Accuracy in training set:"+str(score)
print acc
mac = "Macro:"+str(metrics.precision_recall_fscore_support(y_train, pred_train, average='macro'))
print mac
mic = "Micro:"+str(metrics.precision_recall_fscore_support(y_train, pred_train, average='micro'))
print mic


# Loop over each document; create an index i that goes from 0 to the length
# of the document list
number_of_instances_per_class_test = dict()
for i in xrange( 0, num_test_documents ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    # clean_test_documents.append( test['text'][i] )
    if y_test[i] not in number_of_instances_per_class_test:
    	number_of_instances_per_class_test[y_test[i]] = 1
    else:
    	number_of_instances_per_class_test[y_test[i]] += 1

print number_of_instances_per_class_test
count=0
for value in number_of_instances_per_class_test.values():   # Iterate via values
  	count += value

print "Number of TOTAL instances in test:"+str(count)

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
print "Creating the bag of words for the test set..."

# Careful: here we use transform and not fit_transform
# features_test = tfidf_vect.transform(clean_test_documents)

features_test = np.zeros((num_test_documents,len(unique_words)))

term_num_docs_test = {}

totalLen = 0
for i in range( 0,count ):
    #dG = nx.Graph()
    found_unique_words_test = []
    wordList1 = clean_test_documents[i].split(None)
    wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]

    docLen_test = len(wordList2)
    totalLen += docLen_test

# avgLen = float(totalLen)/count
print "Average document length in test set:"+str(avgLen)
# idf_col_test = {}
# for x in term_num_docs_test:
#     idf_col_test[x] = math.log10((float(num_test_documents)+1.0) / (term_num_docs_test[x]))

for i in range( 0,num_test_documents ):

    tf_test = dict()
    wordList1 = clean_test_documents[i].split(None)
    wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
    docLen_test = len(wordList2)

    for k, word in enumerate(wordList2):
        tf_test[word] = wordList2.count(word)

    blen = len(unique_words)-1
    tlen = len(bigrams)-1

    for k, g in enumerate(tf_test):
        # Degree centrality (local feature)

        if g in unique_words:
            #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
            #tf_g = 1+math.log(1+math.log(tf_test[g]))
            tf_g = tf_test[g]
            if idf_bool:
                #features_test[i,unique_words.index(g)] = (float(tf_g)/(1-b+(b*(float(docLen_test)/avgLen)))) * idf_col[g]
                features_test[i,unique_words.index(g)] = float(tf_test[g]) * idf_col[g]
            else:
                #features_test[i,unique_words.index(g)] = float(tf_g)/(1-b+(b*(float(docLen_test)/avgLen)))
                features_test[i,unique_words.index(g)] = float(tf_test[g])


# # Numpy arrays are easy to work with, so convert the result to an
# # array
# # X_test_tfidf = X_test_tfidf.toarray()
# # print X_test_tfidf.shape

X_test = features_test
rowsX,colsX = X_test.shape
Y_test = y_test

classLabels_test = np.unique(Y_test) # different class labels on the dataset
classNum_test = len(classLabels_test) # number of classes
print "Number of classes:"+str(classNum_test)

classes_in_integers_test = np.zeros(rowsX)
for i in range(rowsX):
	for j in range(classNum_test):
		if classLabels_test[j]==Y_test[i]:
			classes_in_integers_test[i] = j

y_test = classes_in_integers_test

pred_test = forest.predict(X_test)

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

text_file.write("Features shape:"+str(features_test.shape)+"\n")
text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+"\n"+met+"\n\n")
text_file.write("Significance:"+str(p_value)+" "+str(sign_bool)+"\n")

text_file.close()
