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
from sklearn.grid_search import GridSearchCV
import codecs
import scipy
import sys
sys.path.append('../../code')

from library import *
from MyGraph import *

b = 0.20
unique_words = []
bigrams = []
trigrams = []
terms = []

ngrams_par = 1
idf_bool = True
freq_par = "not-binary"

bag_of_words = "MY TF-IDF"
classifier_par = "svm"

words_frequency = {}

# if (not os.path.isfile("20newsgroup_cleaned_data.txt")):
# categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

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

print "Unique words:"+str(len(unique_words))

print "Length of train data:"+str(num_documents)

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
print "Creating the bag of words..."

if bag_of_words=="vectorizer":

    tfidf_vect = TfidfVectorizer(analyzer = "word", tokenizer = None,lowercase= True, min_df=1, max_features=None, norm=None, binary=True, ngram_range=(1,ngrams_par), use_idf=False)
    #print train_data_features.shape
    features = tfidf_vect.fit_transform(clean_train_documents)

else:
    # MY TF-IDF
    print "Number of unique_words:"+str(len(unique_words))
    if ngrams_par==3:
        print "Number of trigrams:"+str(len(trigrams))
        print "Number of bigrams:"+str(len(bigrams))
        features = np.zeros((num_documents,len(unique_words)+len(bigrams)+len(trigrams)))
    elif ngrams_par==2:
        print "Number of bigrams:"+str(len(bigrams))
        features = np.zeros((num_documents,len(unique_words)+len(bigrams)))
    elif ngrams_par==1:
        features = np.zeros((num_documents,len(unique_words)))

    term_num_docs = {}

    totalLen = 0

    for i in range( 0,num_documents ):
        #dG = nx.Graph()
        found_unique_words = []
        wordList1 = clean_train_documents[i].split(None)
        wordList2 = [string.rstrip(x, ',.!?;') for x in wordList1]

        docLen = len(wordList2)
        totalLen += docLen

        bigram=""
        prev_unigram = ""
        prev_bigram = ""
        for k, word in enumerate(wordList2):
        #for k, word in enumerate(terms):

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
    for x,value in term_num_docs.iteritems():
        idf_col[x] = math.log10((float(num_documents)+1.0) / (term_num_docs[x]))

    blen = len(unique_words)-1
    tlen = len(bigrams)-1

    for i in range( 0,num_documents ):
        tf = dict()
        wordList1 = clean_train_documents[i].split(None)
        wordList2 = [string.rstrip(x, ',.!?;') for x in wordList1]
        docLen = len(wordList2)

        for k, word in enumerate(wordList2):
        # for k, word in enumerate(terms):
            # use frequencies
            tf[word] = wordList2.count(word)

            # # use binary freqs
            # if word in clean_train_documents[i]:
            #     if freq_par=="binary":
            #         tf[word] = 1

        # # Compute tf weighs for bigrams
        # found_bigrams = find_bigrams(wordList2)
        # s = " ".join(wordList2)
        # for k, bi in enumerate(found_bigrams):
        #     tf[bi] = s.count(bi)

        # sum_freq_bigrams = sum(bigrams_freq.values())

        # # Compute tf weigths for trigrams
        # found_trigrams = find_ngrams(wordList2,3)
        # for k, tri in enumerate(found_trigrams):
        #     tf[tri] = s.count(tri)

        for g, val in tf.iteritems():
            # Degree centrality (local feature)

            if g in unique_words:
                #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                #tf_g = 1+math.log(1+math.log(tf[g]))
                tf_g = tf[g]
                if idf_bool:
                    #features[i,unique_words.index(g)] = (float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))) * idf_col[g]
                    features[i,unique_words.index(g)] = float(tf[g]) * idf_col[g]
                else:
                    features[i,unique_words.index(g)] = tf_g
                    # features[i,terms.index(g)] = float(tf[g])


if classifier_par=="svm":
    svc = svm.LinearSVC()
    parameters = [{'C':[0.01,0.1,1,10,100,1000]}]
    clf = GridSearchCV(svc, parameters,n_jobs=-1,cv=10)
elif classifier_par=="log":
    clf = SGDClassifier(loss="log")

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

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
features = np.concatenate((features,trainDataVecs),axis=1)
print features.shape

print "Training the classifier..."
start = time.time()

forest = clf.fit( features, y )

if bag_of_words=="vectorizer":
    text_file = open("vectorizer_output_tf_idf_"+str(idf_bool)+"_"+str(ngrams_par)+".txt","w")
else:
    text_file = open("output_tf_idf_"+str(idf_bool)+"_"+str(ngrams_par)+".txt","w")

end = time.time()
elapsed = end - start
print "Total time:"+str(elapsed)

# Loop over each document; create an index i that goes from 0 to the length
# of the document list
# number_of_instances_per_class_test = dict()
# for i in xrange( 0, num_test_documents ):
#     # Call our function for each one, and add the result to the list of
#     # clean reviews
#     # clean_test_documents.append( test['text'][i] )
#     if y_test[i] not in number_of_instances_per_class_test:
#     	number_of_instances_per_class_test[y_test[i]] = 1
#     else:
#     	number_of_instances_per_class_test[y_test[i]] += 1
#
# print number_of_instances_per_class_test
# count=0
# for value in number_of_instances_per_class_test.values():   # Iterate via values
#   	count += value
#
# print "Number of TOTAL instances in test:"+str(count)

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
print "Creating the bag of words for the test set..."

if bag_of_words=="vectorizer":
    # # Careful: here we use transform and not fit_transform
    features_test = tfidf_vect.transform(clean_test_documents)

else:
    if ngrams_par==2:
        features_test = np.zeros((num_test_documents,len(unique_words)+len(bigrams)))
    else:
        features_test = np.zeros((num_test_documents,len(unique_words)))

    term_num_docs_test = {}

    totalLen = 0
    for i in range( 0,num_test_documents ):
        #dG = nx.Graph()
        found_unique_words_test = []
        wordList1 = clean_test_documents[i].split(None)
        wordList2 = [string.rstrip(x, ',.!?;') for x in wordList1]

        docLen_test = len(wordList2)
        totalLen += docLen_test

        bigram=""
        prev_unigram = ""
        prev_bigram = ""

    # avgLen = float(totalLen)/count
    print "Average document length in test set:"+str(avgLen)
    # idf_col_test = {}
    # for x in term_num_docs_test:
    #     idf_col_test[x] = math.log10((float(num_test_documents)+1.0) / (term_num_docs_test[x]))

    for i in range( 0,num_test_documents ):

        tf_test = dict()
        wordList1 = clean_test_documents[i].split(None)
        wordList2 = [string.rstrip(x, ',.!?;') for x in wordList1]
        docLen_test = len(wordList2)

        for k, word in enumerate(wordList2):
            tf_test[word] = wordList2.count(word)

        # for k, word in enumerate(terms):
        #     # use frequencies
        #     # tf[word] = wordList2.count(word)

            # use binary freqs
            # if word in clean_test_documents[i]:
            #     if freq_par=="binary":
            #         tf_test[word] = 1

        # Compute tf weights for bigrams
        # found_bigrams = find_bigrams(wordList2)
        # s = " ".join(wordList2)
        # for k, bi in enumerate(found_bigrams):
        #     tf_test[bi] = s.count(bi)

        # if ngrams_par=="trigrams":
        #     # Compute tf weights for trigrams
        #     found_trigrams = find_ngrams(wordList2,3)
        #     for k, tri in enumerate(found_trigrams):
        #         tf_test[tri] = s.count(tri)

        blen = len(unique_words)-1
        tlen = len(bigrams)-1

        for k, g in enumerate(tf_test):
            # Degree centrality (local feature)

            if g in unique_words:
                #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                #tf_g = 1+math.log(1+math.log(tf_test[g]))
                # tf_g = tf_test[g]
                if idf_bool:
                    #features_test[i,unique_words.index(g)] = (float(tf_g)/(1-b+(b*(float(docLen_test)/avgLen)))) * idf_col[g]
                    features_test[i,unique_words.index(g)] = float(tf_test[g]) * idf_col[g]
                else:
                    features_test[i,unique_words.index(g)] = tf_test[g]
                    # features_test[i,terms.index(g)] = float(tf_test[g])

# # Numpy arrays are easy to work with, so convert the result to an
# # array
# # X_test_tfidf = X_test_tfidf.toarray()
# # print X_test_tfidf.shape

rowsX,colsX = features_test.shape
print features_test.shape

classLabels_test = np.unique(y_test) # different class labels on the dataset
classNum_test = len(classLabels_test) # number of classes
print "Number of classes:"+str(classNum_test)

classes_in_integers_test = np.zeros(rowsX)
for i in range(rowsX):
	for j in range(classNum_test):
		if classLabels_test[j]==y_test[i]:
			classes_in_integers_test[i] = j

y_test = classes_in_integers_test

test_sentences = []
for doc in clean_test_documents:
    test_sentences.append(doc.split())

testDataVecs = getAvgFeatureVecs( test_sentences, model, num_features )

features_test = np.concatenate((features_test, testDataVecs),axis=1)
print features_test.shape

pred_test = forest.predict(features_test)

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

#np.savetxt("ground_truth_predictions_"+str(idf_bool)+".txt",y_test,fmt='%i')

np.savetxt("tfidf_W2VEC_predictions_"+str(idf_bool)+".txt",pred_test,fmt='%i')


text_file.write("Features shape:"+str(features.shape)+"\n")
text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+met+"\n\n")
text_file.write("Significance:"+str(p_value)+" "+str(sign_bool)+"\n")
text_file.close()
