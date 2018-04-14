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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from nltk.stem import *
from nltk.corpus import stopwords
import time
import re
import os.path
import math
import codecs
import sys
sys.path.append('../')
from MyGraph import *

import glob

from sklearn.model_selection import GridSearchCV

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

b = 0.003
idf_pars = ["icw"]
kcore_par = "A0"
classifier_par = "svm"

path = "data/graphs/subject"
sliding_windows = [2]
# sliding_windows = range(2,11)

all_results = open("all_results.txt","a")

accs = []
f1s = []

# centrality_pars = ["degree_centrality","clustering_coefficient","core_number","closeness_centrality","betweenness_centrality","in_degree_centrality","out_degree_centrality","pagerank_centrality","closeness_centrality_directed","betweenness_centrality_directed"]
# centrality_pars = ["degree_centrality"]
centrality_pars = ["degree_centrality"]

for idf_par in idf_pars:
    for sliding_window in sliding_windows:
        for centrality_par in centrality_pars:
            # centrality_col_par = centrality_par
            centrality_col_par = "degree_centrality"

            print "idf:"+idf_par
            print "centrality_par:"+centrality_par
            print "centrality_col_par:"+centrality_col_par

            idfs = {}
            icws = {}
            dGcol_nodes = {}
            max_core_col = []
            max_core_feat = []
            feature_reduction = 0.0
            avgLen = 0.0

            Y = y_train

            classLabels = np.unique(Y) # different class labels on the dataset
            classNum = len(classLabels) # number of classes
            print "Number of classes:"+str(classNum)

            classes_in_integers = np.zeros(num_documents)
            for i in range(num_documents):
                for j in range(classNum):
                    if classLabels[j]==Y[i]:
                        classes_in_integers[i] = j

            y_train = classes_in_integers

            features, idfs_learned,icws_learned,collection_count_nodes, collection_count_edges, dGcol_nodes,avgLen = createGraphFeatures(num_documents,clean_train_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,True,idfs,icws,dGcol_nodes,avgLen,path,y_train)

            #features, idfs_learned,icws_learned,collection_count_nodes, collection_count_edges, dGcol_nodes,avgLen = averageCentralities(num_documents,clean_train_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,True,idfs,icws,dGcol_nodes,avgLen,path,y_train)

            # for node in unique_words:
            #     if node not in dGcol_nodes:
            #         print node
            #         print len(node)
            #         raw_input("enter")

            print "Training the classifier..."
            start = time.time()

            # Initialize a Random Forest classifier with 100 trees
            #clf = RandomForestClassifier(n_estimators = 100)
            # clf = svm.SVC(kernel="linear",probability=True)
            if classifier_par=="svm":
                svc = svm.LinearSVC()
                parameters = [{'C':[0.01,0.1,1,10,100,1000]}]
                clf = GridSearchCV(svc, parameters,n_jobs=-1,cv=10)
            elif classifier_par=="log":
                clf = SGDClassifier(loss="log")


            forest = clf.fit( features, y_train )
            pred_train = forest.predict(features)

            path_results = "results/"
            if idf_par=="no":
                text_file = open(path_results+"output_tw_idf_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+".txt","w")
            elif idf_par=="tf-icw":
                text_file = open(path_results+"output_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+".txt","w")
            elif idf_par=="idf":
                text_file = open(path_results+"output_tw_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+".txt","w")
            elif idf_par=="icw" or idf_par=="icw+idf" or idf_par=="icw-lw":

                text_file = open(path_results+"output_tw_"+idf_par+"_centr_"+centrality_par+"_centrcol_"+centrality_col_par+"_sliding_"+str(sliding_window)+"_UPDATED.txt","w")

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

            text_file.write("\n"+"Shape of features:"+str(features.shape)+"\n")
            text_file.write("Collection training Number of nodes:"+str(collection_count_nodes)+"\n")
            text_file.write("collection training Number of edges:"+str(collection_count_edges)+"\n")
            text_file.write(acc+"\n"+mac+"\n"+mic+"\n")

            end = time.time()
            elapsed = end - start
            print "Total time:"+str(elapsed)

            # Loop over each document; create an index i that goes from 0 to the length
            # of the document list
            # for i in xrange( 0, num_test_documents ):
            #     # Call our function for each one, and add the result to the list of
            #     # clean reviews
            #     clean_test_documents.append( test['text'][i] )
            #     # print train['text'][i]+'\n'

            test_features,idfs,icws,collection_count_nodes,collection_count_edges,dGcol_nodes,avgLen = createGraphFeatures(num_test_documents,clean_test_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,False,idfs_learned,icws_learned,dGcol_nodes,avgLen,path,y_test)

            #test_features,idfs,icws,collection_count_nodes,collection_count_edges,dGcol_nodes,avgLen = averageCentralities(num_test_documents,clean_test_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,False,idfs_learned,icws_learned,dGcol_nodes,avgLen,path,y_test)

            print test_features.shape
            X_test = test_features

            Y_test = y_test

            classLabels_test = np.unique(Y_test) # different class labels on the dataset
            classNum_test = len(classLabels_test) # number of classes
            print "Number of classes:"+str(classNum_test)

            classes_in_integers_test = np.zeros(num_test_documents)
            for i in range(num_test_documents):
                for j in range(classNum_test):
                    if classLabels_test[j]==Y_test[i]:
                        classes_in_integers_test[i] = j

            y_test = classes_in_integers_test
            pred_test = forest.predict(test_features)

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


            text_file.write("\n"+"Features shape:"+str(features.shape)+"\n")
            text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+met+"\n\n")
            text_file.write("Significance:"+str(p_value)+" "+str(sign_bool)+"\n")
            text_file.close()

            if idf_par=="icw" or idf_par=="icw+idf":
                s_res = idf_par+"_centr_"+centrality_par+"_centrcol_"+centrality_col_par+"_sliding_"+str(sliding_window)+".txt"
            else:
                s_res = idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+".txt"

            np.savetxt(centrality_col_par+"_"+str(sliding_window)+".txt",pred_test,fmt='%i')


            all_results.write(s_res+" Accuracy:"+str(score)+" Significance:"+str(p_value)+" "+str(sign_bool)+"\n")

            accs.append(score)
            f1s.append(metrics.f1_score(y_test, pred_test, average='macro'))

            # # Compute confusion matrix
            # cm = confusion_matrix(y_test, pred_test)
            # np.set_printoptions(precision=2)
            # print('Confusion matrix, without normalization')
            # print(cm)
            # plt.figure()
            # plot_confusion_matrix(cm,classLabels)

            # # Normalize the confusion matrix by row (i.e by the number of samples
            # # in each class)
            # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print('Normalized confusion matrix')
            # print(cm_normalized)
            # plt.figure()
            # plot_confusion_matrix(cm_normalized, classLabels,title='Normalized confusion matrix')

            # plt.show()

all_results.close()
print accs
print f1s
