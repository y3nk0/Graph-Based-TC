import networkx as nx
from sklearn import metrics

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
sys.path.append('../')
from MyGraph import *

# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

# A1 stands for remove and then rank features
# A2 stands for rank features and then remove
# kcore_pars = ["A2","A1"]
# kcore_pars = ["B2"]

unique_words = []
bigrams = []
trigrams = []
words_frequency = {}
categories = ['course','faculty','project','student']

clean_train_documents = []
y_train = []

y_test = []
clean_test_documents = []

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

    unique_words = findVocab(train_data)

    f = codecs.open('data/my_WEBKB_train_VOCAB.txt', "w", encoding="utf-8")
    for item in unique_words:
        f.write("%s\n" % item)
    f.close()

    train_data_NEW = []
    for t in train_data:
        train_data_NEW.append(remove_tags(t))

    train_data = parseXmlStopStemRem(train_data_NEW,unique_words,bigrams,trigrams,True)

    train_data, test_data, y_train_all, y_test_all = train_test_split(train_data,y_train_all,test_size=0.33,random_state=42)

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

# Get the number of documents based on the dataframe column size

print "Unique words:"+str(len(unique_words))

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

b = 0.003
idf_pars = ["icw-lw"]
kcore_par = "A0"
classifier_par = "svm"

path = "data/graphs/webkb"
sliding_windows = [2]
# sliding_windows = range(2,11)

all_results = open("all_results.txt","a")

# centrality_pars = ["degree_centrality","clustering_coefficient","core_number","closeness_centrality","betweenness_centrality","in_degree_centrality","out_degree_centrality","pagerank_centrality","closeness_centrality_directed","betweenness_centrality_directed"]
# centrality_pars = ["degree_centrality"]
centrality_pars = ["degree_centrality"]
accs = []
f1s = []

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
                        classes_in_integers[i] = int(j)

            y_train = classes_in_integers


            start = time.time()

            features, idfs_learned,icws_learned,collection_count_nodes, collection_count_edges, dGcol_nodes,avgLen = createGraphFeatures(num_documents,clean_train_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,True,idfs,icws,dGcol_nodes,avgLen,path,y_train)

            #features, idfs_learned,icws_learned,collection_count_nodes, collection_count_edges, dGcol_nodes,avgLen = averageCentralities(num_documents,clean_train_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,True,idfs,icws,dGcol_nodes,avgLen,path,y_train)

            # Initialize a Random Forest classifier with 100 trees
            #clf = RandomForestClassifier(n_estimators = 100)
            # clf = svm.SVC(kernel="linear",probability=True)
            if classifier_par=="svm":
                svc = svm.LinearSVC()
                parameters = [{'C':[0.01,0.1,1,10,100,1000]}]
                clf = GridSearchCV(svc, parameters,n_jobs=-1,cv=10)
            elif classifier_par=="log":
                clf = SGDClassifier(loss="log")
            elif classifier_par=="cnn":
                clf = SGDClassifier(loss="log")

            print "Training the classifier..."
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

            # Loop over each document; create an index i that goes from 0 to the length
            # of the document list
            # for i in xrange( 0, num_test_documents ):
            #     # Call our function for each one, and add the result to the list of
            #     # clean reviews
            #     clean_test_documents.append( test['text'][i] )
            #     # print train['text'][i]+'\n'

            test_features,idfs,icws,collection_count_nodes,collection_count_edges,dGcol_nodes,avgLen = createGraphFeatures(num_test_documents,clean_test_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,False,idfs_learned,icws_learned,dGcol_nodes,avgLen,path,y_test)

            end = time.time()
            elapsed = end - start
            print "Total time:"+str(elapsed)

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
                        classes_in_integers_test[i] = int(j)

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

            # print_top10(unique_words,forest.best_estimator_,classLabels_test)
            # print_bot10(unique_words,forest.best_estimator_,classLabels_test)

all_results.close()
print accs
print f1s
