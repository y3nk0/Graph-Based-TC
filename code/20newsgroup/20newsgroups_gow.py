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
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn import svm
from sklearn.metrics import accuracy_score
from nltk.stem import *
from nltk.corpus import stopwords
import time
import re
import os.path
import math
import sys
sys.path.append('../../code')
from MyGraph import createGraphFeatures

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def parseXmlStopStemRem(raw_data,unique_words,bigrams,test_bool):
    stemmer = PorterStemmer()

    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

    print "stop list:"+str(len(stop))

    reviews = []

    # for element in raw_data:
    for i,j in enumerate(raw_data):
        s = j
        #remove punctuation and split into seperate words
        s = re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)

        stemmed_text = ""
        prev_unigram = ""
        for word in s:
            w = word.lower()
            if re.match("^[a-zA-Z]*$", w) and not re.match("^[~!@#$%^&*()_+{}':;]+$",w):
                if w not in stop:
                    if len(w)>=3:
                        sw = stemmer.stem(w)
                        stemmed_text += sw + " "
                        if test_bool==False:
                            if sw not in unique_words:
                                unique_words.append(sw)
                            # if prev_unigram!="":
                            #     bi = prev_unigram + " " + sw
                            #     if bi not in bigrams:
                            #         bigrams.append(bi)

                        prev_unigram = sw
        
        # print stemmed_text
        reviews.append(stemmed_text)

    return reviews, unique_words, bigrams

def find_bigrams(input_list):
    bigram_list = []
    for i in range(len(input_list)-1):
        bigram_list.append(input_list[i]+" "+input_list[i+1])
    return bigram_list

def weighted_degree_centrality(dG,alpha=0.5):
    degree = {}
    for node1 in dG.nodes():
        k_i = 0.0
        s_i = 0.0
        for node2 in dG.neighbors(node1):
            k_i += 1.0
            s_i += dG.edge[node1][node2]['weight']
        degree[node1] = math.pow(k_i,(1-alpha))*math.pow(s_i,alpha)

    return degree

b = 0.003
idf_pars = ["tf-icw"]

sliding_window = 3

# A1 stands for remove and then rank features
# A2 stands for rank features and then remove
# kcore_pars = ["A2","A1"]
# kcore_pars = ["B2"]
kcore_pars = ["A0"]

unique_words = []
bigrams = []

# categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
raw_data = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
train_data = raw_data.data
categories = list(raw_data.target_names)

print "Length of train data:"+str(len(train_data))
train,unique_words,bigrams = parseXmlStopStemRem(train_data,unique_words,bigrams,False)
y_train = raw_data.target

# Get the number of documents based on the dataframe column size
num_documents = len(train)

# Initialize an empty list to hold the clean-preprocessed documents
clean_train_documents = train

# Loop over each document; create an index i that goes from 0 to the length
# of the document list 
# for i in xrange( 0, num_documents ):
#     # Call our function for each one, and add the result to the list of
#     # clean reviews
#     for n, w in enumerate( train[i].split() ):
#         if w not in unique_words:
#             unique_words.append( w )
#     clean_train_documents.append( train[i] )

print "Unique words:"+str(len(unique_words))

# if (not os.path.isfile("reuters_gow_train.txt")):
#     print ("Creating the graph of words..."),
#     features = createGraphFeatures(num_documents,clean_train_documents,unique_words,sliding_window)
#     np.savetxt("reuters_gow_train.txt", features, fmt='%i')
#     print "\t Done!"
# else:
#     print ("Loading the graph of words..."),
#     features = np.loadtxt("reuters_gow_train.txt")
#     print "\t Done!"

# centrality_pars = ["degree_centrality","clustering_coefficient","core_number","closeness_centrality","betweenness_centrality","in_degree_centrality","out_degree_centrality","pagerank_centrality","closeness_centrality_directed","betweenness_centrality_directed"]
centrality_pars = ["degree_centrality"]

for kcore_par in kcore_pars:
    for idf_par in idf_pars:
        for centrality_par in centrality_pars:
            centrality_col_par = centrality_par

            print "idf:"+idf_par
            print "centrality_par:"+centrality_par
            print "centrality_col_par:"+centrality_col_par

            # the file for keeping the kcore, accuracy and reducing of features
            myfile = open("results_"+kcore_par+".txt","w")

            # choose the k number for removing the top k-core nodes
            for kcore_par_int in range(1,2):

                idfs = {}
                icws = {}
                dGcol_nodes = {}
                max_core_col = []
                max_core_feat = []
                feature_reduction = 0.0
                avgLen = 0.0

                features, idfs_learned,icws_learned,collection_count_nodes, collection_count_edges, dGcol_nodes,max_core_col,feature_reduction, max_core_feat,avgLen = createGraphFeatures(num_documents,clean_train_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,True,idfs,icws,kcore_par,dGcol_nodes,max_core_col,kcore_par_int,max_core_feat,feature_reduction,avgLen)

                print "Training the classifier..."
                start = time.time()

                # Initialize a Random Forest classifier with 100 trees
                #clf = RandomForestClassifier(n_estimators = 100) 
                # clf = svm.SVC(kernel="linear",probability=True)
                clf = svm.LinearSVC(loss="hinge")
                #clf = AdaBoostClassifier(n_estimators=100)

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
                
                forest = clf.fit( features, y )
                pred_train = forest.predict(features)

                if idf_par=="no":
                    text_file = open("output_tw_idf_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+".txt","w")
                elif idf_par=="tf-icw":
                    text_file = open("output_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+".txt","w")
                elif idf_par=="idf":
                    text_file = open("output_tw_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+".txt","w")
                elif idf_par=="icw" or idf_par=="icw+idf":
                    if kcore_par=="A1" or kcore_par=="A2" or kcore_par=="B1" or kcore_par=="B2":
                        text_file = open("output_tw_"+idf_par+"_centr_"+centrality_par+"_centrcol_"+centrality_col_par+"_sliding_"+str(sliding_window)+"_kcore_"+kcore_par+"_"+str(kcore_par_int)+"_UPDATED.txt","w")
                    else:
                        text_file = open("output_tw_"+idf_par+"_centr_"+centrality_par+"_centrcol_"+centrality_col_par+"_sliding_"+str(sliding_window)+"_"+kcore_par+"_UPDATED.txt","w")

                # training score
                score = accuracy_score(y, pred_train)
                #score = metrics.f1_score(y_test, pred_test, pos_label=list(set(y_test)))
                acc = "Accuracy in training set:"+str(score)
                print acc
                mac = "Macro:"+str(metrics.precision_recall_fscore_support(y, pred_train, average='macro'))
                print mac
                mic = "Micro:"+str(metrics.precision_recall_fscore_support(y, pred_train, average='micro'))
                print mic

                tp = np.zeros(classNum)
                fn = np.zeros(classNum)
                fp = np.zeros(classNum) 
                tn = np.zeros(classNum)

                for j in range(classNum):
                    for i in range(rowsX):
                        if y_train[i]==j:
                            if pred_train[i]==j:
                                tp[j] += 1
                            else:
                                fn[j] += 1
                        else:
                            if pred_train[i]==j:
                                fp[j] += 1
                            else:
                                tn[j] += 1


                pr_micro = float(np.sum(tp))/np.sum(np.add(tp,fp))
                pr_micro_str = "Precision micro:"+str(pr_micro)
                print pr_micro_str
                rec_micro = float(np.sum(tp))/np.sum(np.add(tp,fn))
                rec_micro_str = "Recall micro:"+str(rec_micro)
                print rec_micro_str
                f1_score_micro = 2*(float(pr_micro*rec_micro)/(pr_micro+rec_micro))
                f1_score_micro_str = "f1-score micro:"+str(f1_score_micro)               
                print f1_score_micro_str

                met = metrics.classification_report(y_train, pred_train, target_names=categories,digits=4)
                print met

                text_file.write("\n"+"Shape of features:"+str(features.shape)+"\n")
                text_file.write("Collection training Number of nodes:"+str(collection_count_nodes)+"\n")
                text_file.write("collection training Number of edges:"+str(collection_count_edges)+"\n")
                text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+pr_micro_str+"\n"+rec_micro_str+"\n"+f1_score_micro_str+"\n"+met)

                end = time.time()
                elapsed = end - start
                print "Total time:"+str(elapsed)

                ## Testing set
                raw_test_data = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
                test_data = raw_test_data.data
                print "Length of test data:"+str(len(test_data))
                test,unique_words,bigrams = parseXmlStopStemRem(test_data,unique_words,bigrams,True)
                y_test = raw_test_data.target

                # Get the number of documents based on the dataframe column size
                num_test_documents = len(test)

                # Initialize an empty list to hold the clean-preprocessed documents
                clean_test_documents = test

                # Loop over each document; create an index i that goes from 0 to the length
                # of the document list 
                # for i in xrange( 0, num_test_documents ):
                #     # Call our function for each one, and add the result to the list of
                #     # clean reviews
                #     clean_test_documents.append( test['text'][i] )
                #     # print train['text'][i]+'\n'

                test_features,idfs,icws,collection_count_nodes,collection_count_edges,dGcol_nodes,max_core_col,feature_reduction,max_core_feat,avgLen = createGraphFeatures(num_test_documents,clean_test_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,False,idfs_learned,icws_learned,kcore_par,dGcol_nodes,max_core_col,kcore_par_int,max_core_feat,feature_reduction,avgLen)
                
                print test_features.shape
                X_test = test_features
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

                tp_test = np.zeros(classNum_test)
                fn_test = np.zeros(classNum_test)
                fp_test = np.zeros(classNum_test) 
                tn_test = np.zeros(classNum_test)
                for j in range(classNum_test):
                    for i in range(rowsX):
                        if y_test[i]==j:
                            if pred_test[i]==j:
                                tp_test[j] += 1
                            else:
                                fn_test[j] += 1
                        else:
                            if pred_test[i]==j:
                                fp_test[j] += 1
                            else:
                                tn_test[j] += 1

                pr_micro = float(np.sum(tp_test))/np.sum(np.add(tp_test,fp_test))
                pr_micro_str = "Precision micro:"+str(pr_micro)
                print pr_micro_str
                rec_micro = float(np.sum(tp_test))/np.sum(np.add(tp_test,fn_test))
                rec_micro_str = "Recall micro:"+str(rec_micro)
                print rec_micro_str
                f1_score_micro = 2*(float(pr_micro*rec_micro)/(pr_micro+rec_micro))
                f1_score_micro_str = "f1-score micro:"+str(f1_score_micro)               
                print f1_score_micro_str

                met = metrics.classification_report(y_test, pred_test, target_names=categories, digits=4)
                print met

                text_file.write("\n"+"Features shape:"+str(features.shape)+"\n")
                text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+pr_micro_str+"\n"+rec_micro_str+"\n"+f1_score_micro_str+"\n"+met)
                text_file.close()

                myfile.write("Accuracy:"+str(score)+"\n")
                myfile.write("kcore_par_int:"+str(kcore_par_int)+"\n")
                myfile.write("Feature reduction:"+str(feature_reduction)+"\n\n")

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

            myfile.close()