import networkx as nx
from sklearn import metrics
import string
from sys import maxint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn import svm,grid_search
from sklearn.metrics import accuracy_score
import time
import re
import os.path
import math
import sys
sys.path.append('../../code')
from MyGraph import createGraphFeatures


## Open the file with read only permit
f = open('data/webkb-train-stemmed.txt', "r")

## use readlines to read all lines in the file
## The variable "lines" is a list containing all lines
train = [x.strip('\n') for x in f.readlines()]

## close the file after reading the lines.
f.close()

# Get the number of documents based on the dataframe column size
num_documents = len(train)

# Initialize an empty list to hold the clean-preprocessed documents
clean_train_documents = []
unique_words = []
bigrams = []
y_train = []

# Loop over each document; create an index i that goes from 0 to the length
# of the document list 
for i in xrange( 0, num_documents ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    line = train[i].split('\t')
    y_train.append(line[0])
    for n, w in enumerate( line[1].split(' ') ):
        if w not in unique_words:
            unique_words.append(w)
    clean_train_documents.append( line[1] )

print "Unique words:"+str(len(unique_words))

sliding_windows = [2]
b = 0.03
kcore_par_int = 1 
kcore_par = "A0"
classifier_par = "svm"

# idf_par = "idf"
idf_pars = ["icw"]
# centrality_pars = ["degree_centrality","clustering_coefficient","core_number","closeness_centrality","betweenness_centrality","in_degree_centrality","out_degree_centrality","pagerank_centrality","closeness_centrality_directed","betweenness_centrality_directed"]
# centrality_pars = ["out_degree_centrality","pagerank_centrality","closeness_centrality_directed","betweenness_centrality_directed"]
centrality_pars = ["degree_centrality"]

for sliding_window in sliding_windows:
    for idf_par in idf_pars:
        for centrality_par in centrality_pars:

            print "idf:"+idf_par+"  classifier_par:"+classifier_par
            print "centrality_par:"+centrality_par
            centrality_col_par = centrality_par
            print "centrality_col_par:"+centrality_col_par

            # if (not os.path.isfile("reuters_gow_train.txt")):
            #     print ("Creating the graph of words..."),
            #     features = createGraphFeatures(num_documents,clean_train_documents,unique_words,sliding_window)
            #     np.savetxt("reuters_gow_train.txt", features, fmt='%i')
            #     print "\t Done!"
            # else:
            #     print ("Loading the graph of words..."),
            #     features = np.loadtxt("reuters_gow_train.txt")
            #     print "\t Done!"
            
            Y = y_train

            classLabels = np.unique(Y) # different class labels on the dataset
            classNum = len(classLabels) # number of classes
            print "Number of classes:"+str(classNum)

            classes_in_integers = np.zeros((num_documents))
            for i in range(num_documents):
                for j in range(classNum):
                    if classLabels[j]==Y[i]:
                        classes_in_integers[i] = j+1

            y = classes_in_integers

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
            
            # cv = StratifiedKFold(y, n_folds=5)

            # mean_tpr = 0.0
            # mean_fpr = np.linspace(0, 1, 100) 


            # conf_matrix=np.zeros((2,2))

            # for i, (train, test) in enumerate(cv):
            #     print str(i)
            #     # Binarize the output

            #     y_bin = label_binarize(y, classes=[1,2,3,4,5,6,7,8])
            #     n_classes = y_bin.shape[1]
            #     predictedLabels = clf.fit(X[train], y[train]).predict(X[test]) 
            #     predictedLabels_bin = label_binarize(predictedLabels, classes=[1,2,3,4,5,6,7,8])

            #     # Compute ROC curve and ROC area for each class
            #     fpr = dict()
            #     tpr = dict() 
            #     roc_auc = dict()
            #     for i in range(n_classes):
            #         #probas_ = clf.fit(X[train], y_bin[train]).predict_proba(X[test])
            #         #predictedLabels_bin = label_binarize(probas_, classes=[1,2,3,4,5,6,7,8])
            #         fpr[i], tpr[i], _ = roc_curve(y_bin[test, i], predictedLabels_bin[:, i])
            #         roc_auc[i] = auc(fpr[i], tpr[i])

            #     # Compute micro-average ROC curve and ROC area
            #     fpr["micro"], tpr["micro"], _ = roc_curve(y_bin[test].ravel(), predictedLabels_bin.ravel())
            #     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # # Plot ROC curve
            # plt.figure()
            # plt.plot(fpr["micro"], tpr["micro"],
            #          label='micro-average ROC curve (area = {0:0.2f})'
            #                ''.format(roc_auc["micro"]))
            # for i in range(n_classes):
            #     plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
            #                                    ''.format(i, roc_auc[i]))

            # plt.plot([0, 1], [0, 1], 'k--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Some extension of Receiver Operating Characteristic to multi-class')
            # plt.legend(loc="lower right")
            # plt.show()

            # Fit the forest to the training set, using the bag of words as 
            # features and the sentiment labels as the response variable
            #
            # This may take a few minutes to run
            forest = clf.fit( features, y )

            pred_train = forest.predict(features)

            path = "results/"

            if idf_par=="no":
                text_file = open(path+classifier_par+"_output_tw_idf_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+"_kcore_"+kcore_par+".txt", "w")
            elif idf_par=="idf":
                text_file = open(path+classifier_par+"_output_tw_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+"_kcore_"+kcore_par+".txt", "w")
            elif idf_par=="icw" or idf_par=="icw+idf":
                text_file = open(path+classifier_par+"_output_tw_"+idf_par+"_centr_"+centrality_par+"_centrcol_"+centrality_col_par+"_sliding_"+str(sliding_window)+"_kcore_"+kcore_par+".txt", "w")

            # Sort the coef_ as per feature weights and select largest 20 of them
            # 2 shows that we are considering the third class
            for counter, classLabel in enumerate(classLabels):
                inds = np.argsort(forest.coef_[counter, :])[-20:]
                # Now, just iterate over all these indices and get the corresponding
                # feature names
                text_file.write("Class:"+classLabel)
                for i in inds:
                    text_file.write(","+unique_words[i])
                text_file.write("\n\n")

            # training score
            score = metrics.accuracy_score(classes_in_integers, pred_train)
            #score = metrics.f1_score(y_test, pred_test, pos_label=list(set(y_test)))
            acc = "Accuracy in training set:"+str(score)
            print acc
            mac = "Macro:"+str(metrics.precision_recall_fscore_support(classes_in_integers, pred_train, average='macro'))
            print mac
            mic = "Micro:"+str(metrics.precision_recall_fscore_support(classes_in_integers, pred_train, average='micro'))
            print mic

            met = metrics.classification_report(classes_in_integers, pred_train, target_names=classLabels, digits=4)
            print met

            text_file.write("Results_tw_"+idf_par+"_"+centrality_par+"_sliding_"+str(sliding_window)+"\n\n")
            text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+"\n"+met)

            end = time.time()
            elapsed = end - start
            print "Total time:"+str(elapsed)

            ## Testing set
            ## Open the file with read only permit
            f = open('data/webkb-test-stemmed.txt', "r")

            ## use readlines to read all lines in the file
            ## The variable "lines" is a list containing all lines
            test = [x.strip('\n') for x in f.readlines()]

            ## close the file after reading the lines.
            f.close()

            # Get the number of documents based on the dataframe column size
            num_test_documents = len(test)
            print num_test_documents

            # Initialize an empty list to hold the clean-preprocessed documents
            clean_test_documents = []
            y_test = []

            # Loop over each document; create an index i that goes from 0 to the length
            # of the document list 
            for i in xrange( 0, num_test_documents ):
                # Call our function for each one, and add the result to the list of
                # clean reviews
                line = test[i].split('\t')
                y_test.append(line[0])
                clean_test_documents.append( line[1] )

            test_features,idfs,icws,collection_count_nodes,collection_count_edges,dGcol_nodes,max_core_col,feature_reduction,max_core_feat,avgLen = createGraphFeatures(num_test_documents,clean_test_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,False,idfs_learned,icws_learned,kcore_par,dGcol_nodes,max_core_col,kcore_par_int,max_core_feat,feature_reduction,avgLen)
            
            print test_features.shape
            X_test = test_features
            rowsX,colsX = X_test.shape
            Y_test = y_test

            classLabels_test = np.unique(Y_test) # different class labels on the dataset
            classNum_test = len(classLabels_test) # number of classes
            print "Number of classes:"+str(classNum_test)

            classes_in_integers_test = np.zeros((rowsX))
            for i in range(rowsX):
                for j in range(classNum):
                    if classLabels_test[j]==Y_test[i]:
                        classes_in_integers_test[i] = j+1

            y_test = classes_in_integers_test
            pred_test = forest.predict(test_features)

            # testing score
            score = metrics.accuracy_score(y_test, pred_test)
            #score = metrics.f1_score(y_test, pred_test, pos_label=list(set(y_test)))
            acc = "Accuracy in testing set:"+str(score)
            print acc
            mac = "Macro test:"+str(metrics.precision_recall_fscore_support(y_test, pred_test, average='macro'))
            print mac
            mic = "Micro test:"+str(metrics.precision_recall_fscore_support(y_test, pred_test, average='micro'))
            print mic 

            met = metrics.classification_report(y_test, pred_test, target_names=classLabels_test, digits=4)
            print met

            text_file.write("Feature reduction:"+str(feature_reduction)+"\n")
            text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+"\n"+met)
            text_file.close()