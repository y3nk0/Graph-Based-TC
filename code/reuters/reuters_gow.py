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
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn import svm
from sklearn.metrics import accuracy_score
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

cols = ['class', 'text']
train = pd.read_csv("../../data/reuters/r8-train-stemmed.txt", sep="\t", header=None, names=cols)
# Get the number of documents based on the dataframe column size
num_documents = train.shape[0]

# Initialize an empty list to hold the clean-preprocessed documents
clean_train_documents = []
unique_words = []
bigrams = []

# Loop over each document; create an index i that goes from 0 to the length
# of the document list 
for i in xrange( 0, num_documents ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    for n, w in enumerate( train['text'][i].split() ):
        if w not in unique_words:
            unique_words.append( w )
    clean_train_documents.append( train['text'][i] )

print "Unique words:"+str(len(unique_words))

sliding_window = 2
b = 0.003
# idf_par = "idf"
idf_pars = ["icw"]
kcore_pars = ["A0"]

# centrality_pars = ["degree_centrality","closeness_centrality","betweenness_centrality","in_degree_centrality","out_degree_centrality","pagerank_centrality","closeness_centrality_directed","betweenness_centrality_directed"]
# centrality_pars = ["out_degree_centrality","pagerank_centrality","closeness_centrality_directed","betweenness_centrality_directed"]
centrality_pars = ["degree_centrality"]

for kcore_par in kcore_pars:
    for idf_par in idf_pars:
        for centrality_par in centrality_pars:

            print "sliding_window:"+str(sliding_window)
            print "idf:"+idf_par
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

            myfile = open("results_"+kcore_par+".txt","w")

            for kcore_par_int in range(1,2):

                idfs = {}
                icws = {}
                dGcol_nodes = {}
                max_core_col = []
                max_core_feat = []
                feature_reduction = 0.0
                avglen = 0.0

                features, idfs_learned,icws_learned,collection_count_nodes, collection_count_edges, dGcol_nodes,max_core_col,feature_reduction, max_core_feat,avglen = createGraphFeatures(num_documents,clean_train_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,True,idfs,icws,kcore_par,dGcol_nodes,max_core_col,kcore_par_int,max_core_feat,feature_reduction,avglen)

                print "Training the classifier..."
                start = time.time()

                # Initialize a Random Forest classifier with 100 trees
                #clf = RandomForestClassifier(n_estimators = 100) 
                # clf = svm.SVC(kernel="linear",probability=True)
                clf = svm.LinearSVC(loss="hinge")
                #clf = AdaBoostClassifier(n_estimators=100)

                X = features
                rowsX,colsX = X.shape
                Y = train['class']

                classLabels = np.unique(Y) # different class labels on the dataset
                classNum = len(classLabels) # number of classes
                print "Number of classes:"+str(classNum)

                classes_in_integers = np.zeros((rowsX))
                for i in range(rowsX):
                    for j in range(classNum):
                        if classLabels[j]==Y[i]:
                            classes_in_integers[i] = j+1

                y = classes_in_integers
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
                    text_file = open(path+"output_tw_idf_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+"_kcore_"+kcore_par+".txt", "w")
                elif idf_par=="tf-icw":
                    text_file = open(path+"output_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+"_kcore_"+kcore_par+".txt", "w")
                elif idf_par=="idf":
                    text_file = open(path+"output_tw_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+"_kcore_"+kcore_par+".txt", "w")
                elif idf_par=="icw" or idf_par=="icw+idf":
                    if kcore_par=="A1" or kcore_par=="A2" or kcore_par=="B1" or kcore_par=="B2" or kcore_par=="A0":
                        text_file = open(path+"output_tw_"+idf_par+"_centr_"+centrality_par+"_centrcol_"+centrality_col_par+"_sliding_"+str(sliding_window)+"_kcore_"+kcore_par+"_"+str(kcore_par_int)+"_UPDATED.txt", "w")
                    else:
                        text_file = open(path+"output_tw_"+idf_par+"_centr_"+centrality_par+"_centrcol_"+centrality_col_par+"_sliding_"+str(sliding_window)+"_"+kcore_par+"_UPDATED.txt", "w")

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

                text_file.write("LinearSVC_tw_"+idf_par+"_"+centrality_par+"_sliding_"+str(sliding_window)+"\n\n")
                text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+"\n"+met)

                end = time.time()
                elapsed = end - start
                print "Total time:"+str(elapsed)

                ## Testing set
                test = pd.read_csv("data/r8-test-stemmed.txt", sep="\t", header=None, names=cols)

                print test.shape

                # Get the number of documents based on the dataframe column size
                num_test_documents = test.shape[0]

                # Initialize an empty list to hold the clean-preprocessed documents
                clean_test_documents = []

                # Loop over each document; create an index i that goes from 0 to the length
                # of the document list 
                for i in xrange( 0, num_test_documents ):
                    # Call our function for each one, and add the result to the list of
                    # clean reviews
                    clean_test_documents.append( test['text'][i] )
                    # print train['text'][i]+'\n'

                test_features,idfs,icws,collection_count_nodes,collection_count_edges,dGcol_nodes,max_core_col,feature_reduction,max_core_feat,avglen = createGraphFeatures(num_test_documents,clean_test_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,False,idfs_learned,icws_learned,kcore_par,dGcol_nodes,max_core_col,kcore_par_int,max_core_feat,feature_reduction,avglen)
                print test_features.shape
                
                rowsX,colsX = test_features.shape
                Y_test = test['class']

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
                # plot_confusion_matrix(cm_normalized,classLabels, title='Normalized confusion matrix')

                # plt.show()

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

                myfile.write("Accuracy:"+str(score)+"\n")
                myfile.write("kcore_par_int:"+str(kcore_par_int)+"\n")
                myfile.write("Feature reduction:"+str(feature_reduction)+"\n\n")

            myfile.close()

