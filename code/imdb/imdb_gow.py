from scipy.optimize.optimize import fmin_cg, fmin_bfgs, fmin
import numpy as np
import pandas as pd
import networkx as nx
import string
from sys import maxint
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from lxml import etree
from nltk.stem import *
from nltk.corpus import stopwords
import matplotlib.pyplot as pl
import re
import os.path
import math
from sklearn import svm
import time
import sys
sys.path.append('../../code')
from MyGraph import createGraphFeatures
import glob

def parseXmlStopStemRem(raw_data,unigrams,bigrams,test_bool):
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

	reviews = []
	print len(raw_data)
	for i,j in enumerate(raw_data):
		s = j
		#remove punctuation and split into seperate words
		s = re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)

		stemmed_text = ""
		prev_unigram = ""
		for word in s:
			w = word.lower()
			if w not in stop:
				sw = stemmer.stem(w)
				stemmed_text += sw + " "
				if test_bool==False:
					if sw not in unigrams:
						unigrams.append(sw)
				# if (prev_unigram!=""):
				# 	bi = prev_unigram + " " + sw
				# 	if bi not in bigrams:
				# 		bigrams.append(bi)

				prev_unigram = sw
				
		reviews.append(stemmed_text)

	print "Reviews length:", len(reviews)
	return reviews, unigrams, bigrams

sliding_window = 3
b = 0.003
idf_pars = ["tf-icw"]
kcore_par = "A0"
kcore_par_int = 1

# centrality_pars = ["core_number","closeness_centrality","betweenness_centrality","in_degree_centrality","out_degree_centrality","pagerank_centrality","closeness_centrality_directed","betweenness_centrality_directed"]

clean_train_documents = []
clean_test_documents = []
y_train = []
y_test = []
unique_words = []
bigrams = []

path_pos = '../../data/aclImdb/train/pos/*.txt'

files=glob.glob(path_pos)
for file in files:
	f=open(file, 'r')
	clean_train_documents.append( f.read() )
	y_train.append(1)
	f.close() 

path_neg = '../../data/aclImdb/train/neg/*.txt'

files=glob.glob(path_neg)
for file in files:
	f=open(file, 'r')
	clean_train_documents.append( f.read() )
	y_train.append(0)
	f.close() 

path_pos = '../../data/aclImdb/test/pos/*.txt'

files=glob.glob(path_pos) 
for file in files:
	f=open(file, 'r')
	clean_test_documents.append( f.read() )
	y_test.append(1)
	f.close() 

path_neg = '../../data/aclImdb/test/neg/*.txt'

files=glob.glob(path_neg) 
for file in files:
	f=open(file, 'r')
	clean_test_documents.append( f.read() )
	y_test.append(0)
	f.close()

# Get the number of documents based on the dataframe column size
clean_train_documents,unique_words,bigrams = parseXmlStopStemRem(clean_train_documents,unique_words,bigrams,False)

centrality_pars = ["degree_centrality"]

for idf_par in idf_pars:
	for centrality_par in centrality_pars:

		centrality_col_par = "degree_centrality"
		print "idf_par:"+idf_par
		print "centrality_par:"+centrality_par
		print "centrality_col_par:"+centrality_col_par
		print "kcore_par:"+kcore_par
			
		if idf_par=="no":
			# pl.savefig("plot_tw_idf_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+".pdf", bbox_inches='tight')
			text_file = open("output_tw_idf_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+"_kcore_par_"+kcore_par+"_.txt", "w")
		elif idf_par=="tf-icw":
			# pl.savefig("plot_tw_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+".pdf", bbox_inches='tight')
			text_file = open("output_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+"_kcore_par_"+kcore_par+".txt", "w")
		elif idf_par=="idf":
			# pl.savefig("plot_tw_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+".pdf", bbox_inches='tight')
			text_file = open("output_tw_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+"_kcore_par_"+kcore_par+".txt", "w")
		elif idf_par=="icw" or idf_par=="icw+idf":
			# pl.savefig("plot_tw_icw_centr_"+centrality_par+"_centrcol_"+centrality_col_par+"_sliding_"+str(sliding_window)+".pdf", bbox_inches='tight')
			text_file = open("output_tw_"+idf_par+"_centr_"+centrality_par+"_centrcol_"+centrality_col_par+"_sliding_"+str(sliding_window)+"_kcore_par_"+kcore_par+".txt", "w")

		# Get the number of documents based on the dataframe column size
		num_documents = len(clean_train_documents)
		print "Number of documents:"+str(num_documents)

		idfs = {}
		icws = {}
		dGcol_nodes = {}
		max_core_col = []
		max_core_feat = []
		feature_reduction = 0.0
		avgLen = 0.0

		features, idfs_learned,icws_learned,collection_count_nodes, collection_count_edges, dGcol_nodes,max_core_col,feature_reduction, max_core_feat,avgLen = createGraphFeatures(num_documents,clean_train_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,True,idfs,icws,kcore_par,dGcol_nodes,max_core_col,kcore_par_int,max_core_feat,feature_reduction,avgLen)

		classLabels = np.unique(y_train) # different class labels on the dataset
		classNum = len(classLabels) # number of classes
		print "Number of classes:"+str(classNum)

		# Careful: here we use transform and not fit_transform
		# X_test_tfidf = tfidf_vect.transform(clean_test_documents)

		print "Training the classifier..."
		start = time.time()

		# Run classifier
		# clf = svm.SVC(kernel='linear', probability=True)
		clf = svm.LinearSVC(loss="hinge")

		# This may take a few minutes to run
		forest = clf.fit(features, y_train)

		# Testing data
		# Get the number of documents based on the dataframe column size
		clean_test_documents,unique_words,bigrams = parseXmlStopStemRem(clean_test_documents,unique_words,bigrams,True)
		num_test_documents = len(clean_test_documents)

		features_test,idfs,icws,collection_count_nodes,collection_count_edges,dGcol_nodes,max_core_col,feature_reduction,max_core_feat,avgLen = createGraphFeatures(num_test_documents,clean_test_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,False,idfs_learned,icws_learned,kcore_par,dGcol_nodes,max_core_col,kcore_par_int,max_core_feat,feature_reduction,avgLen)

		pred_test = forest.predict(features_test)

		# testing score
		acc = "Accuracy in testing set:"+str(accuracy_score(y_test, pred_test))
		print acc
		mic = "Metrics:"+str(metrics.precision_recall_fscore_support(y_test, pred_test, average='binary'))
		print mic
		met = metrics.classification_report(y_test, pred_test, digits=4)
		print met

		text_file.write("Features shape:"+str(features.shape)+"\n")
		text_file.write("Feature reduction:"+str(feature_reduction)+"\n")
		text_file.write(acc+"\n"+mic+"\n"+met)
		end = time.time()
		elapsed = end - start
		print "Total time:"+str(elapsed)

		text_file.close()
