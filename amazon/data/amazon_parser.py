import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from scipy.optimize.optimize import fmin_cg, fmin_bfgs, fmin
import numpy as np
import pandas as pd
import networkx as nx
import string
from sys import maxint
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from lxml import etree
from nltk.stem import *
from nltk.corpus import stopwords
import re
import os.path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import csv

import math
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit

def parseXmlStopStemRem(path,unigrams,bigrams):
	stemmer = PorterStemmer()

	parser = etree.XMLParser(recover=True)
	reviews = []

	doc = etree.parse(path,parser=parser)
	# print doc.xpath('count(//review)')
	root = doc.getroot()
	for element in root.iter("review_text"):
		s = element.text
		s = re.sub('\s+', ' ', s)
		s = s.split()
		#remove punctuation and split into seperate words
		#s = re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)
		stemmed_text = ""
		prev_unigram = ""
		for word in s:
			w = word
			sw = w
			stemmed_text += sw + " "
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

unigrams = []
bigrams = []
unique_words = []
paths = ['books','dvd','electronics','kitchen_&_housewares']
for path in paths:
	print path
	clean_pos_re, unigrams, bigrams = parseXmlStopStemRem(path+'/positive.review',unigrams,bigrams)
	clean_neg_re, unigrams, bigrams = parseXmlStopStemRem(path+'/negative.review',unigrams,bigrams)

	new_col_1 = np.ones((len(clean_pos_re),1),dtype=np.int)

	# clean_pos_reviews = np.concatenate((new_col_1,np.reshape(clean_pos_re,(len(clean_pos_re),1))), 1)
	# print "clean_pos_reviews.shape:"+str(clean_pos_reviews.shape)

	new_col_0 = np.zeros((len(clean_neg_re),1),dtype=np.int)
	# clean_neg_reviews = np.concatenate((new_col_0,np.reshape(clean_neg_re,(len(clean_neg_re),1))), 1)

	y_classes = np.concatenate((new_col_0,new_col_1),0)
	raw_data = np.concatenate((clean_pos_re,clean_neg_re),0)

	raw_data = np.array(raw_data)
	raw_target = np.array(y_classes)

	sss = StratifiedShuffleSplit(raw_target, 1, test_size=0.2,random_state=0)

	for train_index, test_index in sss:
		# print("TRAIN:", train_index, "DEV:", dev_index)
		data_train, data_test = raw_data[train_index], raw_data[test_index]
		y_train, y_test = raw_target[train_index], raw_target[test_index]

	np.savetxt(path+'/data_train.txt',data_train,fmt='%s')
	np.savetxt(path+'/data_test.txt',data_test,fmt='%s')
	np.savetxt(path+'/y_train.txt',y_train,fmt='%s')
	np.savetxt(path+'/y_test.txt',y_test,fmt='%s')
