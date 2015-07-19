from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as pl
from sklearn.metrics import accuracy_score
from sklearn import svm
import time
from lxml import etree
from nltk.stem import *
from nltk.corpus import stopwords
import re
import os.path
import string
import math
import glob

def parseXmlStopStemRem(raw_data,unigrams,test_bool):
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
	return reviews, unigrams

b = 0.20
unigrams = []
bigrams = []

text_file = open("output_tf_icf.txt", "w")
start = time.time()

unique_words = []

clean_train_documents = []
clean_test_documents = []
y_train = []
y_test = []

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
clean_train_documents,unique_words = parseXmlStopStemRem(clean_train_documents,unique_words,False)
num_documents = len(clean_train_documents)

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool. 
print "Creating the bag of words..."
# vectorizer = CountVectorizer(analyzer = "word")

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
#train_data_features = vectorizer.fit_transform(clean_train_documents)
# tfidf_vect = TfidfVectorizer(analyzer = "word", tokenizer = None,lowercase= True, max_df=1.0, min_df=3, max_features=None, norm=None, binary=True, ngram_range=(1,ngrams_par), use_idf=False)

# #print train_data_features.shape
# features = tfidf_vect.fit_transform(clean_train_documents)
# # features = X_train_tfidf.toarray()

# MY TF-IDF
features = np.zeros((num_documents,len(unique_words)))
term_num_docs = {}
icf_col = {}

totalLen = 0
for i in range( 0,num_documents ):
    #dG = nx.Graph()
	found_unique_words = []
	wordList1 = clean_train_documents[i].split(None)
	wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]

	docLen = len(wordList2)
	totalLen += docLen

	for k, word in enumerate(wordList2):
		if word not in found_unique_words:
			found_unique_words.append(word)
			if word not in term_num_docs:
				term_num_docs[word] = 1
			else:
				term_num_docs[word] += 1

			if word not in icf_col:
				icf_col[word] = wordList2.count(word)
			else:
				icf_col[word] += wordList2.count(word)

avgLen = float(totalLen)/num_documents
print "Average document length:"+str(avgLen)
idf_col = {}
for x in term_num_docs:
	idf_col[x] = math.log10((float(num_documents)+1.0) / (term_num_docs[x]))
	icf_col[x] = math.log10((float(num_documents)+1.0) / (icf_col[x]))

for i in range( 0,num_documents ):
	tf = dict()
	wordList1 = clean_train_documents[i].split(None)
	wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
	docLen = len(wordList2)

	for k, word in enumerate(wordList2):
		tf[word] = wordList2.count(word)

	for k, g in enumerate(tf):
		if g in unique_words:
            #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
			tf_g = 1+math.log(1+math.log(tf[g]))
			features[i,unique_words.index(g)] = (float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))) * icf_col[g]

print "Training the classifier..."

rowsX,colsX = features.shape
print features.shape

classLabels = np.unique(y_train) # different class labels on the dataset
classNum = len(classLabels) # number of classes
print "Number of classes:"+str(classNum)

# Run classifier
# clf = svm.SVC(kernel='linear', probability=True)
clf = svm.LinearSVC(loss="hinge")

# Get the number of documents based on the dataframe column size
clean_test_documents,unique_words = parseXmlStopStemRem(clean_test_documents,unique_words,True)
num_documents_test = len(clean_test_documents)

# # TF-IDF vectorizer for test
# X_test_tfidf = tfidf_vect.transform(clean_test_documents)
# features_test = X_test_tfidf.toarray()
# print features_test.shape

# MY TF-IDF for test
features_test = np.zeros((num_documents_test,len(unique_words)))

for i in range( 0,num_documents_test):
	tf_test = dict()
	wordList1 = clean_test_documents[i].split(None)
	wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
	docLen = len(wordList2)

	for k, word in enumerate(wordList2):
		tf_test[word] = wordList2.count(word)

	for k, g in enumerate(tf_test):
		if g in unique_words:
            #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
			tf_g = 1+math.log(1+math.log(tf_test[g]))
			features_test[i,unique_words.index(g)] = (float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))) * icf_col[g]

# This may take a few minutes to run
forest = clf.fit( features, y_train)

pred_test = forest.predict(features_test)

# tp = np.zeros(classNum)
# fn = np.zeros(classNum)
# fp = np.zeros(classNum)
# tn = np.zeros(classNum)
# for j in range(classNum):
# 	for i in range(rowsX):
# 		if y[i]==j:
# 			if pred_train[i]==j:
# 				tp[j] += 1
# 			else:
# 				fn[j] += 1
# 		else:
# 			if pred_train[i]==j:
# 				fp[j] += 1
# 			else:
# 				tn[j] += 1

# pr_micro = float(np.sum(tp))/np.sum(np.add(tp,fp))
# pr_micro_str = "Precision micro:"+str(pr_micro)
# print pr_micro_str
# rec_micro = float(np.sum(tp))/np.sum(np.add(tp,fn))
# rec_micro_str = "Recall micro:"+str(rec_micro)
# print rec_micro_str
# f1_score_micro = 2*(float(pr_micro*rec_micro)/(pr_micro+rec_micro))
# f1_score_micro_str = "f1-score micro:"+str(f1_score_micro)
# print f1_score_micro_str

acc = "Accuracy in testing set:"+str(accuracy_score(y_test, pred_test))
print acc
mic = "Metrics:"+str(metrics.precision_recall_fscore_support(y_test, pred_test, average='binary'))
print mic
met = metrics.classification_report(y_test, pred_test, digits=4)
print met

text_file.write("Features shape:"+str(features.shape)+"\n")
text_file.write(acc+"\n"+mic+"\n"+met)
end = time.time()
elapsed = end - start
print "Total time:"+str(elapsed)

text_file.close()
