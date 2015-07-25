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

def parseXmlStopStemRem(raw_data,unique_words,bigrams,trigrams,test_bool):
    stemmer = PorterStemmer()

    bigrams_freq = {}

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
        prev_bigram = ""
        bi = ""
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
                            #     if bi not in bigrams_freq:
                            #         bigrams_freq[bi] = 0
                            #     bigrams_freq[bi] += 1

                            # if prev_bigram!="":
                            #     tri = prev_bigram + " " + sw
                            #     if tri not in trigrams:
                            #         trigrams.append(tri)

                        prev_unigram = sw
                        prev_bigram = bi
        
        # print stemmed_text
        reviews.append(stemmed_text)

    return reviews, unique_words, bigrams, trigrams, bigrams_freq

# get complete set of terms that consists of n requested grams
def parseXmlStopStemRem_Ngrams(raw_data,unique_words,ngrams_par,test_bool):
    stemmer = PorterStemmer()

    bigrams_freq = {}

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
        prev_gram = ""
        gram = ""
        # for word in s:
        if len(s)>1:
            for k, word in enumerate(s):

                w = word.lower()
                if re.match("^[a-zA-Z]*$", w) and not re.match("^[~!@#$%^&*()_+{}':;]+$",w):
                    if w not in stop:
                        if len(w)>=3:
                            sw = stemmer.stem(w)
                            stemmed_text += sw + " "
                            if test_bool==False:
                                if sw not in unique_words:
                                    unique_words.append(sw)
                
                # for j in xrange(0,ngrams_par):
                #     if k+j<len(s):
                #         n_word = s[k + j]

                #         w = n_word.lower()
                #         if re.match("^[a-zA-Z]*$", w) and not re.match("^[~!@#$%^&*()_+{}':;]+$",w):
                #             if w not in stop:
                #                 if len(w)>=3:
                #                     sw = stemmer.stem(w)
                #                     if test_bool==False:
                #                         if prev_gram!="":
                #                             gram = prev_gram + " " + sw
                #                         else:
                #                             gram = sw

                #                         if gram not in terms:
                #                             terms.append(gram)

                #                     prev_gram = gram
        
        # print stemmed_text
        reviews.append(stemmed_text)

    return reviews, unique_words

def find_ngrams(input_list,n):
    output = []
    for i in range(len(input_list)-n+1):
        output.append(" ".join(input_list[i:i+n]))
    return output

def find_bigrams(input_list):
    bigram_list = []
    for i in range(len(input_list)-1):
        bigram_list.append(input_list[i]+" "+input_list[i+1])
    return bigram_list

b = 0.20
unique_words = []
bigrams = []
trigrams = []
terms = []

ngrams_par = 6
idf_bool = False
freq_par = "not-binary"

bag_of_words = "MY TF-IDF"

# if (not os.path.isfile("20newsgroup_cleaned_data.txt")):
# categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
raw_data = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
categories = list(raw_data.target_names)
train_data = raw_data.data
print "Length of train data:"+str(len(train_data))
train,unique_words,bigrams,trigrams, bigrams_freq = parseXmlStopStemRem(train_data,unique_words,bigrams,trigrams,False)

# create a n-gram parser

# train, terms = parseXmlStopStemRem_Ngrams(train_data,terms,ngrams_par,False)

y_train = raw_data.target

# Get the number of documents based on the dataframe column size
num_documents = len(y_train)

# Initialize an empty list to hold the clean-preprocessed documents
clean_train_documents = train

# Loop over each document; create an index i that goes from 0 to the length
# of the document list 
number_of_instances_per_class = dict()
for i in xrange( 0, num_documents ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    # for n, w in enumerate( train_data[i].split() ):
    #     if w not in unique_words:
    #         unique_words.append(w)
    # clean_train_documents.append( train_data[i] )
    if y_train[i] not in number_of_instances_per_class:
    	number_of_instances_per_class[y_train[i]] = 1
    else:
    	number_of_instances_per_class[y_train[i]] += 1

print number_of_instances_per_class
count=0
for value in number_of_instances_per_class.values():   # Iterate via values
  	count += value

print "Number of TOTAL instances:"+str(count) 

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool. 
print "Creating the bag of words..."

if bag_of_words!="MY TF-IDF":

    tfidf_vect = TfidfVectorizer(analyzer = "word", tokenizer = None,lowercase= True, min_df=1, max_features=None, norm=None, binary=True, ngram_range=(1,ngrams_par), use_idf=False)
    #print train_data_features.shape
    features = tfidf_vect.fit_transform(clean_train_documents)

    # # Numpy arrays are easy to work with, so convert the result to an 
    # # array
    # X_train_tfidf = X_train_tfidf.toarray()
    # print X_train_tfidf.shape
    # features = X_train_tfidf

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
        wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]

        docLen = len(wordList2)
        totalLen += docLen

        bigram=""
        prev_unigram = ""
        prev_bigram = ""
        # for k, word in enumerate(wordList2):
        for k, word in enumerate(terms):
            if word in clean_train_documents[i]:
            # if ngrams_par=="trigrams":
            #     if prev_bigram!="":
            #         trigram = prev_bigram + " " + word

            #         if trigram not in found_unique_words:
            #             found_unique_words.append(trigram)
            #             if trigram not in term_num_docs:
            #                 term_num_docs[trigram] = 1
            #             else:
            #                 term_num_docs[trigram] += 1

            # if ngrams_par=="bigrams":
            #     if prev_unigram!="":
            #         bigram = prev_unigram + " " + word

            #         if bigram not in found_unique_words:
            #             found_unique_words.append(bigram)
            #             if bigram not in term_num_docs:
            #                 term_num_docs[bigram] = 1
            #             else:
            #                 term_num_docs[bigram] += 1

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
        idf_col[x] = math.log10((float(num_documents)+1.0) / (term_num_docs[x]))

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

        for k, g in enumerate(tf):
            # Degree centrality (local feature)
           
            if g in unique_words:
                #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                tf_g = 1+math.log(1+math.log(tf[g]))
                # tf_g = tf[g]
                if idf_bool:
                    features[i,unique_words.index(g)] = (float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))) * idf_col[g]
                    # features[i,terms.index(g)] = float(tf[g]) * idf_col[g]
                else:
                    features[i,unique_words.index(g)] = float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))
                    # features[i,terms.index(g)] = float(tf[g])

            if ngrams_par=="bigrams" or ngrams_par=="trigrams":
                if g in bigrams:
                    #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                    tf_g = 1+math.log(1+math.log(tf[g]))
                    if idf_bool:
                        features[i,blen+bigrams.index(g)] = (float(tf_g)/sum_freq_bigrams) * idf_col[g]
                    else:
                        features[i,blen+bigrams.index(g)] = float(tf_g)/sum_freq_bigrams
            
            if ngrams_par=="trigrams":
                if g in trigrams:
                    #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                    tf_g = 1+math.log(1+math.log(tf[g]))
                    if idf_bool:
                        features[i,tlen+blen+trigrams.index(g)] = (float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))) * idf_col[g]
                    else:
                        features[i,tlen+blen+trigrams.index(g)] = float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))


print "Training the classifier..."
start = time.time()

# Initialize a Random Forest classifier with 100 trees
#clf = RandomForestClassifier(n_estimators = 100) 
#clf = AdaBoostClassifier(n_estimators=100)
# clf = svm.SVC(kernel="linear",probability=True)
clf = svm.LinearSVC(loss="hinge")

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


# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = clf.fit( features, y )

text_file = open("binary_output_tf_idf_"+str(idf_bool)+"_"+str(ngrams_par)+".txt","w")

# # training score
# pred_train = forest.predict(features)
# score = accuracy_score(y_train, pred_train)
# #score = metrics.f1_score(y_test, pred_test, pos_label=list(set(y_test)))
# acc = "Accuracy in training set:"+str(score)
# print acc
# mac = "Macro:"+str(metrics.precision_recall_fscore_support(y_train, pred_train, average='macro'))
# print mac
# mic = "Micro:"+str(metrics.precision_recall_fscore_support(y_train, pred_train, average='micro'))
# print mic

# tp = np.zeros(classNum)
# fn = np.zeros(classNum)
# fp = np.zeros(classNum) 
# tn = np.zeros(classNum)

# for j in range(classNum):
#     for i in range(rowsX):
#         if y_train[i]==j:
#             if pred_train[i]==j:
#                 tp[j] += 1
#             else:
#                 fn[j] += 1
#         else:
#             if pred_train[i]==j:
#                 fp[j] += 1
#             else:
#                 tn[j] += 1


# pr_micro = float(np.sum(tp))/np.sum(np.add(tp,fp))
# pr_micro_str = "Precision micro:"+str(pr_micro)
# print pr_micro_str
# rec_micro = float(np.sum(tp))/np.sum(np.add(tp,fn))
# rec_micro_str = "Recall micro:"+str(rec_micro)
# print rec_micro_str
# f1_score_micro = 2*(float(pr_micro*rec_micro)/(pr_micro+rec_micro))
# f1_score_micro_str = "f1-score micro:"+str(f1_score_micro)               
# print f1_score_micro_str

# met = metrics.classification_report(y_train, pred_train, target_names=categories, digits=4)
# print met

# text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+pr_micro_str+"\n"+rec_micro_str+"\n"+f1_score_micro_str+"\n"+met)

end = time.time()
elapsed = end - start
print "Total time:"+str(elapsed)

## Testing set
raw_test_data = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
test_data = raw_test_data.data
print "Length of test data:"+str(len(test_data))
test,unique_words,bigrams,trigrams, bigrams_freq = parseXmlStopStemRem(test_data,unique_words,bigrams,trigrams,True)
# test, terms = parseXmlStopStemRem_Ngrams(test_data,terms,ngrams_par,True)
y_test = raw_test_data.target

print "Length of test data:"+str(len(test_data))

# Get the number of documents based on the dataframe column size
num_test_documents = len(y_test)

# Initialize an empty list to hold the clean-preprocessed documents
clean_test_documents = test

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

if bag_of_words!="MY TF-IDF":
    # # Careful: here we use transform and not fit_transform
    features_test = tfidf_vect.transform(clean_test_documents)

else:
    if ngrams_par==2:
        features_test = np.zeros((count,len(unique_words)+len(bigrams)))
    else:
        features_test = np.zeros((count,len(unique_words)))

    term_num_docs_test = {}

    totalLen = 0
    for i in range( 0,count ):
        #dG = nx.Graph()
        found_unique_words_test = []
        wordList1 = clean_test_documents[i].split(None)
        wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]

        docLen_test = len(wordList2)
        totalLen += docLen_test

        bigram=""
        prev_unigram = ""
        prev_bigram = ""
        for k, word in enumerate(wordList2):
        # for k, word in enumerate(terms):
            if word in clean_test_documents[i]:

            # if prev_bigram!="":
            #     trigram = prev_bigram + " " + word

            #     if trigram not in found_unique_words_test:
            #         found_unique_words_test.append(trigram)
            #         if trigram not in term_num_docs_test:
            #             term_num_docs_test[trigram] = 1
            #         else:
            #             term_num_docs_test[trigram] += 1

            # if prev_unigram!="":
            #     bigram = prev_unigram + " " + word

            #     if bigram not in found_unique_words_test:
            #         found_unique_words_test.append(bigram)
            #         if bigram not in term_num_docs_test:
            #             term_num_docs_test[bigram] = 1
            #         else:
            #             term_num_docs_test[bigram] += 1

                if word not in found_unique_words_test:
                    found_unique_words_test.append(word)
                    if word not in term_num_docs_test:
                        term_num_docs_test[word] = 1
                    else:
                        term_num_docs_test[word] += 1

            prev_unigram = word
            prev_bigram = bigram

    # avgLen = float(totalLen)/count
    print "Average document length in test set:"+str(avgLen)
    # idf_col_test = {}
    # for x in term_num_docs_test:
    #     idf_col_test[x] = math.log10((float(num_test_documents)+1.0) / (term_num_docs_test[x]))

    for i in range( 0,count ):

        tf_test = dict()
        wordList1 = clean_test_documents[i].split(None)
        wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
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
                tf_g = 1+math.log(1+math.log(tf_test[g]))
                # tf_g = tf_test[g]
                if idf_bool:
                    features_test[i,unique_words.index(g)] = (float(tf_g)/(1-b+(b*(float(docLen_test)/avgLen)))) * idf_col[g]
                    # features_test[i,terms.index(g)] = float(tf_test[g]) * idf_col_test[g]
                else:
                    features_test[i,unique_words.index(g)] = float(tf_g)/(1-b+(b*(float(docLen_test)/avgLen)))
                    # features_test[i,terms.index(g)] = float(tf_test[g])

            if ngrams_par==2 or ngrams_par==3:
                if g in bigrams:
                    #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                    tf_g = 1+math.log(1+math.log(tf_test[g]))
                    if idf_bool:
                        features_test[i,blen+bigrams.index(g)] = (float(tf_g)/(sum_freq_bigrams)) * idf_col[g]
                    else:
                        features_test[i,blen+bigrams.index(g)] = float(tf_g)/sum_freq_bigrams

            if ngrams_par==3:
                if g in trigrams:
                    #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                    tf_g = 1+math.log(1+math.log(tf_test[g]))
                    if idf_bool:
                        features_test[i,tlen+blen+bigrams.index(g)] = (float(tf_g)/(1-b+(b*(float(docLen_test)/avgLen)))) * idf_col[g]
                    else:
                        features_test[i,tlen+blen+bigrams.index(g)] = float(tf_g)/(1-b+(b*(float(docLen_test)/avgLen)))

# # Numpy arrays are easy to work with, so convert the result to an 
# # array
# # X_test_tfidf = X_test_tfidf.toarray()
# # print X_test_tfidf.shape

rowsX,colsX = features_test.shape
print features_test.shape
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

text_file.write("Features shape:"+str(features.shape)+"\n")
text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+pr_micro_str+"\n"+rec_micro_str+"\n"+f1_score_micro_str+"\n"+met)
text_file.close()