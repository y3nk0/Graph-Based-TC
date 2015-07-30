from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn import svm
import time
import string
import math

b = 0.20
bag_of_words = "MY TF-IDF"
idf_bool = False
binary_par = False
ngrams_par = 1
classifier_par = "svm"
print "idf:"+str(idf_bool)+" classifier:"+classifier_par

## Open the file with read only permit
f = open('../../data/webkb/webkb-train-stemmed.txt', "r")

## use readlines to read all lines in the file
## The variable "lines" is a list containing all lines
train = [x.strip('\n') for x in f.readlines()]

## close the file after reading the lines.
f.close()

print len(train)

# Get the number of documents
num_documents = len(train)

# Initialize an empty list to hold the clean-preprocessed documents
clean_train_documents = []
unique_words = []
y_train = []

# Loop over each document; create an index i that goes from 0 to the length
# of the document list 
number_of_instances_per_class = dict()
for i in xrange( 0, num_documents ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    line = train[i].split('\t')
    y_train.append(line[0])
    for n, w in enumerate( line[1].split(' ') ):
        if w not in unique_words:
            unique_words.append(w)
    clean_train_documents.append( line[1] )

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

if bag_of_words=="vectorizer":

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    #train_data_features = vectorizer.fit_transform(clean_train_documents)
    # tfidf_vect = TfidfVectorizer(analyzer = "word", tokenizer = None,lowercase= True, max_df=1.0, min_df=3, max_features=None, norm=None, binary=binary_par, ngram_range=(1,ngrams_par), use_idf=idf_bool)
    tfidf_vect = TfidfVectorizer(analyzer = "word", tokenizer = None,lowercase= True, min_df=1, max_features=None, norm=None, binary=True, ngram_range=(1,ngrams_par), use_idf=False)

    #print train_data_features.shape
    features = tfidf_vect.fit_transform(clean_train_documents)

else:
    # MY TF-IDF

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

        for k, word in enumerate(wordList2):
            if word not in found_unique_words:
                found_unique_words.append(word)
                if word not in term_num_docs:
                    term_num_docs[word] = 1
                else:
                    term_num_docs[word] += 1


    avgLen = float(totalLen)/num_documents
    print "Average document length:"+str(avgLen)
    idf_col = {}
    for x in term_num_docs:
        idf_col[x] = math.log10((float(num_documents)+1.0) / (term_num_docs[x]))

    for i in range( 0,num_documents ):
        tf = dict()
        wordList1 = clean_train_documents[i].split(None)
        wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
        docLen = len(wordList2)

        for k, word in enumerate(wordList2):
            tf[word] = wordList2.count(word)

        for k, g in enumerate(tf):
            # Degree centrality (local feature)
            if g in unique_words:
                #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                tf_g = 1+math.log(1+math.log(tf[g]))
                if idf_bool:
                    features[i,unique_words.index(g)] = (float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))) * idf_col[g]
                else:
                    features[i,unique_words.index(g)] = float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))

print "Training the classifier..."
start = time.time()

# Initialize a Random Forest classifier with 100 trees
#clf = RandomForestClassifier(n_estimators = 100) 
#clf = AdaBoostClassifier(n_estimators=100)
# clf = svm.SVC(kernel="linear",probability=True)
if classifier_par=="svm":
    clf = svm.LinearSVC(loss="hinge")
elif classifier_par=="log":
    clf = SGDClassifier(loss="log")

rowsX,colsX = features.shape
Y = y_train

classLabels = np.unique(Y) # different class labels on the dataset
classNum = len(classLabels) # number of classes
print "Number of classes:"+str(classNum)

classes_in_integers = np.zeros((rowsX))
for i in range(rowsX):
	for j in range(classNum):
		if classLabels[j]==Y[i]:
			classes_in_integers[i] = j+1

y = classes_in_integers
print "y.shape:"+str(y.shape)
# cv = StratifiedKFold(y, n_folds=5)

# mean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100) 


# conf_matrix=np.zeros((2,2))

# for i, (train, test) in enumerate(cv):
# 	print str(i)
# 	# Binarize the output

# 	y_bin = label_binarize(y, classes=[1,2,3,4,5,6,7,8])
# 	print y_bin.shape
# 	n_classes = y_bin.shape[1]

# 	predictedLabels = clf.fit(X[train], y[train]).predict(X[test]) 
# 	predictedLabels_bin = label_binarize(predictedLabels, classes=[1,2,3,4,5,6,7,8])

# 	# Compute ROC curve and ROC area for each class
# 	fpr = dict()
# 	tpr = dict() 
# 	roc_auc = dict()
# 	for i in range(n_classes):
# 		#predictedLabels_bin = label_binarize(probas_, classes=[])
# 		y_class = y_bin[:,i]
# 		probas_ = clf.fit(X[train], y_class[train]).predict_proba(X[test])
# 		fpr[i], tpr[i], _ = roc_curve(y_class[test], probas_[:, 1])
# 		roc_auc[i] = auc(fpr[i], tpr[i])

# 	# Compute micro-average ROC curve and ROC area
# 	fpr["micro"], tpr["micro"], _ = roc_curve(y_bin[test].ravel(), predictedLabels_bin.ravel())
# 	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

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

if bag_of_words=="vectorizer":
    text_file = open("vectorizer_"+classifier_par+"_output_tf_idf_"+str(idf_bool)+".txt", "w")
else:
    text_file = open("myTfIdf_"+classifier_par+"_output_tf_idf_"+str(idf_bool)+".txt", "w")

# # Sort the coef_ as per feature weights and select largest 20 of them
# # 2 shows that we are considering the third class
# for counter, classLabel in enumerate(classLabels):
#     inds = np.argsort(forest.coef_[counter, :])[-20:]
#     # Now, just iterate over all these indices and get the corresponding
#     # feature names
#     text_file.write("Class:"+classLabel)
#     for i in inds:
#         text_file.write(","+unique_words[i])
#     text_file.write("\n\n")

# training score
score = metrics.accuracy_score(classes_in_integers, pred_train)
#score = metrics.f1_score(y_test, pred_test, pos_label=list(set(y_test)))
acc = "Accuracy in training set:"+str(score)
print acc
mac = "Macro:"+str(metrics.precision_recall_fscore_support(classes_in_integers, pred_train, average='macro'))
print mac
mic = "Micro:"+str(metrics.precision_recall_fscore_support(classes_in_integers, pred_train, average='micro'))
print mic

tp = np.zeros(classNum)
fn = np.zeros(classNum)
fp = np.zeros(classNum)
tn = np.zeros(classNum)
for j in range(classNum):
    for i in range(rowsX):
        if y[i]==j:
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

met = metrics.classification_report(classes_in_integers, pred_train, target_names=classLabels, digits=4)
print met

text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+pr_micro_str+"\n"+rec_micro_str+"\n"+f1_score_micro_str+"\n"+met)

end = time.time()
elapsed = end - start
print "Total time:"+str(elapsed)

## Testing set
## Open the file with read only permit
f = open('../../data/webkb/webkb-test-stemmed.txt', "r")

## use readlines to read all lines in the file
## The variable "lines" is a list containing all lines
test = [x.strip('\n') for x in f.readlines()]

## close the file after reading the lines.
f.close()

print len(test)

# Get the number of documents based on the dataframe column size
num_test_documents = len(test)

# Initialize an empty list to hold the clean-preprocessed documents
clean_test_documents = []
y_test = []

# Loop over each document; create an index i that goes from 0 to the length
# of the document list
number_of_instances_per_class_test = dict()
for i in xrange( 0, num_test_documents ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    line = test[i].split('\t')
    y_test.append(line[0])
    clean_test_documents.append( line[1] )

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

if bag_of_words=="vectorizer":
    # Careful: here we use transform and not fit_transform
    features_test = tfidf_vect.transform(clean_test_documents)

else:
    features_test = np.zeros((count,len(unique_words)))
    term_num_docs_test = {}

    totalLen = 0
    for i in range( 0,count ):
        #dG = nx.Graph()
        found_unique_words_test = []
        wordList1 = clean_test_documents[i].split(None)
        wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]

        docLen = len(wordList2)
        totalLen += docLen

        for k, word in enumerate(wordList2):
            if word not in found_unique_words_test:
                found_unique_words_test.append(word)
                if word not in term_num_docs_test:
                    term_num_docs_test[word] = 1
                else:
                    term_num_docs_test[word] += 1

    # We use the avgLen given by the training set
    # avgLen = float(totalLen)/count
    print "Average document length in test set:"+str(avgLen)
    idf_col_test = {}
    for x in term_num_docs_test:
        idf_col_test[x] = math.log10((float(num_test_documents)+1.0) / (term_num_docs_test[x]))

    for i in range( 0,count ):

        tf_test = dict()
        wordList1 = clean_test_documents[i].split(None)
        wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
        docLen = len(wordList2)

        for k, word in enumerate(wordList2):
            tf_test[word] = wordList2.count(word)

        for k, g in enumerate(tf_test):
            # Degree centrality (local feature)
            if g in unique_words:
                #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                tf_g = 1+math.log(1+math.log(tf_test[g]))
                if idf_bool:
                    features_test[i,unique_words.index(g)] = (float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))) * idf_col[g]
                else:
                    features_test[i,unique_words.index(g)] = float(tf_g)/(1-b+(b*(float(docLen)/avgLen)))

# Numpy arrays are easy to work with, so convert the result to an 
# array
# features_test = X_test_tfidf.toarray()
print features_test.shape

rowsX,colsX = features_test.shape
Y_test = y_test

classLabels_test = np.unique(Y_test) # different class labels on the dataset
classNum_test = len(classLabels_test) # number of classes
print "Number of classes:"+str(classNum_test)

classes_in_integers_test = np.zeros((rowsX))
for i in range(rowsX):
	for j in range(classNum_test):
		if classLabels_test[j]==Y_test[i]:
			classes_in_integers_test[i] = j+1

y_test = classes_in_integers_test

pred_test = forest.predict(features_test)

# testing score
score = metrics.accuracy_score(y_test, pred_test)
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

met = metrics.classification_report(y_test, pred_test, target_names=classLabels_test, digits=4)
print met

text_file.write(acc+"\n"+mac+"\n"+mic+"\n"+pr_micro_str+"\n"+rec_micro_str+"\n"+f1_score_micro_str+"\n"+met)
text_file.close()