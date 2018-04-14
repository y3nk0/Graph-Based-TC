from nltk.stem import *
from nltk.corpus import stopwords
import time
import re
import os.path
import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse import csr_matrix
# import sparse_dot_topn.sparse_dot_topn as ct

#removed 'no', 'nor', 'not', 'only',
stop = set(stopwords.words('english')) - set(['no', 'nor', 'not', 'only'])

# def get_matches_df(sparse_matrix, name_vector, top=100):
#     non_zeros = sparse_matrix.nonzero()
#
#     sparserows = non_zeros[0]
#     sparsecols = non_zeros[1]
#
#     if top:
#         nr_matches = top
#     else:
#         nr_matches = sparsecols.size
#
#     left_side = np.empty([nr_matches], dtype=object)
#     right_side = np.empty([nr_matches], dtype=object)
#     similairity = np.zeros(nr_matches)
#
#     for index in range(0, nr_matches):
#         left_side[index] = name_vector[sparserows[index]]
#         right_side[index] = name_vector[sparsecols[index]]
#         similairity[index] = sparse_matrix.data[index]
#
#     return pd.DataFrame({'left_side': left_side,
#                           'right_side': right_side,
#                            'similairity': similairity})
#
# def awesome_cossim_top(A, B, ntop, lower_bound=0):
#     # force A and B as a CSR matrix.
#     # If they have already been CSR, there is no overhead
#     A = A.tocsr()
#     B = B.tocsr()
#     M, _ = A.shape
#     _, N = B.shape
#
#     idx_dtype = np.int32
#
#     nnz_max = M*ntop
#
#     indptr = np.zeros(M+1, dtype=idx_dtype)
#     indices = np.zeros(nnz_max, dtype=idx_dtype)
#     data = np.zeros(nnz_max, dtype=A.dtype)
#
#     ct.sparse_dot_topn(
#         M, N, np.asarray(A.indptr, dtype=idx_dtype),
#         np.asarray(A.indices, dtype=idx_dtype),
#         A.data,
#         np.asarray(B.indptr, dtype=idx_dtype),
#         np.asarray(B.indices, dtype=idx_dtype),
#         B.data,
#         ntop,
#         lower_bound,
#         indptr, indices, data)
#
#     return csr_matrix((data,indices,indptr),shape=(M,N))

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000 == 0:
           print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs

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

def findVocab(raw_data):
    words_frequency = {}

    bigrams_freq = {}

    #stop = set(stopwords.words('english'))

    print "stop list:"+str(len(stop))

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
            # w = word.lower()
            w = word.strip()
            if re.match("^[a-zA-Z]*$", w) and not re.match("^[~!@#$%^&*()_+{}':;><\"]+$",w):
                if w.lower() not in stop:
                    if len(w)>=3:
                        if w not in words_frequency:
                            words_frequency[w] = 1
                        else:
                            words_frequency[w] = words_frequency[w] + 1

    unique_words = [k for (k,v) in words_frequency.items() if v>=1]

    return unique_words

def parseXmlStopStemRem(raw_data,unique_words,bigrams,trigrams,train_bool):
    bigrams_freq = {}

    #stop = set(stopwords.words('english'))

    reviews = []

    # for element in raw_data:
    for i,j in enumerate(raw_data):
        s = j
        #remove punctuation and split into seperate words
        s = re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)

        stemmed_text = ""

        for word in s:
            # w = word.lower()
            w = word
            if re.match("^[a-zA-Z]*$", w) and not re.match("^[~!@#$%^&*()_+{}':;><\"]+$",w):
                if w.lower() not in stop:
                    if len(w)>=3:
                        # sw = stemmer.stem(w)
                        sw = w
                        if sw in unique_words:
                            stemmed_text += sw + " "

        # print stemmed_text
        reviews.append(stemmed_text.strip())

    return reviews


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

def print_top10(feature_names, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    # coef stores the weights of each feature (in unique term), for each class
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label," ".join(feature_names[j] for j in top10)))

def print_bot10(feature_names, clf, class_labels):
    """Prints features with the lowest coefficient values, per class"""
    for i, class_label in enumerate(class_labels):
        bot10 = np.argsort(clf.coef_[i])[0:9]
        print("%s: %s" % (class_label," ".join(feature_names[j] for j in bot10)))
