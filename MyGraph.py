#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
import string
from sys import maxint
import pandas as pd
import numpy as np
import time
import re
import os.path
import math
from scipy.stats import pearsonr,kendalltau
import matplotlib.pyplot as plt
import json

## GENSIM UTILITIES
from gensim.models import word2vec
from gensim.models import LsiModel
from gensim.models.wrappers import Wordrank
from gensim import corpora, models, similarities

from sklearn.feature_extraction.text import TfidfVectorizer

## NLTK UTILITIES
from nltk.corpus import wordnet as wn

## LOGGING
import logging

from itertools import combinations

import scipy
from multiprocessing import cpu_count, Process, Queue, Pool
from contextlib import closing

import collections

from scipy.sparse import csr_matrix,lil_matrix
model = word2vec.Word2Vec(size=300,min_count=1)

from library import *

import community

def getOnlyDataWord2VecModel(documents):

    global model

    # wr_path = 'wordrank' # path to Wordrank directory
    # out_dir = 'model' # name of output directory to save data to
    # data = '../webkb/data/my_WEBKB_train.txt' # sample corpus
    #
    # model = Wordrank.train(wr_path, data, out_dir, iter=11, dump_period=5)

    sentences = []
    for doc in documents:
        sentences.append(doc.split())

    # model.train(sentences)
    token_count = sum([len(sentence) for sentence in sentences])
    # print len(model.wv.vocab)
    #
    # ## use pretrained word vectors
    model.build_vocab(sentences)
    model.intersect_word2vec_format('/Users/konstantinosskianis/Documents/phd/w2v_distances/wmd/GoogleNews-vectors-negative300.bin',binary=True)
    # model.intersect_word2vec_format('../word2vec/glove_model.txt')
    # model.train(sentences,total_examples = token_count,epochs = model.iter)

    print len(model.wv.vocab)

    # Demo: Loads the newly created glove_model.txt into gensim API.
    # model= word2vec.Word2Vec.load_word2vec_format("../glove_model.txt",binary=False) #GloVe Model
    # model = word2vec.Word2Vec.load("word2vec/"+'text8.model')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def plot_degree_histogram(G):
    degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
    #print "Degree sequence", degree_sequence
    dmax=max(degree_sequence)

    plt.loglog(degree_sequence,'b-',marker='o')
    plt.title("Degree rank plot")
    plt.ylabel("degree")
    plt.xlabel("rank")

    plt.savefig("degree_histogram.png")
    plt.show()

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    idx = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        idx.append(range(int(last),int(last+avg)))
        last += avg

    return out,idx

#===============================================================================
# core_Size_Distribution(core_sequence)
#===============================================================================
def core_Size_Distribution(core_sequence):
    """
    core_Size_Distribution(core_sequence)

    The function takes as input the core sequence of a graph and returns the distribution
    of the cores' sizes (k-core size vs. k)
    """

    max_core_number = max(core_sequence.values()) #maximum core number

    # core_sizes: size of each core indexed by position
    (core_sizes, x, y) = plt.hist(core_sequence.values(), bins=max_core_number)

    print "Core sizes:", core_sizes, "\n"
    #print max_core_number

    # plot in log-log axis
    # x: core size, k    y: number of nodes in k-core
    plt.loglog(range(1, len(core_sizes)+1), core_sizes, 'o-', linewidth=2)
    plt.xlabel('Core Number, k')
    plt.ylabel('Size of Core')

    plt.show()

def splitGraphFeatures(documents,idx,idf_par,centrality_par,dGcol_nodes, idf_col,icw_col,sliding_window,unique_words,train_par,path):
    features = np.zeros((len(documents),len(unique_words)))
    #features = csr_matrix((len(documents),len(unique_words)))
    # features = lil_matrix((len(documents),len(unique_words)))

    if centrality_par=="weighted_degree_centrality" or centrality_par=="weighted_pagerank_centrality":
        tf_par = "word2vec"
        global model
    else:
        tf_par = "word2ve"

    if not train_par:
        path = path+"test_"


    for i,doc in enumerate(documents):

        ind = idx[i]

        if not os.path.exists(path+str(ind)+"_sliding_"+str(sliding_window)+"_"+str(tf_par)+"_graph.edgelist"):

            # print "Creating the graph of words for documents..."

            if centrality_par=="pagerank_centrality" or centrality_par=="in_degree_centrality" or centrality_par=="out_degree_centrality" or centrality_par=="closeness_centrality_directed" or centrality_par=="betweenness_centrality_directed" or centrality_par=="weighted_pagerank_centrality":
                dG = nx.DiGraph()
            else:
                dG = nx.Graph()

            wordList1 = doc.split(None)
            wordList2 = [string.rstrip(x, ',.!?;') for x in wordList1]
            docLen = len(wordList2)

            #if len(wordList2)>1:
            for k, word in enumerate(wordList2):
                for j in xrange(1,sliding_window):
                    try:
                        next_word = wordList2[k + j]

                        if not dG.has_node(word):
                            dG.add_node(word)
                            dG.node[word]['count'] = 1
                        else:
                            dG.node[word]['count'] += 1

                        if not dG.has_node(next_word):
                            dG.add_node(next_word)
                            dG.node[next_word]['count'] = 1
                        else:
                            dG.node[next_word]['count'] += 1

                        if not dG.has_edge(word, next_word):
                            dG.add_edge(word, next_word, weight = 1)
                            # dG.edge[word][next_word]['w2vec'] = 0.0001
                            if tf_par=="word2vec":

                                if word in model.wv.vocab and next_word in model.wv.vocab:
                                    dG.edge[word][next_word]['w2vec'] = model.wv.similarity(word,next_word)
                                    # dG.edge[word][next_word]['w2vec'] = np.linalg.norm(model[word]-model[next_word])


                        else:
                            dG.edge[word][next_word]['weight'] += 1

                    except IndexError:
                        if not dG.has_node(word):
                            dG.add_node(word)
                            dG.node[word]['count'] = 1
                        else:
                            dG.node[word]['count'] += 1
                    except:
                        raise

            dG.remove_edges_from(dG.selfloop_edges())
            # for node1, node2 in dG.edges_iter():
            #     dG.edge[node1][node2]['inv_weight'] = 1.0 / dG.edge[node1][node2]['weight']

                    ## best until now
                    # d['weight'] = d['weight']*((d['w2vec'])**2)
                    # d['weight'] = dice*f

            # nx.write_edgelist(dG,path+str(ind)+"_sliding_"+str(sliding_window)+"_"+str(tf_par)+"_graph.edgelist",data=True)

        else:
            print "Parsing the graph of words for documents..."
            # dG = nx.read_edgelist(path+str(ind)+"_sliding_"+str(sliding_window)+"_"+str(tf_par)+"_graph.edgelist")


        if tf_par=="word2vec":
            for u,v,d in dG.edges(data=True):
                if 'w2vec' in d:

                    # dice = (2*d['weight'])/(dG.node[u]['count']+dG.node[v]['count'])
                    # dG.edge[u][v]['weight'] = dice * (dG.node[u]['count']*dG.node[v]['count'])/((d['w2vec'])**2)

                    # d['weight'] = (dG.node[u]['count']*dG.node[v]['count'])/((1-d['w2vec']))

                    ## angular

                    # dice = (2*d['weight'])/(dG.node[u]['count']+dG.node[v]['count'])
                    # f = (dG.node[u]['count']*dG.node[v]['count'])/(d['w2vec']**2)
                    # print d['w2vec']
                    # d['weight'] = d['weight']/(d['w2vec'])
                    # if u not in counter_word2vec:
                    #     counter_word2vec.append(u)
                    #
                    # if v not in counter_word2vec:
                    #     counter_word2vec.append(v)

                    ## my_w2v_similarity
                    dG.edge[u][v]['w2vec'] = np.arccos(d['w2vec'])/math.pi
                    dG.edge[u][v]['w2vec'] = 1-dG.edge[u][v]['w2vec']
                    dG.edge[u][v]['weight'] = dG.edge[u][v]['w2vec']

                    ## attraction score
                    # d['w2vec'] = np.arccos(d['w2vec'])/math.pi
                    # f_u_v = float(dG.node[u]['count']*dG.node[v]['count'])/(d['w2vec']**2)
                    # dice = float(2*d['weight'])/(dG.node[u]['count']+dG.node[v]['count'])
                    # dG.edge[u][v]['weight'] = f_u_v * dice

                else:
                    dG.edge[u][v]['weight'] = 0.0001
                    # dG.edge[u][v]['weight'] = 1-dG.edge[u][v]['weight']


        #if len(dG)>1:
        if centrality_par == "degree_centrality":
            centrality = nx.degree_centrality(dG)
        elif centrality_par == "weighted_degree_centrality":
            centrality = dG.degree(weight="weight")
            # centrality = weighted_degree_centrality(dG)
        elif centrality_par == "in_degree_centrality":
            centrality = nx.in_degree_centrality(dG)
        elif centrality_par == "out_degree_centrality":
            centrality = nx.out_degree_centrality(dG)
        elif centrality_par == "pagerank_centrality":
            centrality = nx.pagerank(dG)
        elif centrality_par == "weighted_pagerank_centrality":
            centrality = nx.pagerank(dG,weight="weight")
        elif centrality_par =="betweenness_centrality" or centrality_par=="betweenness_centrality_directed":
            centrality = nx.betweenness_centrality(dG,weight="weight")
        elif centrality_par =="triangles":
            centrality = nx.triangles(dG)
        elif centrality_par =="eigenvector_centrality":
            centrality = nx.eigenvector_centrality_numpy(dG)
        elif centrality_par =="core_number":
            centrality = nx.core_number(dG)
        elif centrality_par =="clustering_coefficient":
            centrality = nx.clustering(dG)
        elif centrality_par == "closeness_centrality" or centrality_par=="closeness_centrality_directed":
            centrality = nx.closeness_centrality(dG)
        elif centrality_par == "closeness_centrality_weighted":
            centrality = nx.closeness_centrality(dG,distance='weight')
        elif centrality_par == "communicability_centrality":
            centrality = nx.communicability_centrality(dG)
        elif centrality_par == "closeness_centrality_not_normalized":
            centrality = nx.closeness_centrality(dG,normalized=False)

        #print "Number of self-loops:"+str(dG.number_of_selfloops())
        #centrality = nx.out_degree_centrality(dG)
        #centrality = pg.pagerank(dG,max_iter=1000)
        #centrality = nx.katz_centrality(dG,max_iter=10000)

        # totalNodes += dG.number_of_nodes()
        # totalEdges += dG.number_of_edges()

        #if len(dG)>1:
        for k, g in enumerate(dG.nodes()):
            if g in dGcol_nodes:
                if idf_par=="no":
                    features[i,unique_words.index(g)] = centrality[g]
                    #tfs.append(wordList2.count(g))
                    # centralities.append(centrality[g])
                elif idf_par=="tf-icw":
                    #tf_g = 1+math.log(1+math.log(wordList2.count(g)))
                    tf_g = wordList2.count(g)
                    # features[i,unique_words.index(g)] = (tf_g/(1-b+(b*(float(docLen)/avgLen)))) * icw_col[g]
                    features[i,unique_words.index(g)] = tf_g * icw_col[g]
                elif idf_par=="idf":
                    features[i,unique_words.index(g)] = centrality[g] * idf_col[g]
                    # features[i,unique_words.index(g)] = centrality[g] * idf_col[g]
                elif idf_par=="icw" or idf_par=="icw-lw":
                    features[i,unique_words.index(g)] = centrality[g] * icw_col[g]
                    # features[i,unique_words.index(g)] = centrality[g]/(1-b+(b*(float(docDiam)/avgDiam))) * icw_col[g]
                elif idf_par=="icw+idf":
                    tf_g = wordList2.count(g)
                    #features[i,unique_words.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * icw_col[g] * idf_col[g]
                    features[i,unique_words.index(g)] = centrality[g] * icw_col[g] * idf_col[g]

        dG.clear()

    return features,idx



def createGraphFeatures(num_documents,clean_train_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,train_par,idf_learned,icw_learned,dGcol_nodes,avgLen,path,y_train):
    #features = np.zeros((num_documents,len(unique_words)))
    features = lil_matrix((num_documents,len(unique_words)))
    unique_words_len = len(unique_words)
    term_num_docs = {}

    if train_par:
        print "Training set..."

        if centrality_col_par=="weighted_degree_centrality" or centrality_col_par=="weighted_pagerank_centrality":
            tf_par = "word2vec"
            getOnlyDataWord2VecModel(clean_train_documents)
        else:
            tf_par = "word2ve"

        print "sliding_window:"+str(sliding_window)
        idfs = {}
        dGcol_nodes = {}
        icws = {}
        max_core_feat = []

        ## this is for the label graphs
        dGlabels = []

        totalLen = 0
        totalDiam = 0

        for label in list(set(y_train)):
            dGlabels.append(nx.Graph())

        # ## IDW
        # print "Creating the graph of documents (IDW).."
        # # getOnlyDataWord2VecModel(clean_train_documents)
        #
        #
        # all_doc_nodes = []
        # for i in range( 0,num_documents ):
        #     all_doc_nodes.append(i)
        #
        #
        # edges = combinations(all_doc_nodes, 2)
        # dGdocs = nx.Graph()
        #
        # vectorizer = TfidfVectorizer(min_df=1)
        # tf_idf_matrix = vectorizer.fit_transform(clean_train_documents)

        # for e in edges:
        #     # dGdocs.add_edge(e,weight=metrics.pairwise.cosine_similarity(w2v.wv.wmdistance(clean_train_documents[e[0]],clean_train_documents[e[1]])))
        #     vect = TfidfVectorizer(min_df=1)
        #     tfidf = vect.fit_transform([clean_train_documents[e[0]],clean_train_documents[e[1]]])
        #     dGdocs.add_edge(e[0],e[1],weight=tfidf[0,1])

        t1 = time.time()
        # matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), len(clean_train_documents), 0)
        # t = time.time()-t1
        # print "SELFTIMED:"+str(t)
        #
        # # matches_df = get_matches_df(matches, clean_train_documents)
        # for e in edges:
        #     dGdocs.add_edge(e[0],e[1],weight=matches[e[0],e[1]])
        #
        # del matches

        if not os.path.exists(path+"_collection_"+str(sliding_window)+"_"+str(tf_par)+"_graph.edgelist"):

            print "Creating the graph of words for collection..."

            if centrality_col_par=="pagerank_centrality" or centrality_col_par=="in_degree_centrality" or centrality_col_par=="out_degree_centrality" or centrality_col_par=="closeness_centrality_directed" or centrality_col_par=="betweenness_centrality_directed" or centrality_col_par=="weighted_pagerank_centrality":
                dGcol = nx.DiGraph()
            else:
                dGcol = nx.Graph()

            totalLen = 0
            totalDiam = 0

            for i in range( 0,num_documents ):
                # dG = nx.Graph()
                lg = int(y_train[i])

                found_unique_words = []
                wordList1 = clean_train_documents[i].split(None)
                wordList2 = [string.rstrip(x, ',.!?;') for x in wordList1]

                docLen = len(wordList2)

                # print clean_train_documents[i]

                #if len(wordList2)>1:
                totalLen += docLen
                for k, word in enumerate(wordList2):

                    if word not in found_unique_words:
                        found_unique_words.append(word)
                        if word not in term_num_docs:
                            term_num_docs[word] = 1
                        else:
                            term_num_docs[word] += 1

                    for j in xrange(1,sliding_window):
                        try:
                            next_word = wordList2[k + j]
                            # print word+"\t"+next_word
                            # time.sleep(2)
                            if not dGcol.has_node(word):
                                dGcol.add_node(word)
                                dGcol.node[word]['count'] = 1
                            else:
                                dGcol.node[word]['count'] += 1

                            if not dGcol.has_node(next_word):
                                dGcol.add_node(next_word)
                                dGcol.node[next_word]['count'] = 1
                            else:
                                dGcol.node[next_word]['count'] +=1

                            if not dGcol.has_edge(word, next_word):
                                dGcol.add_edge(word, next_word, weight = 1)
                                # dGcol.edge[word][next_word]['w2vec'] = 0.01
                                if tf_par=="word2vec":

                                    if word in model.wv.vocab and next_word in model.wv.vocab:
                                        dGcol.edge[word][next_word]['w2vec'] = model.wv.similarity(word,next_word)
                                        # dGcol.edge[word][next_word]['w2vec'] = np.linalg.norm(model[word]-model[next_word])
                            else:
                                dGcol.edge[word][next_word]['weight'] += 1


                            ## this is for label graphs

                            if not dGlabels[lg].has_node(word):
                                dGlabels[lg].add_node(word)
                                dGlabels[lg].node[word]['count'] = 1
                            else:
                                dGlabels[lg].node[word]['count'] += 1

                            if not dGlabels[lg].has_node(next_word):
                                dGlabels[lg].add_node(next_word)
                                dGlabels[lg].node[next_word]['count'] = 1
                            else:
                                dGlabels[lg].node[next_word]['count'] +=1

                            if not dGlabels[lg].has_edge(word, next_word):
                                dGlabels[lg].add_edge(word, next_word, weight = 1)
                                # dGcol.edge[word][next_word]['w2vec'] = 0.01
                                if tf_par=="word2vec":

                                    if word in model.wv.vocab and next_word in model.wv.vocab:
                                        dGlabels[lg].edge[word][next_word]['w2vec'] = model.wv.similarity(word,next_word)
                            else:
                                dGlabels[lg].edge[word][next_word]['weight'] += 1

                            # # # again for average,5,6,7,8,9
                            # if not dG.has_edge(word, next_word):
                            #     dG.add_edge(word, next_word, weight = 1)
                            # else:
                            #     dG.edge[word][next_word]['weight'] += 1

                        except IndexError:
                            if not dGcol.has_node(word):
                                dGcol.add_node(word)
                                dGcol.node[word]['count'] = 1
                            else:
                                dGcol.node[word]['count'] += 1

                            if not dGlabels[lg].has_node(word):
                                dGlabels[lg].add_node(word)
                                dGlabels[lg].node[word]['count'] = 1
                            else:
                                dGlabels[lg].node[word]['count'] += 1

                        except:
                            raise

                # nx.draw(dG,pos=nx.spring_layout(dG))
                # plt.show()
                # nx.write_edgelist(dG,path+"_YO_"+str(sliding_window)+"_"+str(tf_par)+"_graph.edgelist")
                # raw_input("enter")
                # totalDiam += nx.diameter(dG)

            # nx.write_edgelist(dGcol,path+"_collection_"+str(sliding_window)+"_"+str(tf_par)+"_graph.edgelist")
            # json.dump(term_num_docs,open(path+"_term_num_docs"+str(sliding_window)+".txt","w"))

        else:
            print "Parsing the graph of words for collection..."
            # term_num_docs = json.load(open(path+"_term_num_docs"+str(sliding_window)+".txt","r"))
            # dGcol = nx.read_edgelist(path+"_collection_"+str(sliding_window)+"_"+str(tf_par)+"_graph.edgelist")

        print "Number of self-loops for collection graph:"+str(dGcol.number_of_selfloops())
        dGcol.remove_edges_from(dGcol.selfloop_edges())
        collection_count_nodes = dGcol.number_of_nodes()
        collection_count_edges = dGcol.number_of_edges()
        print "Number of nodes in collection graph:"+str(collection_count_nodes)
        print "Number of edges in collection graph:"+str(collection_count_edges)

        # plot_degree_histogram(dGcol)
        # raw_input("enter")

        # avgLen = float(totalLen)/num_documents
        avgLen = 0
        # colDiam = nx.diameter(dGcol)
        # avgDiam = float(totalDiam)/num_documents

        print "Average document length:"+str(avgLen)

        if idf_par=="icw" or idf_par=="icw+idf" or idf_par=="tf-icw" or idf_par=="icw-lw":
            icw_col = {}

            if tf_par=="word2vec":
                for u,v,d in dGcol.edges(data=True):
                    if 'w2vec' in d:

                        ## my w2v similarity
                        dGcol.edge[u][v]['w2vec'] = np.arccos(d['w2vec'])/math.pi
                        dGcol.edge[u][v]['w2vec'] = 1-dGcol.edge[u][v]['w2vec']
                        dGcol.edge[u][v]['weight'] = dGcol.edge[u][v]['w2vec']

                        ## attraction score
                        # f_u_v = float(dGcol.node[u]['count']*dGcol.node[v]['count'])/(d['w2vec']**2)
                        # dice = float(2*d['weight'])/(dGcol.node[u]['count']+dGcol.node[v]['count'])
                        # dGcol.edge[u][v]['weight'] = f_u_v * dice

                        #dGcol.edge[u][v]['weight'] = d['weight']*dGcol.edge[u][v]['w2vec']
                        #dGcol.edge[u][v]['weight'] = float(d['weight'])/(dGcol.edge[u][v]['w2vec']**2)
                    else:
                        # dGcol.edge[u][v]['weight'] = np.arccos(0.0001)/math.pi
                        dGcol.edge[u][v]['weight'] = 0.0001


            if centrality_col_par == "degree_centrality":
                centrality_col = nx.degree_centrality(dGcol)
            elif centrality_col_par == "weighted_degree_centrality":
                # centrality_col = nx.degree_centrality(dGcol,weight='weight')
                centrality_col = dGcol.degree(weight='weight')
            elif centrality_col_par=="in_degree_centrality":
                centrality_col = nx.in_degree_centrality(dGcol)
            elif centrality_col_par=="out_degree_centrality":
                centrality_col = nx.out_degree_centrality(dGcol)
            elif centrality_col_par == "pagerank_centrality":
                centrality_col = nx.pagerank(dGcol)
            elif centrality_col_par == "weighted_pagerank_centrality":
                centrality_col = nx.pagerank(dGcol,weight="weight")
            elif centrality_col_par == "eigenvector_centrality":
                centrality_col = nx.eigenvector_centrality(dGcol,max_iter=1000)
            elif centrality_col_par == "betweenness_centrality" or centrality_col_par=="betweenness_centrality_directed":
                centrality_col = nx.betweenness_centrality(dGcol)
            elif centrality_col_par == "triangles":
                centrality_col = nx.triangles(dGcol)
            elif centrality_col_par == "clustering_coefficient":
                centrality_col = nx.clustering(dGcol)
            elif centrality_col_par == "core_number":
                centrality_col = nx.core_number(dGcol)
            elif centrality_col_par == "closeness_centrality" or centrality_col_par=="closeness_centrality_directed":
                centrality_col = nx.closeness_centrality(dGcol)
            elif centrality_col_par == "closeness_centrality_weighted":
                centrality_col = nx.closeness_centrality(dGcol)
            elif centrality_col_par == "communicability_centrality":
                centrality_col = nx.communicability_centrality(dGcol)


            centrality_labels = []

            # partition = community.best_partition(dGcol)
            #
            # all_nodes = []
            # partitions = []
            # count = 0
            # for com in set(partition.values()):
            #     count = count + 1
            #     list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
            #     partitions.append(list_nodes)
            #
            #
            # print "Clusters:"+str(len(partitions))
            #
            #
            # lens = [len(partition) for partition in partitions]
            # print lens
            # t = lens.index(max(lens))
            #
            # print "len of biggest cluster:"+str(len(partitions[t]))

            # raw_input("enter")

            for i,dGlabel in enumerate(dGlabels):
                # centrality_labels.append(nx.pagerank(dGlabel))
                # centrality_labels.append(nx.degree_centrality(dGlabel))
                # print "before:"+str(dGlabel.number_of_nodes())

                ## this is for clustering
                # partition = community.best_partition(dGlabel)
                # all_nodes = []
                # count = 0
                # for com in set(partition.values()):
                #     count = count + 1
                #     list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
                #     all_nodes.append(list_nodes)
                #
                # partitions.append(all_nodes)
                #
                # print str(i)+": "+str(count)+" clusters"

                # G = dGlabel.copy()
                # setA = set(dGlabel.nodes())

                # setB = set(partitions[t])
                # dGlabel.remove_nodes_from(list(setB))

                # setB = set(partitions[1])
                # dGlabel.remove_nodes_from(list(setB))

                # setB = set(partitions[1])
                # dGlabel.remove_nodes_from(list(setB))

                print "after:"+str(dGlabel.number_of_nodes())
                centrality_labels.append(nx.degree_centrality(dGlabel))
                # raw_input("en")

            centr_sum = sum(centrality_col.values())
            # centr_sum = max(centrality_col.values())
            # print centr_sum

            minc = [min(d.values()) for d in centrality_labels]
            minc = min(minc)

            for k,g in enumerate(dGcol.nodes()):
                if centrality_col[g]!=0:
                    if idf_par=="icw" or idf_par=="tf-icw" or idf_par=="icw+idf":
                        #print centrality_col[g]
                        # icw_col[g] = math.log10(float(centr_sum*num_documents)/(centrality_col[g]*term_num_docs[g]))
                        # print g


                        seq = [x.get(g, 0) for x in centrality_labels]
                        centr_max_c = max(seq)
                        ind_max = seq.index(centr_max_c)
                        # print g

                        # topics = []
                        # for i, partition in enumerate(partitions):
                        # for w in partitions[ind_max]:
                        #     if g in w:
                        #         topics = w

                        # print str(topics)
                        # raw_input("enter")
                        # all_words = centrality_labels[ind_max].keys -
                        # sum_all_topics = sum([centrality_labels[ind_max].get(word, 0) for word in topics])
                        # sum_all_topics = sum([centrality_col.get(word, 0) for word in topics])
                        # G = dGlabels[ind_max].copy()
                        # # print "before:"+str(G.number_of_nodes())

                        # G = dGcol.copy()
                        # setA = set(G.nodes())
                        # setB = set(partitions[t])
                        # G.remove_nodes_from(list(setB))
                        # # G.remove_nodes_from(list(setB))
                        #
                        # # print "after:"+str(G.number_of_nodes())
                        # if G.degree(g):
                        #     centr_max_c = G.degree(g)

                        # # print sum_all_topics
                        # # raw_input("enter")

                        centr_sum_c = sum(seq)
                        n_el = sum(s>0 for s in seq)

                        # dGlab = seq.index(centr_max_c)

                        del seq[ind_max]

                        # centr_sum_lab = sum(seq)
                        # print seq
                        # raw_input("enter")

                        term_graphs = []
                        for j,doc in enumerate(dGdocs.nodes()):
                            if g in clean_train_documents[j].split():
                                term_graphs.append(dGdocs.degree(j,weight='weight'))

                        avg_term = np.mean(term_graphs)
                        # print avg_term
                        max_term = sum(term_graphs)
                        #.
                        # icw_col[g] = math.log10((float(centr_sum)/centrality_col[g]) * (float(max_term)/avg_term))
                        # icw_col[g] = math.log10(float(max_term)/avg_term)

                        icw_col[g] = math.log10((float(centr_sum)/centrality_col[g]) * (float(centr_max_c)/max(np.mean(seq),minc)))

                        # a = np.mean(seq)
                        # crc = 2 + ((centr_max_c/max(a,minc)*(float(len(centrality_labels))/n_el)))
                        # icw_col[g] = math.log(crc,2)

                        # icw_col[g] = math.log10((float(centr_sum)/centrality_col[g])) * math.log(crc,2)

                    elif idf_par=="icw-lw":
                        icw_col[g] = math.log10((float(centr_sum)/centrality_col[g]))
                else:
                    icw_col[g] = 0

        # elif idf_par=="idf":
        idf_col = {}
        if idf_par=="idf" or idf_par=="icw+idf":
            for x in term_num_docs:
                idf_col[x] = math.log10(float(num_documents) / term_num_docs[x])

        dGcol_nodes = dGcol.nodes()
        dGcol.clear()


    # for the testing set
    else:

        if idf_par=="idf":
            idf_col = idf_learned
        elif idf_par=="icw" or idf_par=="tf-icw":
            icw_col = icw_learned
        elif idf_par=="icw+idf":
            idf_col = idf_learned
            icw_col = icw_learned

        collection_count_nodes = 0
        collection_count_edges = 0


    totalNodes = 0
    totalEdges = 0

    corrs_per_category = [[] for i in range(4)]
    counter_word2vec = []

    # print "number of word2vec words in docs:"+str(len(counter_word2vec))
    if idf_par=="no":
        idfs = {}
        icws = {}
    if idf_par=="idf":
        idfs = idf_col
        icws = {}
    elif idf_par=="icw" or idf_par=="tf-icw" or idf_par=="icw-lw":
        idfs = {}
        icws = icw_col
    elif idf_par=="icw+idf":
        idfs = idf_col
        icws = icw_col

    processes = cpu_count()
    # processes=1
    all_pairs,idx = chunkIt(clean_train_documents,processes)

    y_final = []

    pool = Pool(processes)
    print "Number of processes:"+str(processes)
    results = [pool.apply_async( splitGraphFeatures, (t,idx[k], idf_par,centrality_par, dGcol_nodes,idfs,icws, sliding_window,unique_words,train_par,path)) for k,t in enumerate(all_pairs)]
    count_rows = 0
    for i,result in enumerate(results):
        r,y = result.get()

        for y_ind,row in enumerate(r):
            features[count_rows,:] = row[:]
            #y_final.append(y_train[y_ind])
            count_rows += 1

    pool.close()

    # if train_par:

        # print "Average number of nodes:"+str(float(totalNodes)/num_documents)
        # print "Average number of edges:"+str(float(totalEdges)/num_documents)

    # all_pairs,idx = chunkIt(clean_train_documents,1)
    # r,y = splitGraphFeatures(all_pairs[0],idx[0], idf_par,centrality_par, dGcol_nodes,idfs,icws, sliding_window,unique_words,train_par,path)
    #
    # count_rows = 0
    # for y_ind,row in enumerate(r):
    #     features[count_rows,:] = row[:]
    #     count_rows += 1

    return features, idfs,icws,collection_count_nodes, collection_count_edges, dGcol_nodes, avgLen
