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

def createGraphFeatures(num_documents,clean_train_documents,unique_words,bigrams,sliding_window,b,idf_par,centrality_par,centrality_col_par,train_par,idf_learned,icw_learned,kcore_par,dGcol_nodes,max_core_col,kcore_par_int,max_core_feat,feature_reduction,avgLen):
    features = np.zeros((num_documents,len(unique_words)))
    unique_words_len = len(unique_words)
    term_num_docs = {}

    print "sliding_window:"+str(sliding_window)
    if train_par:
        print "Training set..."
        idfs = {}
        dGcol_nodes = {}
        icws = {}
        max_core_feat = []

        print "Creating the graph of words for collection..."

        if centrality_col_par=="pagerank_centrality" or centrality_col_par=="in_degree_centrality" or centrality_col_par=="out_degree_centrality" or centrality_col_par=="closeness_centrality_directed" or centrality_col_par=="betweenness_centrality_directed":
            dGcol = nx.DiGraph()
        else:
            dGcol = nx.Graph()
        
        totalLen = 0
        for i in range( 0,num_documents ):
            #dG = nx.Graph()
            found_unique_words = []
            wordList1 = clean_train_documents[i].split(None)
            wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]

            docLen = len(wordList2)
            totalLen += docLen

            # print clean_train_documents[i]
            if len(wordList2)>1:
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
                                dGcol.node[next_word]['count'] = 0

                            if not dGcol.has_edge(word, next_word):
                                dGcol.add_edge(word, next_word, weight = 1)
                            else:
                                dGcol.edge[word][next_word]['weight'] += 1
                        except IndexError:
                            if not dGcol.has_node(word):
                                dGcol.add_node(word)
                                dGcol.node[word]['count'] = 1
                            else:
                                dGcol.node[word]['count'] += 1
                        except:
                            raise

        print "Number of self-loops for collection graph:"+str(dGcol.number_of_selfloops())
        dGcol.remove_edges_from(dGcol.selfloop_edges())
        collection_count_nodes = dGcol.number_of_nodes()
        collection_count_edges = dGcol.number_of_edges()
        print "Number of nodes in collection graph:"+str(collection_count_nodes)
        print "Number of edges in collection graph:"+str(collection_count_edges)
        avgLen = float(totalLen)/num_documents
        print "Average document length:"+str(avgLen)
        
   
        if idf_par=="icw" or idf_par=="icw+idf" or idf_par=="tf-icw":
            icw_col = {}

            if(kcore_par=="A1" or kcore_par=="A2"):
                collection_core = nx.core_number(dGcol)
                max_core = max(collection_core.values())
                print "Max core of collection:"+str(max_core)
                # core_Size_Distribution(collection_core)
                for k,g in enumerate(dGcol.nodes()):
                    if kcore_par=="A1":
                        # A1 method: remove features and then rank
                        for x in range(0,kcore_par_int):
                            if collection_core[g]==max_core-x:
                                dGcol.remove_node(g)
                    else:
                        # A2 method: rank first and then remove features
                        for x in range(0,kcore_par_int):
                            if collection_core[g]==max_core-x:
                                max_core_col.append(g)


            if centrality_col_par == "degree_centrality":
                centrality_col = nx.degree_centrality(dGcol)
            elif centrality_col_par=="in_degree_centrality":
                centrality_col = nx.in_degree_centrality(dGcol)
            elif centrality_col_par=="out_degree_centrality":
                centrality_col = nx.out_degree_centrality(dGcol)
            elif centrality_col_par == "pagerank_centrality":
                # centrality_col = pg.pagerank(dGcol,max_iter=1000)
                centrality_col = nx.pagerank(dGcol)
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

            centr_sum = sum(centrality_col.values())
            for k,g in enumerate(dGcol.nodes()):
                if centrality_col[g]!=0:
                    if idf_par=="icw" or idf_par=="tf-icw" or idf_par=="icw+idf":
                        icw_col[g] = math.log10(float(centr_sum)/centrality_col[g])
                else:
                    icw_col[g] = 0

        # elif idf_par=="idf":
        idf_col = {}
        for x in term_num_docs:
            if idf_par=="idf":
                idf_col[x] = math.log10((float(num_documents)+1.0) / term_num_docs[x])
            elif idf_par=="icw+idf":
                idf_col[x] = math.log10((float(num_documents)+1.0) / term_num_docs[x])

        dGcol_nodes = dGcol.nodes()

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

    # nx.write_edgelist(dGcol,"test.edgelist",data=True,delimiter="\t")

    print "Creating the graph of words for each document..."
    totalNodes = 0
    totalEdges = 0

    corrs_per_category = [[] for i in range(4)]

    for i in range( 0,num_documents ):

        if centrality_par=="pagerank_centrality" or centrality_par=="in_degree_centrality" or centrality_par=="out_degree_centrality" or centrality_par=="closeness_centrality_directed" or centrality_par=="betweenness_centrality_directed":
            dG = nx.DiGraph()
        else:
            dG = nx.Graph()

        wordList1 = clean_train_documents[i].split(None)
        wordList2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
        docLen = len(wordList2)

        if len(wordList2)>1:
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

                        if not dG.has_edge(word, next_word):
                            dG.add_edge(word, next_word, weight = 1)
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
            for node1, node2 in dG.edges_iter():
                dG.edge[node1][node2]['inv_weight'] = 1.0 / dG.edge[node1][node2]['weight']

            if train_par:
                if(kcore_par=="B1" or kcore_par=="B2"):
                    max_core_doc = []
                    document_core = nx.core_number(dG)
                    max_core = max(document_core.values())
                    # print "Max core of document:"+str(max_core)
                    # core_Size_Distribution(document_core)
                    for k,g in enumerate(dG.nodes()):
                        if kcore_par=="B1":
                            # B1 method: remove features and then rank
                            for x in range(0,kcore_par_int):
                                if document_core[g]==max_core-x:
                                    dG.remove_node(g)
                        else:
                            # B2 method: rank first and then remove features
                            for x in range(0,kcore_par_int):
                                if document_core[g]==max_core-x:
                                    max_core_doc.append(g)
                                    if g not in max_core_feat:
                                        max_core_feat.append(g)
            
            # centrality = nx.degree_centrality(dG)
            #centrality = nx.core_number(dG)
            if centrality_par == "degree_centrality":
                centrality = nx.degree_centrality(dG)
            elif centrality_par == "in_degree_centrality":
                centrality = nx.in_degree_centrality(dG)
            elif centrality_par == "out_degree_centrality":
                centrality = nx.out_degree_centrality(dG)
            elif centrality_par == "pagerank_centrality":
                # centrality = pg.pagerank(dG,max_iter=1000)
                centrality = nx.pagerank(dG)
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
            elif centrality_par == "degree_centrality_weighted":
                centrality = weighted_degree_centrality(dG)
            #print "Number of self-loops:"+str(dG.number_of_selfloops())
            #centrality = nx.out_degree_centrality(dG)
            #centrality = pg.pagerank(dG,max_iter=1000)
            #centrality = nx.katz_centrality(dG,max_iter=10000)

            totalNodes += dG.number_of_nodes()
            totalEdges += dG.number_of_edges()

            tfs = []
            centralities = []
            centr_sum_doc = sum(centrality.values())

            for k, g in enumerate(dG.nodes()):
                if g in dGcol_nodes:
                    if kcore_par=="B2":
                        if g in max_core_feat:
                            # Degree centrality (local feature)
                            if g in unique_words:
                                #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                                if idf_par=="no":
                                    features[i,unique_words.index(g)] = centrality[g]/(1-b+(b*(float(docLen)/avgLen)))
                                elif idf_par=="idf":
                                    features[i,unique_words.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * idf_col[g]
                                    # features[i,unique_words.index(g)] = centrality[g] * idf_col[g]
                                elif idf_par=="icw":
                                    features[i,unique_words.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * icw_col[g]
                                    # features[i,unique_words.index(g)] = centrality[g] * icw_col[g]
                                elif idf_par=="icw+idf":
                                    features[i,unique_words.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * icw_col[g] * idf_col[g]
                                    # features[i,unique_words.index(g)] = centrality[g] * math.log10(icw_col[g] * idf_col[g])

                            elif g in bigrams:
                                #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                                if idf_par=="no":
                                    features[i,unique_words_len+bigrams.index(g)] = centrality[g]/(1-b+(b*(float(docLen)/avgLen)))
                                elif idf_par=="idf":
                                    features[i,unique_words_len+bigrams.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * idf_col[g]
                                    # features[i,unique_words.index(g)] = centrality[g] * idf_col[g]
                                elif idf_par=="icw":
                                    features[i,unique_words_len+bigrams.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * icw_col[g]
                                    # features[i,unique_words.index(g)] = centrality[g] * icw_col[g]
                                elif idf_par=="icw+idf":
                                    features[i,unique_words_len+bigrams.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * icw_col[g] * idf_col[g]
                                    # features[i,unique_words.index(g)] = centrality[g] * math.log10(icw_col[g] * idf_col[g])
                    else:
                        if g in unique_words:
                            #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                            if idf_par=="no":
                                features[i,unique_words.index(g)] = centrality[g]/(1-b+(b*(float(docLen)/avgLen)))
                                tfs.append(wordList2.count(g))
                                centralities.append(centrality[g])
                            elif idf_par=="tf-icw":
                                tf_g = 1+math.log(1+math.log(wordList2.count(g)))
                                features[i,unique_words.index(g)] = (tf_g/(1-b+(b*(float(docLen)/avgLen)))) * icw_col[g]
                            elif idf_par=="idf":
                                features[i,unique_words.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * idf_col[g]
                                # features[i,unique_words.index(g)] = centrality[g] * idf_col[g]
                            elif idf_par=="icw":
                                features[i,unique_words.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * icw_col[g]
                                # features[i,unique_words.index(g)] = centrality[g] * icw_col[g]
                            elif idf_par=="icw+idf":
                                features[i,unique_words.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * icw_col[g] * idf_col[g]
                                # features[i,unique_words.index(g)] = centrality[g] * math.log10(icw_col[g] * idf_col[g])

                        elif g in bigrams:
                            #features[i,unique_words.index(g)] = dG.degree(nbunch=g,weight='weight') * idf_col[g]
                            if idf_par=="no":
                                features[i,unique_words_len+bigrams.index(g)] = centrality[g]/(1-b+(b*(float(docLen)/avgLen)))
                            elif idf_par=="idf":
                                features[i,unique_words_len+bigrams.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * idf_col[g]
                                # features[i,unique_words.index(g)] = centrality[g] * idf_col[g]
                            elif idf_par=="icw":
                                features[i,unique_words_len+bigrams.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * icw_col[g]
                                # features[i,unique_words.index(g)] = centrality[g] * icw_col[g]
                            elif idf_par=="icw+idf":
                                features[i,unique_words_len+bigrams.index(g)] = (centrality[g]/(1-b+(b*(float(docLen)/avgLen)))) * icw_col[g] * idf_col[g]
                                # features[i,unique_words.index(g)] = centrality[g] * math.log10(icw_col[g] * idf_col[g])
    #     if train_par:
    #         # pears = pearsonr(tfs,centralities)

    #         ind_tfs = sorted(range(len(tfs)), key=lambda k: tfs[k])[-20:]
    #         ind_centr = sorted(range(len(centralities)), key=lambda k: centralities[k])[-20:]
    #         tau, p_value = kendalltau([unique_words[k] for k in ind_tfs],[unique_words[k] for k in ind_centr])
            
    #         corrs_per_category[int(y[i])-1].append(tau)
    
    # if train_par:

    #     text_file = open("kendal_tfs_tws_output_tw_idf_"+idf_par+"_centr_"+centrality_par+"_sliding_"+str(sliding_window)+"_kcore_"+kcore_par+".txt", "w")
        
    #     text_file.write(str(corrs_per_category))
    #     text_file.close()

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)

    #     ax.boxplot(corrs_per_category[:])

    #     plt.show()


    if idf_par=="no":
        idfs = {}
        icws = {}
    if idf_par=="idf":
        idfs = idf_col
        icws = {}
    elif idf_par=="icw" or idf_par=="tf-icw":
        idfs = {}
        icws = icw_col
    elif idf_par=="icw+idf":
        idfs = idf_col
        icws = icw_col

    if train_par:
        if kcore_par=="B2":
            feature_reduction = float(len(max_core_feat))/len(dGcol_nodes)
            print "Percentage of features kept:"+str(feature_reduction)
        print "Average number of nodes:"+str(float(totalNodes)/num_documents)
        print "Average number of edges:"+str(float(totalEdges)/num_documents)
    
    return features, idfs,icws,collection_count_nodes, collection_count_edges, dGcol_nodes,max_core_col,feature_reduction, max_core_feat,avgLen

