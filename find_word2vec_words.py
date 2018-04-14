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

from gensim.models import LsiModel, word2vec, KeyedVectors
from gensim import corpora, models, similarities

## NLTK UTILITIES
from nltk.corpus import wordnet as wn

## LOGGING
import logging

import scipy
from multiprocessing import cpu_count, Process, Queue, Pool
from contextlib import closing
import codecs

# ## use pretrained word vectors
model = KeyedVectors.load_word2vec_format('/Users/konstantinosskianis/Documents/phd/w2v_distances/wmd/GoogleNews-vectors-negative300.bin',binary=True)
#model.intersect_word2vec_format('../word2vec/glove_model.txt')
#model.train(sentences)
print len(model.wv.vocab)

# Demo: Loads the newly created glove_model.txt into gensim API.
# model= word2vec.Word2Vec.load_word2vec_format("../glove_model.txt",binary=False) #GloVe Model
# model = word2vec.Word2Vec.load("word2vec/"+'text8.model')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

paths = ['imdb']
paths1 = ['IMDB']
# paths = ['amazon','reuters','webkb','snippet','20newsgroup','imdb']
# paths1 = ['amazon','reuters','webkb','snippet','20NG','imdb']

for i,path in enumerate(paths):

    f = codecs.open(path+'/data/my_'+paths1[i]+'_train_VOCAB.txt', "r", encoding="utf-8")
    unique_words = [x.strip('\n') for x in f.readlines()]
    f.close()

    print path+" "+str(len(set(unique_words).intersection(model.wv.vocab)))
