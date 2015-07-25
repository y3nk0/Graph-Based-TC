# Graph-Based-TC
## Graph-based framework for text classification

### Main instructions
There are 3 python files for every dataset. We use the fetch_20newsgroups built-in python to get the 20newsgroup dataset. For the IMDB dataset, you can download it [here](http://ai.stanford.edu/~amaas/data/sentiment/). 

Files ending with main.py contain tf and tf-idf, files ending with tf-icf.py contain tf-icf(term frequency-inverse collection frequency) and files ending with gow.py contain the tw, tw-idf and tw-icw methods.

### Parameters
In each file there are some parameters to set.

#### parameters for main.py files
bag_of_words: use our tf-idf or the tf-idf vectorizer
ngrams_par: the number of ngrams
idf_bool: use idf or not

#### parameters for gow.py files
idf_pars: {"no","idf","icw","tf-icw"}, "no" for tw method, "idf" for tw-idf, "icw" for tw-icw, "tf-icw" for tf-icw
sliding_window: the parameter for creating edges betweeen words. 2 is for connecting only to the next word
kcore_pars: {"A0","B2"}, "A0" stands for no kcore feature selection and "B2" for max core feature selection on document level