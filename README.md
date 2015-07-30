# Graph-Based-TC
## Graph-based framework for text classification

### Main instructions
There are 3 python files for every dataset. We use the fetch_20newsgroups built-in python to get the 20newsgroup dataset. For the IMDB dataset, you can download it [here](http://ai.stanford.edu/~amaas/data/sentiment/). 

Files ending with main.py contain tf and tf-idf, files ending with tf-icf.py contain tf-icf(term frequency-inverse collection frequency) and files ending with gow.py contain the tw, tw-idf and tw-icw methods.

### Parameters
Inside each file there are several parameters to set in order to get the result of the desired method.

#### parameters for main.py files
- bag_of_words: use our tf-idf or the tf-idf vectorizer(scikit-learn)
- ngrams_par: the number of ngrams
- idf_bool: use idf or not

#### parameters for gow.py files
- idf_pars: {"no","idf","icw","tf-icw"}, "no" for tw method, "idf" for tw-idf, "icw" for tw-icw, "tf-icw" for tf-icw
- sliding_window: the parameter for creating edges betweeen words. 2 is for connecting only to the next word
- kcore_pars: {"A0","B2"}, "A0" stands for no kcore feature selection and "B2" for max core feature selection on document level
- centrality_par: the centrality metric which we use for term weighting

### Example
For the WebKb dataset you go in the code/webkb/:
- for the tf you run: python webkb_main.py with parameters: idf_bool = False
- for the tf-idf you run: python webkb_main.py with parameters: idf_bool = True
- for the tf-icf you run: python webkb_tf_icf.py
- for the tw with degree centrality you run: python webkb_gow.py with parameters: idf_par="no"
- for the tw-idf with degree centrality you run: python webkb_gow.py with parameters: idf_par="idf"
- for the tw-icw with degree centrality on both tw and icw you run webkb_gow.py with parameters: idf_par="icw"