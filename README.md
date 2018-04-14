# Graph-Based-TC
## Graph-based framework for text classification

### Datasets
This is the code for the paper ["Fusing Document, Collection and Label Graph-based Representations with Word Embeddings for Text Classification"](https://www.lix.polytechnique.fr/~kskianis/papers/tw-icw-w2v_textgraphs2018.pdf), presented at the TextGraphs workshop, NAACL 2018, New Orleans, USA.

Our implementation includes 6 datasets: 20newsgroups, IMDB, WebKb, Reuters, Subjectivity and Amazon. We use the fetch_20newsgroups built-in python to get the 20newsgroup dataset. For the IMDB dataset, you can download it [here](http://ai.stanford.edu/~amaas/data/sentiment/). We add all remaining datasets [here](https://www.dropbox.com/sh/1qhbsvfhqgsu3wy/AADshIA30o6M8daul4GDVlBpa?dl=0), due to GitHub size limits.

Files ending with main.py contain tf and tf-idf, and files ending with gow.py contain the tw, tw-idf, tw-icw and tw-icw-lw methods.

### Parameters
Inside each file there are several parameters to set in order to get the result of the desired method.

#### parameters for main.py files
- bag_of_words: use our tf-idf or the tf-idf vectorizer(scikit-learn)
- ngrams_par: the number of ngrams
- idf_bool: use idf or not

#### parameters for gow.py files
- idf_pars: {"no","idf","icw”,”icw-lw”}, "no" for tw method, "idf" for tw-idf, "icw" for tw-icw, “icw-lw” for tw-icw-lw
- sliding_window: the parameter for creating edges between words. 2 is for connecting only to the next word
- centrality_par: the centrality metric which we use for term weighting (e.g. weighted_degree_centrality for weighted w2v version)
- centrality_col_par: the centrality metric which we use for the collection graph

### Example
For the WebKb dataset you go in the webkb/:
- for tf run: webkb_main.py with parameter idf_bool = False
- for tf-idf run: python webkb_main.py with parameter idf_bool = True
- for tw with degree centrality run: python webkb_gow.py with parameter idf_par="no"
- for tw-idf with degree centrality run: python webkb_gow.py with parameter idf_par="idf"
- for tw-icw with degree centrality on both tw and icw run: python webkb_gow.py with parameter idf_par="icw"
- for tf-icw with degree centrality on icw run: python webkb_gow.py with parameter idf_par="tf-icw"
- for tw-icw-lw with degree centrality on both tw,icw and lw run: python webkb_gow.py with parameter idf_par="icw-lw"
