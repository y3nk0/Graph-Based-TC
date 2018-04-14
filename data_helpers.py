import numpy as np
import re
import itertools
from collections import Counter
import codecs

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("rt-polarity.pos", "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("rt-polarity.neg", "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_labels_train(path_train,path_test,categories):
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    f = codecs.open(path_train, "r")
    train = [x.strip('\n') for x in f.readlines()]
    f.close()

    clean_train_documents = []
    clean_test_documents = []
    y_train = []
    y_test = []

    num_documents = len(train)

    for i in range(num_documents ):
        # Call our function for each one, and add the result to the list of
        # clean reviews
        line = train[i].split('\t')

        y_train.append(line[0])
        clean_train_documents.append( line[1] )

    f = codecs.open(path_test, "r")
    test = [x.strip('\n') for x in f.readlines()]
    f.close()

    num_test_documents = len(test)

    for i in range( num_test_documents ):
        # Call our function for each one, and add the result to the list of
        # clean reviews
        line = test[i].split('\t')
        y_test.append(line[0])
        clean_test_documents.append( line[1] )

    # Split by words
    # x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # x_text = [s.split(" ") for s in x_text]
    x_text_train = [s.split(" ") for s in clean_train_documents]
    x_text_test = [s.split(" ") for s in clean_test_documents]

    # Generate labels
    labels_train = []
    for label in y_train:
        listofzeros = [0] * len(categories)
        listofzeros[categories.index(label)] = 1
        labels_train.append(listofzeros)

    labels_test = []
    for label in y_test:
        listofzeros = [0] * len(categories)
        listofzeros[categories.index(label)] = 1
        labels_test.append(listofzeros)

    return [x_text_train,x_text_test, labels_train, labels_test]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def pad_sentences_pre_split(sentences_train,sentences_test, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences_train)
    # sequence_length_test = max(len(x) for x in sentences_test)
    # sequence_length = max(sequence_length_train,sequence_length_test)

    padded_sentences_train = []
    for i in range(len(sentences_train)):
        sentence = sentences_train[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences_train.append(new_sentence)

    # sequence_length = max(len(x) for x in sentences_test)
    padded_sentences_test = []
    for i in range(len(sentences_test)):
        sentence = sentences_test[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences_test.append(new_sentence)

    return [padded_sentences_train,padded_sentences_test]


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def build_input_data_pre_split(sentences_train,sentences_test, labels_train,labels_test, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x_train = np.array([[vocabulary[word] for word in sentence] for sentence in sentences_train])
    y_train = np.array(labels_train)

    x_test = np.array([[vocabulary[word] for word in sentence if word in vocabulary] for sentence in sentences_test])
    y_test = np.array(labels_test)

    return [x_train, x_test, y_train, y_test]

def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def load_data_pre_split(path_train,path_test,categories):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences_train,sentences_test, labels_train,labels_test = load_data_and_labels_train(path_train,path_test,categories)
    sentences_padded_train,sentences_padded_test = pad_sentences_pre_split(sentences_train,sentences_test)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded_train)
    x_train, x_test, y_train, y_test = build_input_data_pre_split(sentences_padded_train, sentences_padded_test, labels_train, labels_test, vocabulary)
    return [x_train,x_test, y_train,y_test, vocabulary, vocabulary_inv]
