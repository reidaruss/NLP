import numpy as np
import nltk

class CorpusReader_TFIDF:
    def __init__(self, corpus, tf, idf, stopword, stemmer, ignorecase):
        return

    #  return a list of ALL tf-idf vector (each vector should be a list) for the corpus,
    # ordered by the order where filelds are returned (the dimensions of the vector should be
    # sorted in alphabetical order)
    def tf_idf(self):
        return

    #  return the tf-idf vector corresponding to that file
    def tf_idf(self, fileid):
        return

    # return a list of vectors, corresponding to the tf-idf to the list of
    # fileid input
    def tf_idf(self, filelist):
        return

    # return the list of the words in the order of the dimension of each
    # corresponding to each vector of the tf-idf vector
    def tf_idf_dim(self):
        return

    #  the input should be a list of words (treated as a document). The
    # function should return a vector corresponding to the tf_idf vector for the new
    # document (with the same stopword, stemming, ignorecase treatment applied, if
    # necessary). You should use the idf for the original corpus to calculate the result (i.e. do
    # not treat the document as a part of the original corpus)
    def tf_idf_new(self, words):
        return
    # return the cosine similarity between two documents
    # in the corpus
    def cosine_sim(self, fileid):
        return

    #  the [words] is a list of words as is in the parameter of
    # tf_idf_new() method. The fileid is the document in the corpus. The function return the
    # cosine similarity between fileid and the new document specify by the [words] list. (Once
    # again, use the idf of the original corpus).
    def cosine_sim_new(self,words,fileid):
        return


