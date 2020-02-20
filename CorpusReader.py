import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class CorpusReader_TFIDF:
    def __init__(self, corpus, tf = "raw", idf = "base", stopword = "yes", stemmer = "porter", ignorecase = "ignore"):
        self.docs = corpus.fileids()
        self.corpus = corpus
        docs = self.corpus.fileids()
        self.df = pd.DataFrame([doc, self.corpus.words(fileids=[doc])] for doc in docs)

        # Use tf #######
        if tf.lower() == "raw" or tf.lower() == "log" or tf.lower() == "binary": #  Use log normalized term frequency
            self.tf = tf.lower()
            self.ptf = self.tf_calc(self.tf) # gets dictionary of doc, unique words and their frequency
        else:
            print("Invalid tf parameter.")
            return
        # for x in self.ptf:
        #     print(x)
        #
        # # Use idf #######
        # if idf.lower() == "base" or idf.lower() == "smooth": # Use inverse frequency
        #     self.idf = idf.lower()
        # else:
        #     print("Invalid idf parameter")
        #     return
        #
        # # Use stopword #######
        # if stopword.lower() == "yes" or stopword.lower() == "none":
        #     self.stopword = stopword.lower()
        # else:
        #     print("Invalid stopword parameter.")
        #     return
        #
        # self.stemmer = stemmer.lower()
        #
        # if ignorecase.lower() == "ignore" or ignorecase.lower() == "no":
        #     self.ignorecase = ignorecase.lower()
        # else:
        #     print("Invalid ignorecase parameter.")
        #     return

    def tf_calc(self, tfType):
        return_dict = {}
        total_unique = []
        if tfType == 'raw':
            for doc in range(self.df[1].size):
                list_set = set(self.df.iloc[doc][1])    # Changing to a set then back to a list gets only unique words
                unique_list = (list(list_set))
                total_unique = list(set(unique_list) - set(total_unique))   # Create a list of unique words for all docs
                return_dict[self.df.iloc[doc][0]] = self.df.iloc[doc][1]
        #print(sorted(total_unique))
        return_df = pd.DataFrame(0, index = self.docs, columns = sorted(total_unique) )
        for doc in self.docs:
            for word in total_unique:
                return_df.loc[doc, word] = return_dict[doc].count(word)
        # for doc in self.df[0]:
        #     for word in return_dict[doc]:


        return return_dict


    def stem(self):
        if self.stemmer == "porter":
            self.stemmer = PorterStemmer()

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


