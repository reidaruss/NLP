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
        freqs = []
        return_df = pd.DataFrame(index=self.docs)
        if tfType == 'raw':
            for doc in range(self.df[1].size):

                a = np.array(self.df.iloc[doc][1])
                unique, counts = np.unique(a, return_counts = True)
                tempdict = dict(zip(unique,counts))
                #table = pd.DataFrame(tempdict)
                return_df[doc].append(tempdict)
                #return_df.loc[self.docs[doc]] = tempdict
                #total_unique = list(set(unique) - set(total_unique))   # Create a list of unique words for all docs
                freqs.append(tempdict)
                #print(table)

        total_unique = sorted(total_unique)
       # return_df = pd.DataFrame(0,columns=total_unique)
        print(return_df)

        # for dictindex in range(len(freqs)):
        #     for word in freqs[dictindex]:
        #         return_df.loc[self.docs[dictindex], word] = freqs[dictindex][word]
        #     print(dictindex)
        #print(return_dict)
        #for doc in self.docs:
        # i = 0
        # for word in total_unique:
        #     for doc in self.docs:
        #         return_df.loc[doc, word] = return_dict[doc].count(word)
        #         i = i +1
        #         if i % 500 == 0:
        #             print(return_df)
                #print(return_dict[doc].count(word))


        # for doc in self.df[0]:
        #     for word in return_dict[doc]:
        print(return_df)

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


