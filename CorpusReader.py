import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class CorpusReader_TFIDF:
    def __init__(self, corpus, tf = "raw", idf = "base", stopword = "english", stemmer = "porter", ignorecase = "ignore"):
        self.docs = corpus.fileids()    # Get list of docs
        self.corpus = corpus

        # This line creates a dataframe of document names and the words in that document for all documents
        self.df = pd.DataFrame([doc, self.corpus.words(fileids=[doc])] for doc in self.docs)

        # Set ingnorecase ####
        if ignorecase.lower() == "ignore" or ignorecase.lower() == "no":
            self.ignorecase = ignorecase.lower()

        # Set Stemmer ######
        if stemmer.lower() == "porter":
            self.stemmer = PorterStemmer()

        # Set stopword #######
        if stopword.lower() == "english":
            self.stopwords = set(stopwords.words('english'))
        elif stopword.lower() == "none":
            self.stopwords = 'NULL'
        else:
            print("Invalid stopword parameter.")
            return


        # Use tf #######
        if tf.lower() == "raw" or tf.lower() == "log" or tf.lower() == "binary": #  Use log normalized term frequency
            self.tf = tf.lower()
            self.ptf = self.tf_calc(self.tf) # gets dataframe of docs, unique words and their frequency
        else:
            print("Invalid tf parameter.")
            return



        # # Use idf #######
        # if idf.lower() == "base" or idf.lower() == "smooth": # Use inverse frequency
        #     self.idf = idf.lower()
        # else:
        #     print("Invalid idf parameter")
        #     return
        #

        return

    def tf_calc(self, tfType):
        return_dict = {}
        total_unique = np.array([])
        freqs = []

        if tfType == 'raw':
            for doc in range(self.df[1].size):
                a = np.array(self.df.iloc[doc][1])  # Get each set of words associated with a document
                if self.ignorecase == "ignore":     # Ignore case or do not ignore case
                    a = np.char.lower(a)

                for i in range(len(a)):             # Use stemmer before getting unique values
                    a[i] = self.stemmer.stem(a[i])


                if self.stopwords != "NULL":     # Remove stopwords
                    a = [w for w in a if not w in self.stopwords]

                unique, counts = np.unique(a, return_counts = True)     # Get the unique values and their counts
                tempdict = dict(zip(unique,counts))   # Put unique vals and counts into dictionary

                total_unique = np.unique(np.concatenate([total_unique,unique]))   # Create a list of unique words for all docs
                freqs.append(tempdict)

        return_df = pd.DataFrame(0,index=self.docs, columns=sorted(total_unique))   # Create dataframe with all unique
                                                                                    # words as columns and each doc
                                                                                    # as row index.


        for doc in range(len(self.docs)):
            return_dict[self.docs[doc]] = freqs[doc]  # Asociate each document with its dictionary of unique words and freqs

        for doc, tdict in return_dict.items():  # Fill dataframe with each frequency by doc and word in dictionary
            tdict = dict(tdict)
            for word, freq in tdict.items():
                return_df.loc[doc, word] = freq
        print(return_df.describe())

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


