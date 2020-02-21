import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import copy

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

        if idf.lower() == "base" or idf.lower() == "smooth":
            self.idf = idf.lower()
        else:
            print("Invalid idf parameter")
            return


        # Calculate tf-idf #######
        if tf.lower() == "raw" or tf.lower() == "log" or tf.lower() == "binary": #  Use log normalized term frequency
            self.tf = tf.lower()
            self.ptf = self.tf_calc(self.tf) # gets dataframe of docs, unique words and their frequency
        else:
            print("Invalid tf parameter.")
            return

        # Calculate idf #######

        return

    def tf_calc(self, tfType):
        scale_factor = 0.5 # Look at a smaller dataset
        N = round(self.df[1].size * scale_factor)
        tf_idf = []
        total_unique = np.array([]) # A np array that tracks all unique words
        freqs = []  # Tracks the dictionaries associated with each document
        for doc in range(round(self.df[1].size*scale_factor)):
            a = np.array(self.df.iloc[doc][1])  # Get each set of words associated with a document
            if self.ignorecase == "ignore":     # Ignore case or do not ignore case
                a = np.char.lower(a)
            a = [word for word in a if word.isalpha()]
            for i in range(len(a)):             # Use stemmer before getting unique values
                a[i] = self.stemmer.stem(a[i])

            if self.stopwords != "NULL":     # Remove stopwords
                a = [w for w in a if not w in self.stopwords]


            unique, counts = np.unique(a, return_counts = True)     # Get the unique values and their counts
            if tfType == 'log':
                counts = 1 + (np.log(counts) / np.log(2))
            if tfType == 'binary':
                counts = np.ones(len(counts))
            tempdict = dict(zip(unique,counts))   # Put unique vals and counts into dictionary

            total_unique = np.concatenate([total_unique,unique])   # Create a list of unique words for all docs
            freqs.append(tempdict)
            if doc % 50 == 0 and doc != 0:
                print((doc/round(self.df[1].size*scale_factor))*100)
        total_unique = np.unique(total_unique)
        ni = np.zeros(len(total_unique))
        for i in range(len(total_unique)):
            for subdict in freqs:
                if total_unique[i] in subdict:
                    ni[i] = ni[i] + 1



        for i in range(len(freqs)):
            tfidf = copy.deepcopy(freqs[i])
            for j in freqs[i].items():
                if self.idf == 'smooth':
                    tfidf[j[0]] =  j[1] * (np.log(1+(N/ni[np.where(total_unique == j[0])]))/np.log(2))
                else:
                    tfidf[j[0]] = j[1] * (np.log(N / ni[np.where(total_unique == j[0])]) / np.log(2))
            tf_idf.append(tfidf)

        return tf_idf

    def getTF(self):
        return self.ptf

    def getIDF(self):
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


