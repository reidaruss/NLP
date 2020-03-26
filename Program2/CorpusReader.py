# Reid Russell
# NLP - CS5391
# Program 1
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
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
            self.ptf = self.tfidf_calc(self.tf) # gets tf_idf of docs in dictionary format
        else:
            print("Invalid tf parameter.")
            return

        return

    # Extends corpus class method fileids()
    def fileids(self):
        return self.corpus.fileids()

    # Extends corpus class method raw()
    def raw(self, fileids = []):
        if len(fileids) == 0:
            return self.corpus.raw()
        else:
            return self.corpus.raw(fileids)


    # Extends corpus class method words()
    def words(self, fileids = []):
        if len(fileids) == 0:
            return self.corpus.words()
        else: return self.corpus.words(fileids)

    def open(self, fileid):
        return self.corpus.open(fileid)

    def abspath(self, fileid):
        return self.corpus.abspath(fileid)


    def tfidf_calc(self, tfType):
        scale_factor = 1 # Look at a smaller dataset
        N = round(self.df[1].size * scale_factor)
        tf_idf = []
        total_unique = np.array([]) # A np array that tracks all unique words
        freqs = []  # Tracks the dictionaries associated with each document
        for doc in range(round(self.df[1].size*scale_factor)):
            a = np.array(self.df.iloc[doc][1])  # Get each set of words associated with a document
            if self.ignorecase == "ignore":     # Ignore case or do not ignore case
                a = np.char.lower(a)
            a = [word for word in a if word.isalpha()]  # Remove punctuation and numbers
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
            #if doc % 50 == 0 and doc != 0:
                #print((doc/round(self.df[1].size*scale_factor))*100)    # when uncommented gives % status of read in
        total_unique = np.unique(total_unique) # Tracks the list of all unique words in the corpus in alphabetical order
        self.total_unique = total_unique
        ni = np.zeros(len(total_unique))
        for i in range(len(total_unique)):  # Calculate ni to use in idf
            for subdict in freqs:
                if total_unique[i] in subdict:
                    ni[i] = ni[i] + 1

        for i in range(len(freqs)):
            tfidf = copy.deepcopy(freqs[i])
            for j in freqs[i].items():
                if self.idf == 'smooth':    # This is the formula implementation of the final tf_idf step
                    tfidf[j[0]] =  j[1] * (np.log(1+(N/ni[np.where(total_unique == j[0])]))/np.log(2))
                else:
                    tfidf[j[0]] = j[1] * (np.log(N / ni[np.where(total_unique == j[0])]) / np.log(2))
            tf_idf.append(tfidf)

        return tf_idf


    #  return a list of ALL tf-idf vector (each vector should be a list) for the corpus,
    # ordered by the order where filelds are returned (the dimensions of the vector should be
    # sorted in alphabetical order)
    def tf_idf(self,fileid= []):
        if len(fileid) == 0:
            return self.ptf
        elif len(fileid) == 1:
            for i in range(len(self.docs)): # If there is 1 file return the document vector for that file
                if self.docs[i] == fileid[0]:
                    return self.ptf[i]
        else:
            a = []
            for x in fileid:
                for i in range(len(self.docs)): # if there are multiple create an array of the doc vectors and return
                    if self.docs[i] == x:
                        a.append(self.ptf[i])
            return a


    # return the list of the words in the order of the dimension of each
    # corresponding to each vector of the tf-idf vector
    def tf_idf_dim(self):
        return self.total_unique

    #  the input should be a list of words (treated as a document). The
    # function should return a vector corresponding to the tf_idf vector for the new
    # document (with the same stopword, stemming, ignorecase treatment applied, if
    # necessary). You should use the idf for the original corpus to calculate the result (i.e. do
    # not treat the document as a part of the original corpus)
    def tf_idf_new(self, words = []):
        N = 1
        tf_idf = []
        total_unique = np.array([]) # A np array that tracks all unique words
        freqs = []  # Tracks the dictionaries associated with each document
        a = np.array(words)  # Get each set of words associated with a document
        if self.ignorecase == "ignore":     # Ignore case or do not ignore case
            a = np.char.lower(a)
        a = [word for word in a if word.isalpha()]  # Remove punctuation and numbers
        for i in range(len(a)):             # Use stemmer before getting unique values
            a[i] = self.stemmer.stem(a[i])

        if self.stopwords != "NULL":     # Remove stopwords
            a = [w for w in a if not w in self.stopwords]


        unique, counts = np.unique(a, return_counts = True)     # Get the unique values and their counts
        if self.tf == 'log':
            counts = 1 + (np.log(counts) / np.log(2))
        if self.tf == 'binary':
            counts = np.ones(len(counts))
        tempdict = dict(zip(unique,counts))   # Put unique vals and counts into dictionary

        total_unique = np.concatenate([total_unique,unique])   # Create a list of unique words for all docs
        freqs.append(tempdict)
        ni = np.zeros(len(total_unique))
        for i in range(len(total_unique)):  # Calculate ni to use in idf
            for subdict in freqs:
                if total_unique[i] in subdict:
                    ni[i] = ni[i] + 1

        for i in range(len(freqs)):
            tfidf = copy.deepcopy(freqs[i])
            for j in freqs[i].items():
                if self.idf == 'smooth':    # This is the formula implementation of the final tf_idf step
                    tfidf[j[0]] =  j[1] * (np.log(1+(N/ni[np.where(total_unique == j[0])]))/np.log(2))
                else:
                    tfidf[j[0]] = j[1] * (np.log(N / ni[np.where(total_unique == j[0])]) / np.log(2))
            tf_idf.append(tfidf)
        return tf_idf

    # return the cosine similarity between two documents
    # in the corpus
    def cosine_sim(self, fileid = []):
        file1 = fileid[0]
        file2 = fileid[1]
        file1ind = 0
        file2ind = 0
        for i in range(len(self.docs)): # Get the index for the file
            if self.docs[i] == file1:
                file1ind = i
        for i in range(len(self.docs)):
            if self.docs[i] == file2:
                file2ind = i
        a1 = []
        a2 = []
        for file in self.ptf[file1ind].items():
            a1.append(file[0])
        for file in self.ptf[file2ind].items(): # Get the vector of words for each file
            a2.append(file[0])
        a1 = set(a1)
        rvector = a1.union(set(a2)) # Take the union to find intersections
        x = []
        z = []
        for w in rvector:
            if w in a1: x.append(1) # Count the similarities
            else: x.append(0)
            if w in a2: z.append(1)
            else: z.append(0)
        c = 0

        for i in range(len(rvector)):
            c+= x[i]*z[i]
        cosine_sim = c / float((sum(x)*sum(z))) # Calculate cosine similarity

        return cosine_sim

    #  the [words] is a list of words as is in the parameter of
    # tf_idf_new() method. The fileid is the document in the corpus. The function return the
    # cosine similarity between fileid and the new document specify by the [words] list. (Once
    # again, use the idf of the original corpus).
    def cosine_sim_new(self,words,fileid):
        file1 = words
        file2 = fileid
        file2ind = 0
        for i in range(len(self.docs)):
            if self.docs[i] == file2:
                file2ind = i
        a1 = file1
        a2 = []
        for file in self.ptf[file2ind].items(): # Get the vector of words for each file
            a2.append(file[0])
        a1 = set(a1)
        rvector = a1.union(set(a2)) # Take the union to find intersections
        x = []
        z = []
        for w in rvector:
            if w in a1: x.append(1) # Count the similarities
            else: x.append(0)
            if w in a2: z.append(1)
            else: z.append(0)
        c = 0

        for i in range(len(rvector)):
            c+= x[i]*z[i]
        cosine_sim = c / float((sum(x)*sum(z))) # Calculate cosine similarity

        return cosine_sim

    # Creates a vector for a particular file with the tf_idf values for first 15 unique words in the corpus
    def get_tfidf_vec(self, file_index):
        a = np.zeros(15)
        for file in self.ptf[file_index].items():
            for i in range(15):
                if file[0] == self.total_unique[i]:
                    a[i] = file[1]

        return a



