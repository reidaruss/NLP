# Reid Russell
# NLP - CS5391
# Program 1
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import nltk

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import LdaModel
import numpy as np
import sys
import os
import re


# print('Number of args: ', len(sys.argv), 'arguments.')
# print('List of args: ', str(sys.argv))


# print(directory, output, params)

# TODO:Need to add only checking first word / first character etc
def load_params(paramFile):
    rd = open(paramFile, "r")
    #print(paramFile)
    out = []
    while True:
        line = rd.readline()
        #print(line)
        if not line:
            break
        out.append(line.strip())

    rd.close()

    return out

def nltkPOS_to_wnPOS(pos, posType):
    if posType == 'F' or posType == 'A':
        if pos.startswith('NN'):
            return wn.NOUN
        elif pos.startswith('VB'):
            return wn.VERB
        elif pos.startswith('JJ'):
            return wn.ADJ
        elif pos.startswith('RB'):
            return wn.ADV
        else:
            return ''
    elif posType == 'N':
        if pos.startswith('NN'):
            return wn.NOUN
        elif pos.startswith('JJ'):
            return wn.ADJ
        else:
            return ''
    elif posType == 'n':
        if pos.startswith('NN'):
            return wn.NOUN
        else:
            return ''



def load_docs(directory, stopwords, stemmer, posType, tokenType):
    filenames = []
    docs = []
    if stemmer == 'L':
        lemmatizer = WordNetLemmatizer()
    for file in os.listdir(directory):
        infile=open(directory+'/'+file)
        lines = infile.read()
        filenames.append(file)

        # Check for mixed alphanumeric token condition:
        # if tokenType == 'a':
        #     re.sub(r'[^\w]', '', lines)
        a = nltk.word_tokenize(lines)
        a = np.array(a)
        if stopwords != 'none':
            a = [w for w in a if not w in stopwords] # Remove Stopwords
        # KEEP SPECIFIED TOKENS ########
        if tokenType == 'A':
            nums = [1,2,3,4,5,6,7,8,9]
            for i in range(len(a)):
                if len(a[i]) == 1 and a[i].isalpha() == False and a[i] not in nums:
                    a[i] = ''
            a = [word for word in a if word != '']
            # Keep everything except single character non-alphanumeric characters

        elif tokenType == 'a':
            a = [word for word in a if len(re.findall(r'[^a-zA-Z0-9 ]', word)) != 1 ]
            # Same as 'A' except remove symbols from mixed symbol-alphanumeric tokens

        elif tokenType == 'N':
            nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            a = [word for word in a if word.isalpha() or word in nums]
            # only alphanumeric tokens, but include numbers only tokens (e.g. “689”)

        elif tokenType == 'n':
            a = [word for word in a if word.isalpha()]
            # only alphanumeric tokens, but does not include numbers only tokens



        if stemmer != 'N':
            if stemmer == 'B': # Use Porter Stemmer
                ps = PorterStemmer()
                a = [ps.stem(w) for w in a] # Stem words
                a = nltk.pos_tag(a) # POS tag
                a = [(x, nltkPOS_to_wnPOS(y, posType)) for x, y in a]

            elif stemmer == 'L': # POS tag then Lemmatize
                a = nltk.pos_tag(a)
                a = [(x, nltkPOS_to_wnPOS(y, posType)) for x, y in a]
                a = [lemmatizer.lemmatize(x,y) for x,y in a if y != '']
            else:
                print("Invalid Lemm/Stem parameter: ", stemmer)
        else:
            a = nltk.pos_tag(a) # POS Tag if not stemming
            a = [(x, nltkPOS_to_wnPOS(y, posType)) for x, y in a]
        docs.append(np.char.lower(a)) # Add to matrix of docs

    return docs, filenames


def get_stopwords(filename):
    if filename == 'none': return 'none'
    elif filename == 'nltk': return set(stopwords.words('english'))
    else:
        rd = open(filename, "r")
        out = []
        while True:
            line = rd.readline()
            if not line:
                break
            out.append(line.strip())
        rd.close()
        return out

def create_model(numTopics, docs, vmType, alpha):
    dct = Dictionary(docs)
    corpus = [dct.doc2bow(line) for line in docs]

    # Build Vector Models ######
    # Binary Model
    if vmType == 'B':
        model = TfidfModel(corpus, smartirs='bnn') # Binary term frequency weighting

    # TFIDF Model
    elif vmType == 'T':
        model = TfidfModel(corpus, smartirs='tfn') # fit tfidf

    # Term Frequency model
    elif vmType == 't':
        model = TfidfModel(corpus, smartirs='tnn') # Term frequency only

    else:
        print("Invalid Vector Model parameter.")

    # Build LDA Model ##########
    corpus = model[corpus]
    lda = LdaModel(corpus=corpus, id2word=dct, num_topics=numTopics, alpha=alpha)
    return lda

# TODO: Save model using save() to output file name in params

def main():
    default_params = [8,'B','auto','n','A','L','nltk']
    args = list(sys.argv)
    directory = args[1]
    output = args[2]
    if len(sys.argv) < 4:
        params = default_params
    else:
        paramList = args[3]
        params = load_params(paramList)
    stopwords = get_stopwords(params[-1])

    docs, filenames = load_docs(directory, stopwords, params[5], params[4],params[3] )
    lda = create_model(params[0],docs, params[1], params[2])

    print(lda)



main()