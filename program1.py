import nltk.corpus
import CorpusReader

brown = nltk.corpus.brown
sotu = nltk.corpus.state_union

corpus_list = [brown, sotu]

print("Number of words in brown corpus: " + str(len(brown.words())))
print("Number of words in sotu corpus: " + str(len(sotu.words())))

for corpus in corpus_list:
    CR = CorpusReader.CorpusReader_TFIDF(corpus, )