import nltk.corpus
import CorpusReader
import pandas as pd

brown = nltk.corpus.brown
sotu = nltk.corpus.state_union

CR1 = CorpusReader.CorpusReader_TFIDF(brown) # Load CorpusReader_TFIDF objects for each corpus
CR2 = CorpusReader.CorpusReader_TFIDF(sotu)

df1 = CR1.tf_idf()  # Get tf_idf vector for each corpus for use below
df2 = CR2.tf_idf()


print("Corpus : brown")
dim = CR1.tf_idf_dim()
print(dim[:15])
for file in range(len(df1)):
    print('<',CR1.fileids()[file],'>','<',CR1.get_tfidf_vec(file),'>')
print("Cosine Similarity: ",CR1.cosine_sim(['ca01','ca02']))

print('\n','\n')
print("Corpus : State of the Union")
dim = CR2.tf_idf_dim()
print(dim[:15])
for file in range(len(df2)):
    print('<', CR2.fileids()[file], '>', '<', CR2.get_tfidf_vec(file), '>')
print("Cosine Similarity: ", CR2.cosine_sim(['1945-Truman.txt', '1961-Kennedy.txt']))