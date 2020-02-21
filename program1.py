import nltk.corpus
import CorpusReader
import pandas as pd

brown = nltk.corpus.brown
sotu = nltk.corpus.state_union

CR1 = CorpusReader.CorpusReader_TFIDF(brown)
#CR2 = CorpusReader.CorpusReader_TFIDF(sotu)

df1 = CR1.getTF()
#df2 = CR2.getTF()

print("BROWN::::::::::")
for x in df1:
    print(x)
print(" ")
print(len(df1))
# print("SOTU::::::::::::")
# for x in df2:
#     print(x)


#print(df1.loc[])