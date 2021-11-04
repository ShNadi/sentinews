# Initialize a new word2vec model with pre-trained model weights and fine tune it with my own corpus

from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
import pandas as pd

# https://github.com/clips/dutchembeddings
# model = Word2Vec.load_word2vec_format("../../../sentinews/pre-trained/wikipedia-320.txt",
model = word2vec.KeyedVectors.load_word2vec_format("../../../pre-trained/wikipedia-160.txt", binary=False)
katvec = model['dodelijk ']
print(model.most_similar('dodelijk '))
print(katvec)


def tuned_word2vec_model():
    df = pd.read_csv('../../../data/processed/news-dataset--2010-04-21.csv')
    df.dropna(subset=['clean_text'], inplace=True)
    my_corpus = df['clean_text']
    model = Word2Vec(vector_size=300, min_count=1)
    model.build_vocab(my_corpus)
    model.intersect_word2vec_format("../../../pre-trained/wikipedia-160.txt", binary=True, lockf=1.0)
    model.train(my_corpus, total_examples=len(my_corpus))

