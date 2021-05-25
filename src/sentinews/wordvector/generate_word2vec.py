from gensim.models import Word2Vec
import pandas as pd
from pathlib import Path
import dill
from gensim.models import KeyedVectors


def train_word2vec_model(df):
    df.dropna(subset=['clean_text'], inplace=True)
    sent = [row.split(' ') for row in df['clean_text']]
    model = Word2Vec(sent, min_count=1, size=100, workers=3, window=5, sg=1)
    path = Path(__file__).parent / "../../../results/models/word2vec.model"
    # model.save(path)
    with open(path, 'wb') as f:
        dill.dump(model, f)

def check_most_similar(word, n):
    path = Path(__file__).parent / "../../../results/models/word2vec.model"
    with open (path, 'rb') as f:
        model = dill.load(f)
    sims = model.wv.most_similar(word, topn=n)
    print(sims)

def check_similarities(word1, word2):
    path = Path(__file__).parent / "../../../results/models/word2vec.model"
    with open(path, 'rb') as f:
        model = dill.load(f)
    sims = model.wv.similarity(word1, word2)
    print(sims)

# Turney 2002- score sentiment for sentence
def check_similarities(word1, word2):
    path = Path(__file__).parent / "../../../results/models/word2vec.model"
    with open(path, 'rb') as f:
        model = dill.load(f)
    sims = model.wv.similarity(word1, word2)
    print(sims)

if __name__ == '__main__':
    # df = pd.read_csv('../../../data/processed/news-dataset--2010-04-21.csv')
    # df.dropna(subset=['clean_text'], inplace=True)
    # train_word2vec_model(df)
    # model = Word2Vec.load("../../../results/models/word2vec.model")
    # sims = model.wv.most_similar('vaccin', topn=10)
    # print(sims)
    # check_most_similar('slecht', 5)
    # check_similarities('corona', 'vaccin')

    # Get numpy vector of a word
    # Load back with memory-mapping = read-only, shared across processes.
    model = Word2Vec.load("../../../results/models/word2vec.model")
    vector = model.wv['goed']  # Get numpy vector of a word
    print(vector)

