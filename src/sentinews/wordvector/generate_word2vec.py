from gensim.models import Word2Vec
import pandas as pd
from pathlib import Path
import dill
from gensim.models import KeyedVectors


def train_word2vec_model(df):
    """
    Builds wordvectors using gensim library on the clean_text column of the input dataset
    :param df:
    :type df: DataFrame
    """
    df.dropna(subset=['clean_text'], inplace=True)
    sent = [row.split(' ') for row in df['clean_text']]
    model = Word2Vec(sent, min_count=1, size=300, workers=3, window=5, sg=1)
    path = Path(__file__).parent / "../../../results/models/word2vec.model"
    # model.save(path)
    with open(path, 'wb') as f:
        dill.dump(model, f)

def check_most_similar(word, n):
    """
    find the most n similar words to 'word' in the vector space

    :param word: The word we would like to find similar words
    :type word: str
    :param n: number of most similar words
    :type n: int
    """
    path = Path(__file__).parent / "../../../results/models/word2vec.model"
    with open (path, 'rb') as f:
        model = dill.load(f)
    sims = model.wv.most_similar(word, topn=n)
    print(sims)

def check_similarities(word1, word2):
    """
    finds similarity between wordvector 1 and wordvector 2
    :param word1:
    :type word1: str
    :param word2:
    :type word2: str
    """
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
    df = pd.read_csv('../../../data/processed/news-dataset--2010-04-21.csv')
    df.dropna(subset=['clean_text'], inplace=True)
    train_word2vec_model(df)
    # model = Word2Vec.load("../../../results/models/word2vec.model")
    # sims = model.wv.most_similar('vaccin', topn=10)
    # print(sims)
    # check_most_similar('slecht', 5)
    # check_similarities('corona', 'vaccin')

    # Get numpy vector of a word
    # Load back with memory-mapping = read-only, shared across processes.
    # model = Word2Vec.load("../../../results/models/word2vec.model")
    # vector = model.wv['goed']  # Get numpy vector of a word
    # print(vector)

