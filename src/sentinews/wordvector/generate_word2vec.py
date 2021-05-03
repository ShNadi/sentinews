from gensim.models import Word2Vec
import pandas as pd


def generate_word2vec_model(df):
    sent = [row.split(' ') for row in df['clean_text']]
    model = Word2Vec(sent, min_count=1, size=50, workers=3, window=3, sg=1)
    model.save("word2vec.model")


if __name__ == '__main__':
    df = pd.read_csv('../../../data/processed/news-dataset--2010-04-21.csv')
    df.dropna(subset=['clean_text'], inplace=True)
    generate_word2vec_model(df)
    model = Word2Vec.load("word2vec.model")
    sims = model.wv.most_similar('vaccin', topn=10)
    print(sims)
