from gensim.models import Word2Vec
import pandas as pd
from scipy import spatial
import gensim.models.keyedvectors as word2vec
from sklearn.metrics.pairwise import cosine_similarity

def sentiment_calc(df):
    # Load word vectore for word 'goed'
    model = Word2Vec.load("../../../results/models/word2vec.model") # Load model
    vector_pos = model.wv['goed']  # Get numpy vector of word 'goed'
    vector_neg = model.wv['slecht']  # Get numpy vector of word 'slecht'


    # fetch word vector for each word in each row of dataset
    for i in range(len(df)):
        # Sum of sentence scores, initial value = 0
        sum_scores = 0
        text = df.loc[i, "clean_text"]
        word = text.split()
        for w in word:
            word_vector = model.wv[w]
            sw_pos = 1 - spatial.distance.cosine(word_vector, vector_pos)
            sw_neg = 1 - spatial.distance.cosine(word_vector, vector_neg)
            sum_scores += sw_pos - sw_neg
        df.loc[i,'cos_score'] = sum_scores
    print(df)
    df.to_csv('../../../data/processed/news_sentiment.csv', index=False)

def sentiment_calc_pretrained(df):
    model = word2vec.KeyedVectors.load_word2vec_format("../../../pre-trained/wikipedia-160.txt", binary=False)
    vector_pos = model['goed']
    vector_neg = model['slecht']

    for i in range(len(df)):
        # Sum of sentence scores, initial value = 0
        sum_scores = 0
        text = df.loc[i, "clean_text"]
        word = text.split()
        for w in word:
            if w in model.vocab:
                word_vector = model[w]
                sw_pos = 1 - spatial.distance.cosine(word_vector, vector_pos)
                sw_neg = 1 - spatial.distance.cosine(word_vector, vector_neg)
                sum_scores += sw_pos - sw_neg
        df.loc[i,'cos_score_pretrained'] = sum_scores
    print(df)
    df.to_csv('../../../data/processed/news_sentiment_pretrained.csv', index=False)


if __name__=='__main__':
    df = df = pd.read_csv('../../../data/processed/news_sentiment.csv')
    df.dropna(subset=['clean_text'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    sentiment_calc_pretrained(df)
