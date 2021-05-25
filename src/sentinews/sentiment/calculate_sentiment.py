from gensim.models import Word2Vec
import pandas as pd
from scipy import spatial


def sentiment_calc(df):
    # Sum of sentence scores, initial value = 0


    # Load word vectore for word 'goed'
    model = Word2Vec.load("../../../results/models/word2vec.model") # Load model
    vector_pos = model.wv['goed']  # Get numpy vector of word 'goed'
    vector_neg = model.wv['slecht']  # Get numpy vector of word 'slecht'


    # fetch word vector for each word in each row of dataset
    for i in range(len(df)):
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



if __name__=='__main__':
    df = df = pd.read_csv('../../../data/processed/news-dataset--2010-04-21.csv')
    df.dropna(subset=['clean_text'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    sentiment_calc(df)
# function ss(sentence s)
# {
# a = wordvec(good)
# b = wordvec(bad)
#
# sum = 0
# for each word w in sentence s:
# 	sw = cos(w,a)-cos(w,b)
# 	sum +=sw
#
# return sum
# }
#
# # cos = PMI
