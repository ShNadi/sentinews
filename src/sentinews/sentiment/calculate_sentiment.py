from gensim.models import Word2Vec
import pandas as pd
from scipy import spatial
import gensim.models.keyedvectors as word2vec
import nltk
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
#
# function sa(article d)
# {
# tokenize sentence(d)
# sum = 0
# for each sentence s in article d:
#   	sum += ss(s)
# return sum
# }

def sentiment_sentence_to_word(sentence):
    # Load word vectore for word 'goed'
    model = Word2Vec.load("../../../results/models/word2vec.model")  # Load model
    vector_pos = model.wv['goed']  # Get numpy vector of word 'goed'
    vector_neg = model.wv['slecht']  # Get numpy vector of word 'slecht'

    sum =0
    for word in sentence.split():
        if word in model.wv.vocab:
            word_vector = model.wv[word]
            sw_pos = 1 - spatial.distance.cosine(word_vector, vector_pos)
            sw_neg = 1 - spatial.distance.cosine(word_vector, vector_neg)
            sum += sw_pos - sw_neg
    return sum

def sentiment_calc_sentence(df):
    for i in range(len(df)):
        sum =0
        sentence = nltk.tokenize.sent_tokenize(df.loc[i,'text'])
        for s in sentence:
            sum+=sentiment_sentence_to_word(s)
        df.loc[i,'score_sentence']=sum


def sentiment_calc_list(df, pos_list, neg_list):
    # Load word vectore for word 'goed'
    model = Word2Vec.load("../../../results/models/word2vec.model") # Load model
    pos_vec = []
    neg_vec = []
    for pos_word in pos_list:
        pos_vec.append(model.wv[pos_word])

    for neg_word in neg_list:
        neg_vec.append(model.wv[neg_word])


    # fetch word vector for each word in each row of dataset
    for i in range(len(df)):
        # Sum of sentence scores, initial value = 0
        sum_scores = 0
        text = df.loc[i, "clean_text"]
        word = text.split()
        for w in word:
            sw_pos = 0
            sw_neg = 0
            word_vector = model.wv[w]
            for x in pos_vec:
                sw_pos+= 1 - spatial.distance.cosine(word_vector, x)
            for y in neg_vec:
                sw_neg+= 1 - spatial.distance.cosine(word_vector, y)
            sum_scores += sw_pos - sw_neg
        df.loc[i,'cos_score'] = sum_scores
    print(df)

if __name__=='__main__':
    df = df = pd.read_csv('../../../data/processed/news_sentiment.csv')
    # df.dropna(subset=['clean_text'], inplace=True)
    # df.reset_index(drop=True, inplace=True)
    # sentiment_calc_pretrained(df)
    df = df.head(5)

    # sentiment_calc_sentence(df)
    # print(df)
    pos_list = ['prima', 'goed']
    neg_list = ['slecht', 'kwaad']
    sentiment_calc_list(df, pos_list, neg_list)



