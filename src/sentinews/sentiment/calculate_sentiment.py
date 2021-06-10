from gensim.models import Word2Vec
import pandas as pd
from scipy import spatial
import nltk


def sentiment_sentence_to_word(sentence):
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

def document_score(df):
    for i in range(len(df)):                                      # For each document in each row of the dataset
        doc_score = 0                                             # Initial score of the document is 0
        sentence = nltk.tokenize.sent_tokenize(df.loc[i,'text'])  # tokenize the document to the sentences
        for s in sentence:                                        # For each sentence in the list of sentences
            doc_score+= sentiment_sentence_to_word(s)             # Calculate sentence's score & add it to the doc_score
        df.loc[i,'cos_score_sentence']=doc_score                  # write the document's score in a new column in the df


def sentiment_list(df, pos_list, neg_list):
    model = Word2Vec.load("../../../results/models/word2vec.model") # Load model
    pos_vec = []
    neg_vec = []
    for pos_word in pos_list:                    # for every positive word in pos_list load the wordvector
        pos_vec.append(model.wv[pos_word])       # the wordvector for all positive words are stored in pos_vec

    for neg_word in neg_list:                   # for every negative word in neg_list load the wordvector
        neg_vec.append(model.wv[neg_word])      # the wordvector for all negative words are stored in neg_vec


    for i in range(len(df)):                    # fetch wordvector for each word in each row of dataset
        sum_scores = 0                          # Sum of word scores, initial value = 0
        text = df.loc[i, "clean_text"]
        word = text.split()
        for w in word:
            sw_pos = 0                          # Similarity between word w and pos_vec
            sw_neg = 0                          # Similarity between word w and neg_vec
            word_vector = model.wv[w]
            for x in pos_vec:
                sw_pos+= 1 - spatial.distance.cosine(word_vector, x)
            for y in neg_vec:
                sw_neg+= 1 - spatial.distance.cosine(word_vector, y)
            sum_scores += sw_pos - sw_neg
        df.loc[i,'cos_score_list'] = sum_scores
    print(df)

if __name__=='__main__':
    df = pd.read_csv('../../../data/processed/news-dataset--2021-05-11.csv')
    df = df.head(5)





