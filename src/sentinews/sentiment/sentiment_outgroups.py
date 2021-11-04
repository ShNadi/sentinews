import pandas as pd
import numpy as np
import time
from nltk.corpus import stopwords
import re
from gensim.models import Word2Vec
import nltk


def sentence_score(sentence, model, neg_vec):
    sent_score = 0
    for word in sentence.split():           # iterates over each word in the sentence
        sw_neg = 0                          # Similarity between word w and neg_vec
        if word in model.wv.vocab:
            for x in neg_vec:
                sw_neg += model.wv.similarity(word, x)
            avg_sw_neg = sw_neg/len(neg_vec)
            sent_score += avg_sw_neg         # Calculate the average negativity sentiment for each word
    return sent_score


def document_score(df):
    # Load the word2vec model trained on the whole dataset
    model = Word2Vec.load("../../../results/models/word2vec.model")
    # Load the pre-trained word2vec model trained on wikipedia-dutch
    # model = word2vec.KeyedVectors.load_word2vec_format("../../../pre-trained/wikipedia-160.txt", binary=False)

    # Load list of selected negative words from Subjectivity lexicon Dutch
    negative_file = open("../../../dic/negative_words_selected.txt", "r")
    neg_list = negative_file.readlines()
    neg_list = [x.replace('\n', '') for x in neg_list]
    neg_list = [x.replace(' ', '') for x in neg_list]

    # Lowercase all the words in the list of negative words
    for i in range(len(neg_list)):
        neg_list[i] = neg_list[i].lower()

    # for every negative word in neg_list load the wordvector
    neg_vec = []
    for neg_word in neg_list:
        if neg_word in model.wv.vocab:
            neg_vec.append(neg_word)  # the wordvector for all negative words are stored in neg_vec

    for i in range(len(df)):                                       # iterates over each document(each row of dataset)
        doc_score = 0                                              # Initialize document score
        sentence = nltk.tokenize.sent_tokenize(df.loc[i, 'text'])  # tokenize the document to the sentences
        for s in sentence:                                         # For each sentence in the list of sentences
            doc_score += sentence_score(s, model, neg_vec)          # Calculate sentence's score & add it to the
        df.loc[i, 'negativity_score'] = doc_score/len(sentence)    # write the document's score in a new

    # Normalizing negativity scores between 0 and 1- using min/max normalization
    max_value = df['negativity_score'].max()
    min_value = df['negativity_score'].min()
    df['normalized_score'] = (df['negativity_score'] - min_value) / (max_value - min_value)
    df['negativity_degree'] = df['normalized_score'].apply(
        lambda x: 1 if 0 <= x < 0.20 else 2 if 0.20 <= x < 0.40 else 3 if 0.40 <= x < 0.60 else 4 if 0.60 <= x < 0.80
        else 5)

    df.to_csv('../../../data/processed/outNL_negativity_sentiment.csv', index=False)
    # df.to_excel('../../../data/processed/outgroups_negativity_sentiment.xlsx', index=False)
    # df.to_csv('../../../data/processed/sample_ds_negativity_sentiment.csv', index=False)
    # df.to_csv('../../../data/processed/filtered_outgroups_negativity_sentiment.csv', index=False)
    # df.to_excel('../../../data/processed/sample_ds_negativity_sentiment.xlsx', index=False)
    print(df)


# def normalize_scores(df):
#     max_value = df['negativity_score'].max()
#     min_value = df['negativity_score'].min()
#     df['normalized_score'] = (df['negativity_score'] - min_value) / (max_value - min_value)
#     df['negativity_degree'] = df['negativity_score'].apply(
#         lambda x: 1 if 0 <= x < 0.25 else 2 if 0.25 <= x < 0.5 else 3 if 0.5 <= x < 0.75 else 4)
#     # df.to_csv('../../../data/processed/outgroups_negativity_sentiment.csv', index=False)
#     df.to_csv('../../../data/processed/sample_ds_negativity_sentiment.csv', index=False)


if __name__ == '__main__':
    start = time.time()
    df = pd.read_csv('../../../data/processed/filtered_news.csv')
    # df = pd.read_csv('../../../data/processed/filtered_outgroups.csv')
    # df = pd.read_csv('../../../data/processed/random_sample_ds.csv')
    stop_words = stopwords.words('Dutch')

    df.text.replace('\n', '', inplace=True)
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df.text.apply(lambda x: re.sub('<[^<]+?>', '', x))
    df.text.replace('', np.nan, inplace=True)
    df.dropna(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    document_score(df)
    end = time.time()
    print("--- %s seconds ---" % (end - start))





