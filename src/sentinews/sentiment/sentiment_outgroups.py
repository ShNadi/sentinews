import pandas as pd
import numpy as np
import time
from nltk.corpus import stopwords
import re
from gensim.models import Word2Vec
import nltk
import gensim.models.keyedvectors as word2vec


def sentence_score(sentence, model, neg_vec):
    sent_score = 0
    for word in sentence.split():           # iterates over each word in the sentence
        sw_neg = 0                          # Similarity between word w and neg_vec
        if word in model.wv.vocab:
            for x in neg_vec:
                sw_neg += model.wv.similarity(word, x)
            sent_score += sw_neg
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
    # df.to_csv('../../../data/processed/news_sentiment_score.csv', index=False)
    print(df)


if __name__ == '__main__':
    start = time.time()
    df = pd.read_csv('../../../data/processed/filtered_news.csv')
    df = df.head(3)
    stop_words = stopwords.words('Dutch')

    df.text.replace('\n', '', inplace=True)
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df.text.apply(lambda x: re.sub('<[^<]+?>', '', x))
    df.text.replace('', np.nan, inplace=True)
    df.dropna(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # print(df)

    document_score(df)
    end = time.time()
    print("--- %s seconds ---" % (end - start))





