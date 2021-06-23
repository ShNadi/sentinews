import pandas as pd
from sentinews.preprocessing import clean_text as pp
from sentinews.wordvector import generate_word2vec as wtv
from sentinews.sentiment import calculate_sentiment as st

def main_function(operation, **kwargs):
    """
    Set operation to preprocessing: for removing stopwords, html tags and punctuations
    Set operation to word2vec: for building wordvectors from the corpus
    Set operation to similar words: for most n similar words to word w
    Set operation to check similarity: for finding similarity between word1 and word2

    :param operation: Determines the type of operation
    :type operation: Str
    """
    word = kwargs.get('word', None)    # Input parameter for function check_most_similar()
    n = kwargs.get('n', None)          # Input parameter for function check_most_similar()
    word1 = kwargs.get('word1', None)  # Input parameter for function check_similarities()
    word2 = kwargs.get('word2', None)  # Input parameter for function check_similarities()

    # ********************** Pre-processing *********************
    if operation == 'preprocessing':
        # Read the original dataset
        df = pd.read_csv('../data/raw/news-dataset--2021-05-11.csv') # Period of six month

        # clean dataset including removing stopwords, html tags, newlines, and drop nulls
        pp.clean(df)

    # ********************** Word2vec *********************
    if operation == 'word2vec':
        df = pd.read_csv('../data/processed/news-dataset--2021-05-11.csv')
        wtv.train_word2vec_model(df)

    if operation == 'similar words':
        wtv.check_most_similar(word, n)

    if operation == 'check similarity':
        wtv.check_similarities(word1, word2)

    # ********************** Sentiment *********************
    if operation == 'sentiment':
        df = pd.read_csv('../data/processed/news-dataset--2021-05-11.csv')
        df = df.head(5)
        st.sentiment_words(df)

from pathlib import Path

if __name__=='__main__':
    # main_function('preprocessing') # Preprocess the original dataset (text column) and add it to clean_text column

    # main_function('word2vec') # Build wordvectors from the corpus

    main_function('similar words', word='immigratie', n=5) # find most n similar words to word w

    # main_function('check similarity', word1='corona', word2='vaccin') # find similarity between word1 and word2

    # main_function('sentiment') # find sentiment of each document in the dataset

    # main_function('sentiment')

