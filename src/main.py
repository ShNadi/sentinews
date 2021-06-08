import pandas as pd
from sentinews.preprocessing import clean_text as pp
from sentinews.wordvector import generate_word2vec as wtv

def main_function(operation, **kwargs):
    """
    Set operation to preprocessing: for removing stopwords, html tags and punctuations
    Set operation to word2vec: for building wordvectors from the corpus
    Set operation to similar words: for most n similar words to word w
    Set operation to check similarity: for finding similarity between word1 and word2

    :param operation: Determines the type of operation
    :type operation: Str
    """
    word = kwargs.get('word', None)
    n = kwargs.get('n', None)
    word1 = kwargs.get('word1', None)
    word2 = kwargs.get('word2', None)

    if operation == 'preprocessing':
        # Read the original dataset
        df = pd.read_csv('../data/raw/news-dataset--2021-05-11.csv') # Period of six month

        # clean dataset including removing stopwords, html tags, newlines, and drop nulls
        pp.clean(df)


    if operation == 'word2vec':
        df = pd.read_csv('../data/processed/news-dataset--2021-05-11.csv')
        wtv.train_word2vec_model(df)

    if operation == 'similar words':
        wtv.check_most_similar(word, n)

    if operation == 'check similarity':
        wtv.check_similarities(word1, word2)

if __name__=='__main__':
    # main_function('preprocessing') # Preprocess the original dataset (text column) and add it to clean_text column

    # main_function('word2vec') # Build wordvectors from the corpus

    # main_function('similar words', word='islam', n=5) # find most n similar words to word w

    main_function('check similarity', word1='corona', word2='vaccin') # find similarity between word1 and word2

