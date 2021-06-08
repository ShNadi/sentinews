import pandas as pd
from sentinews.preprocessing import clean_text as pp
from sentinews.wordvector import generate_word2vec as wtv

def main_function(operation):
    """
    Set operation to preprocessing: for removing stopwords, html tags and punctuations
    Set operation to word2vec: for building wordvectors from the corpus

    :param operation: Determines the type of operation
    :type operation: Str
    """
    if operation == 'preprocessing':
        # Read the original dataset
        df = pd.read_csv('../data/raw/news-dataset--2021-05-11.csv') # Period of six month

        # clean dataset including removing stopwords, html tags, newlines, and drop nulls
        pp.clean(df)


    if operation == 'word2vec':
        df = pd.read_csv('../data/processed/news-dataset--2021-05-11.csv')
        wtv.train_word2vec_model(df)

    if operation == 'most similarity':
        wtv.check_most_similar('islam', 5)

    if operation == 'check similarites':
        wtv.check_similarities('corona', 'vaccin')

if __name__=='__main__':
    # main_function('preprocessing') # Preprocess the original dataset (text column) and add it to clean_text column

    main_function('word2vec') # Build wordvectors from the corpus

    # main_function('check similarites')