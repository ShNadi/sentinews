import pandas as pd
from sentinews.preprocessing import clean_text as pp
# from sentinews.wordvector import generate_word2vec as wtv

def main_function(operation):
    if operation == 'preprocessing':
        # Read the original dataset
        df = pd.read_csv('../data/raw/news-dataset--2021-05-11.csv') # Period of six month

        # clean dataset including removing stopwords, html tags, newlines, and drop nulls
        pp.clean(df)


    # if operation == 'word2vec':
    #     df = pd.read_csv('../data/processed/news-dataset--2010-04-21.csv')
    #     wtv.train_word2vec_model(df)
    #
    # if operation == 'most similarity':
    #     wtv.check_most_similar('islam', 5)
    #
    # if operation == 'check similarites':
    #     wtv.check_similarities('corona', 'vaccin')

if __name__=='__main__':
    main_function('preprocessing')
    # main_function('check similarites')