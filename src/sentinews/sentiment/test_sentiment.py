from gensim.models import Word2Vec
import pandas as pd
from scipy import spatial
import gensim.models.keyedvectors as word2vec
import nltk
from pathlib import Path

def sentiment_words(df):
    """Calculates the sentiment of the document considering the wordvector of the word 'goed' as positive vector and
    the word vector of the word 'slecht' as negative vector. The wordvectores are trained by our own corpus.
    For every word in clean_text column, the wordvector of the word is loaded and cosine distance of the word and
    positive vector( word goed) and negative vector(word slecht) is calculated as sw_pos and sw_neg respectively. The
    difference between these two variable is the sentiment number which is inserted in the dataframe as a new column.

    :param df: The dataframe including clean_text column
    :type df: DataFrame
    """
    # mod_path = Path(__file__).parent
    # relative_path_model = '../../../results/models/word2vec.model'
    # model_path = (mod_path / relative_path_model).resolve()
    model = Word2Vec.load('../results/models/word2vec.model') # Load model
    # model = Word2Vec.load("../../../results/models/word2vec.model") # Load model
    vector_pos = model.wv['goed']       # Get numpy vector of word 'goed', trained on our own corpus
    vector_neg = model.wv['slecht']     # Get numpy vector of word 'slecht', trained on our own corpus


    # fetch word vector for each word in each row of the dataset
    for i in range(len(df)):
        sum_scores = 0                  # Sum of word scores, initial value = 0
        text = df.loc[i, "clean_text"]  # Read the clean text of row i in the text variable
        word = text.split()             # Split text to words
        for w in word:                  # For each word in the list of words calculate the cosine distance
            word_vector = model.wv[w]
            sw_pos = 1 - spatial.distance.cosine(word_vector, vector_pos)
            sw_neg = 1 - spatial.distance.cosine(word_vector, vector_neg)
            sum_scores += sw_pos - sw_neg
        df.loc[i,'cos_score_words'] = sum_scores
    # df.to_csv('../../../data/processed/news_sentiment_word.csv', index=False)
    print(df)


def sentiment_pretrained(df):
    """Calculates the sentiment of the document considering the wordvector of the word 'goed' as positive vector and
    the word vector of the word 'slecht' as negative vector. The wordvectores are loaded form pre-trained vectors on
    wikipedia for Dutch language.
    For every word in clean_text column, the pre-trained wordvector of the word is loaded and cosine distance of the
    word and positive vector( word goed) and negative vector(word slecht) is calculated as sw_pos and sw_neg
    respectively. The difference between these two variable is the sentiment number which is inserted in the dataframe
    as a new column.
    :param df: The dataframe including clean_text column
    :type df: DataFrame
    """
    model = word2vec.KeyedVectors.load_word2vec_format("../../../pre-trained/wikipedia-160.txt", binary=False)
    vector_pos = model['goed']          # Load pre-trained word vector of word 'goed'
    vector_neg = model['slecht']        # Load pre-trained word vector of word 'slecht'

    for i in range(len(df)):
        sum_scores = 0                  # Sum of sentence scores, initial value = 0
        text = df.loc[i, "clean_text"]  # Read the clean text of row i in the text variable
        word = text.split()             # Split text to words
        for w in word:                  # For each word in the list of words calculate the cosine distance
            if w in model.vocab:
                word_vector = model[w]
                sw_pos = 1 - spatial.distance.cosine(word_vector, vector_pos)
                sw_neg = 1 - spatial.distance.cosine(word_vector, vector_neg)
                sum_scores += sw_pos - sw_neg
        df.loc[i,'cos_score_pretrained'] = sum_scores
    df.to_csv('../../../data/processed/news_sentiment_pretrained_words.csv', index=False)
