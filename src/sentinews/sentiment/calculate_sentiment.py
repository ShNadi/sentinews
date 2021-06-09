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


def sentiment_sentence_to_word(sentence):
    """Calculates the sentiment of the sentence considering the wordvector of the word 'goed' as positive vector and the
     word vector of the word 'slecht' as negative vector. The wordvectores are trained by our own corpus.
     For every sentence in text column, the wordvector of the word is loaded and cosine distance of the word and
     positive vector(word goed) and negative vector(word slecht) is calculated as sw_pos and sw_neg respectively. The
     difference between these two variable is the sentiment number for each word in the sentence. sum of word
     sentiments makes sentence sentiment.

    :param sentence: The document sentences which are passed to this function through sentiment_calc_sentence()
    :type sentence:str
    :return: Sentiment number of the sentence
    :rtype: float
    """
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

def sentiment_sentence(df):
    """ Tokenize each document to sentences and for each sentence calls sentiment_sentence_to_word(sentence) to
    calculate sentence sentiments. sum of sentence sentiments makes the document sentiment.
    This function works with original text field without preprocessing.

    :param df: The dataframe including clean_text column
    :type df: DataFrame
    """
    for i in range(len(df)):
        sum = 0
        sentence = nltk.tokenize.sent_tokenize(df.loc[i,'text'])
        for s in sentence:
            sum+= sentiment_sentence_to_word(s)
        df.loc[i,'cos_score_sentence']=sum


def sentiment_list(df, pos_list, neg_list):
    """Calculates the sentiment of the document word based considering a list of positive and a list of negative words.
     The wordvectores are trained by our own corpus.
     For every word in clean_text column, the wordvector of the word is loaded and cosine distance of the word and
     all the wordvectores in the pos_vec is calculated.

    :param df: The dataframe including clean_text column
    :type df: DataFrame
    :param pos_list:
    :type pos_list: list
    :param neg_list:
    :type neg_list: list
    """
    # Load word vectore for word 'goed'
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
    # df.dropna(subset=['clean_text'], inplace=True)
    # df.reset_index(drop=True, inplace=True)
    # sentiment_calc_pretrained(df)
    df = df.head(5)

    # sentiment_calc_sentence(df)
    # print(df)
    # pos_list = ['prima', 'goed']
    # neg_list = ['slecht', 'kwaad']
    # sentiment_list(df, pos_list, neg_list)

    sentiment_words(df)



