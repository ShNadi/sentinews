import pandas as pd
from nltk.corpus import stopwords
import re
import string
from pathlib import Path


desired_width = 320
pd.set_option('display.width', desired_width)

# Download list of NLTK Dutch stopwords
stop_words = stopwords.words('Dutch')


def clean(df):
    # Drop null values in text column
    df.dropna(subset=['text'], inplace=True)

    # Remove Dutch stopwords
    df['clean_text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # Remove htm tags
    df.clean_text.apply(lambda x: re.sub('<[^<]+?>', '', x))

    # Remove newlines
    df.text.replace('\n', '', inplace=True)

    # Remove punctuation
    df['clean_text'] = df['clean_text'].apply(remove_punctuations)

    # Drop null values in clean_text
    df.dropna(subset=['clean_text'], inplace=True)

    path = Path(__file__).parent / "../../../data/processed/news-dataset--2021-08-11.csv"
    df.to_csv(path, index=False)


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


if __name__ == '__main__':
    df = pd.read_csv('../../../data/raw/ninemonths_news.csv')
    clean(df)


