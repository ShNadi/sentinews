# Filter outgroups from the whole set of articles
# Data provided by "online news database for a period of 8 months" : Start date: 10-11-2020   End date: 11-05-2021
# Last edit on 19-08-2021 by "Shiva Nadi"


# Packages
import pandas as pd
import re


# Exclude articles related to theater, books, films, art, and lifestyle
def filter_art(df):
    art_list = ['theater', 'boek', 'film', 'kunst', 'roman']
    for index, row in df.iterrows():
        df.loc[index, 'art'] = 0
        if any(x in df.loc[index, 'text'] for x in art_list):
            df.loc[index, 'art'] = 1
    df = df[df['art'] == 0]
    return df


# Filter OutGroups articles including any subset of the words present in the filter list
def filter_outgroups(df):
    # In this new version word 'afkomst' is excluded from the filter list
    filter_list = ['immigr', 'migrant', 'migratie', 'asielzoeker', 'vluchteling', 'vreemdeling', 'illegalen',
                   'allochto', 'gastarbeider', 'nieuwe nederlander', 'etnische minderhe',
                   'land van herkomst', 'moslim']
    filter_list2 = ['vreemdelingenlegioen']
    for index, row in df.iterrows():
        df.loc[index, 'out_group'] = 0
        for word in filter_list:
            if re.search(word, df.loc[index, 'text']):
                df.loc[index, 'out_group'] = 1
        for word in filter_list2:
            if re.search(word, df.loc[index, 'text']):
                df.loc[index, 'out_group'] = 0
    df = df[df['out_group'] == 1]
    return df


# Filter articles including 'Nederland' or a name of cities or states in the Netherlands
def filter_location(df):
    cities_df = pd.read_csv('../../../data/cites_list/cities.csv', sep=';')
    cities = cities_df['Naam_2'].tolist()
    states = cities_df['Naam_4'].tolist()
    directions = cities_df['Naam_6'].tolist()
    living_areas_filter = cities + states + directions
    # living_areas_filter.append('Nederland')
    # living_areas_filter.append('Nederlands')
    search_filter = ['nederland', 'holland']

    for i in range(len(living_areas_filter)):
        living_areas_filter[i] = living_areas_filter[i].lower()

    living_areas_filter = set(living_areas_filter)

    for index, row in df.iterrows():
        df.loc[index, 'netherlands'] = 0
        if any(x in df.loc[index, 'text'] for x in living_areas_filter):
            df.loc[index, 'netherlands'] = 1
        for word in search_filter:
            if re.search(word, df.loc[index, 'text']):
                df.loc[index, 'netherlands'] = 1

    df_location = df[df['netherlands'] == 1]
    return df_location


if __name__ == '__main__':
    df_news = pd.read_csv('../../../data/processed/news-dataset--2021-05-11.csv')
    df_news['text'] = df_news['text'].str.lower()
    print("Size of original dataset:", df_news.shape)

    df_news = filter_art(df_news)
    print("Size of dataset after filtering articles related to books, theater,...:", df_news.shape)
    df_news.to_csv('../../../data/processed/filtered_art.csv', index=False)

    df_news = filter_outgroups(df_news)
    print("Size of dataset after filtering out groups:", df_news.shape)
    df_news.to_csv('../../../data/processed/filtered_outgroups.csv', index=False)

    df_news = filter_location(df_news)
    print("Size of dataset after filtering articles related to the Nederlands:", df_news.shape)
    df_news.to_csv('../../../data/processed/filtered_news.csv', index=False)
    # df_news.to_csv('../../../data/processed/news-dataset_filters.csv', index=False)


