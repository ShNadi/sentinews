# Filter outgroups from the whole set of articles
# Data provided by "online news database for a period of 8 months" : Start date: 10-11-2020   End date: 10-07-2020
# Last edit on 20-07-2021 by "Shiva Nadi"


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
    filter_list = ['immigr', 'migrant', 'migratie', 'asielzoeker', 'vluchteling', 'vreemdeling', 'illegalen',
                   'allochto', 'gastarbeider', 'nieuwe Nederlander', 'etnische minderhe', 'afkomst',
                   'land van herkomst', 'moslim', 'NOT vreemdelingenlegioen']
    for index, row in df.iterrows():
        df.loc[index,'out_group'] = 0
        for word in filter_list:
            if re.search(word, df.loc[index, 'text']):
                df.loc[index, 'out_group'] = 1
    df = df[df['out_group'] == 1]
    return df


# Filter articles including 'Nederland' or a name of cities or states in the Netherlands
def filter_location(df_a):
    cities_df = pd.read_csv('../../../data/cites_list/cities.csv', sep=';')
    cities = cities_df['Naam_2'].tolist()
    states = cities_df['Naam_4'].tolist()
    directions = cities_df['Naam_6'].tolist()
    living_areas_filter = cities + states + directions
    living_areas_filter.append('Nederland')
    living_areas_filter = set(living_areas_filter)

    for index, row in df_a.iterrows():
        df_a[index, 'netherlands'] = 0
        if any(x in df_a.loc[index, 'text'] for x in living_areas_filter):
            df_a.loc[index, 'netherlands'] = 1

    df_location = df_a[df_a['netherlands'] == 1]
    return df_location


if __name__ == '__main__':
    df = pd.read_csv('../../../data/processed/news-dataset--2021-05-11.csv')
    print(df.shape)

    df_filter_art = filter_art(df)
    print(df_filter_art.shape)
    df_filter_art.to_csv('../../../data/processed/filtered_art.csv', index=False)
    #
    # df_filter_outgrp = filter_outgroups(df_filter_art)
    # print(df_filter_outgrp.shape)
    # df_filter_outgrp.to_csv('../../../data/processed/filtered_outgroups.csv', index=False)
    #
    # df_filter_location = filter_location(df_filter_outgrp)
    # print(df_filter_location.shape)
    # df_filter_location.to_csv('../../../data/processed/filtered_news.csv', index=False)


