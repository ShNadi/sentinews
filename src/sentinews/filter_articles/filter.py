import pandas as pd
import re


#
# Exclude articles related to theater, books, films, art, and lifestyle
def exclude_art(df):
    filter_art = ['theater', 'boek', 'film', 'kunst', 'roman']
    for index, row in df.iterrows():
        df.loc[index, 'art'] = 0
        if any(x in df.loc[index, 'text'] for x in filter_art):
            df.loc[index, 'art'] = 1
    df = df[df['art']==0]
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
    df = df[df['out_group']==1]
    return df

# Filter articles including 'Nederland' or a name of cities or states in the Netherlands
def filter_cities(df):
    cities_df = pd.read_csv('../../../data/cites_list/cities.csv', sep=';')
    cities_name = cities_df['Naam_2'].tolist()
    states_name = cities_df['Naam_4'].tolist()
    living_areas_filter = cities_name + states_name
    living_areas_filter.append('Nederland')
    living_areas_filter = set(living_areas_filter)

    for index, row in df.iterrows():
        df[index, 'netherlands'] = 0
        if any(x in df.loc[index, 'text'] for x in living_areas_filter):
            df.loc[index, 'netherlands'] = 1

    df = df[df['netherlands']==1]
    return df



if __name__=='__main__':
    df = pd.read_csv('../../../data/processed/news-dataset--2021-05-11.csv')
    print(df.shape)

    df_filter_art = exclude_art(df)
    print(df_filter_art.shape)
    df_filter_art.to_csv('../../../data/processed/filtered_art.csv', index=False)

    df_filter_outgrp = filter_outgroups(df_filter_art)
    print(df_filter_outgrp.shape)
    df_filter_outgrp.to_csv('../../../data/processed/filtered_outgroups.csv', index=False)
    #
    df_filter_city = filter_cities(df_filter_outgrp)
    print(df_filter_city.shape)
    df_filter_city.to_csv('../../../data/processed/filtered_news.csv', index=False)


