import pandas as pd
import re



# # print(df.loc[0, 'text'])
#
# filter_list = ['immigr', 'migrant', 'migratie', 'asielzoeker', 'vluchteling', 'vreemdeling', 'illegalen', 'allochto',
#      'gastarbeider',  'nieuwe Nederlander', 'etnische minderhe', 'afkomst', 'land van herkomst', 'moslim',
#      'NOT vreemdelingenlegioen']
# for word in filter_list:
#     if re.search(word, df.loc[0, 'text']):
#         print('1')

# Filter OutGroups articles including any subset of the words present in the filter list
def filter_art(df):
    df['out_group']=0
    for index, row in df.iterrows():
        filter_list = ['immigr', 'migrant', 'migratie', 'asielzoeker', 'vluchteling', 'vreemdeling', 'illegalen',
                   'allochto',
                   'gastarbeider', 'nieuwe Nederlander', 'etnische minderhe', 'afkomst', 'land van herkomst', 'moslim',
                   'NOT vreemdelingenlegioen']
        for word in filter_list:
            if re.search(word, df.loc[index, 'text']):
                df.loc[index, 'out_group']=1
    df = df[df['out_group']==1]
    return df

# Filter articles including 'Nederland' or a name of cities or states in the Netherlands
def filter_cities(df):
    df['netherlands']=0
    cities_df = pd.read_csv('../../../data/cites_list/cities.csv', sep=';')
    cities_name = cities_df['Naam_2'].tolist()
    states_name = cities_df['Naam_4'].tolist()
    living_areas_filter = cities_name + states_name
    living_areas_filter.append('Nederland')
    living_areas_filter = set(living_areas_filter)
    for index, row in df.iterrows():
        if any(x in df.loc[index, 'text'] for x in living_areas_filter):
            df.loc[index, 'netherlands'] = 1
        # else:
        #     df.loc[index, 'netherlands'] = 0
    df = df[df['netherlands']==1]
    return df



if __name__=='__main__':
    df = pd.read_csv('../../../data/processed/news-dataset--2021-05-11.csv')
    df_filter_art = filter_art(df)
    df_filter_city = filter_cities(df_filter_art)
    df_filter_city.to_csv('../../../data/processed/filtered_news.csv', index=False)


