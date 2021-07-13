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

    # print(1 if re.search(word, df.loc[0, 'text']) else 0)
def filter_art(df):

    filter_list = ['immigr', 'migrant', 'migratie', 'asielzoeker', 'vluchteling', 'vreemdeling', 'illegalen',
                   'allochto',
                   'gastarbeider', 'nieuwe Nederlander', 'etnische minderhe', 'afkomst', 'land van herkomst', 'moslim',
                   'NOT vreemdelingenlegioen']
    for word in filter_list:
        if re.search(word, df.loc[0, 'text']):
            print('1')

if __name__=='__main__':
    df = pd.read_csv('../../../data/processed/news-dataset--2021-05-11.csv')
    df = df.head(10)
    filter_art(df)
