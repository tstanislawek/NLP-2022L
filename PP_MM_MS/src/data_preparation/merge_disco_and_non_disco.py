import os
import pandas as pd


os.chdir('../data/')
df1 = pd.read_csv('songs_data_tokenized2.csv')
df1['label'] = 1
df2 = pd.read_csv('non_disco_songs_data_tokenized2.csv')
df2['label'] = 0
n = min(len(df1), len(df2))
df = pd.concat([df1.sample(n), df2.sample(n)], axis=0)
df.sample(frac=1).to_csv('disco_vs_not_tokenized.csv', index=False)
