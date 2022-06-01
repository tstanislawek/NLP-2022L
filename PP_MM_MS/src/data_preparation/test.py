import pickle
import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
import seaborn as sn

df = pd.read_csv("../data/songs_data_tokenized.csv")
print(df.columns)


n_chorus_avg = df.song.apply(lambda x: x.count("<RBEG>")).sum() / len(df)
n_stanza_avg = df.song.apply(lambda x: x.count("<EOST>")).sum() / len(df)
len_stanza_avg = df.song.apply(lambda x: len(x)).sum() / len(df) / 5
print(n_chorus_avg)
print(n_stanza_avg)
print(len_stanza_avg)
sn.displot(df.song.apply(lambda x: len(x)))
plt.show()

r = r'R(efren|EFREN|ef|EF|)(\.|:|\.:)|\[[0-9]x:\]'
s1 = "Ref.:"
s2 = "REF:"
s3 = "REFREN.:"
s4 = "Ref."
s5 = "[4x:]"

print(re.match(r, s1))
print(re.match(r, s2))
print(re.match(r, s3))
print(re.match(r, s4))
print(re.match(r, s5))

"""l = [1,23,4,5,22,6,7,8,89]

def f(x):
    if x==4:
        pass
    else:
        return x

print(l)
print([f(i) for i in l])

path = "../data/songs_data_tokenized.csv"
df = pd.read_csv(path)
print(df)

raw_text = ''
start_token = "<|startoftext|>"
end_token = "<|endoftext|>"
with open(path, 'r', encoding='utf8', errors='ignore') as fp:
    fp.readline()  # skip header
    reader = csv.reader(fp)
    for row in reader:
        raw_text += start_token + row[0] + end_token + "\n"
        """

