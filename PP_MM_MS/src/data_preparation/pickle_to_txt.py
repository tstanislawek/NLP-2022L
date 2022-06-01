import pandas as pd
import re


def clean(txt):
    newlines = txt.replace('\\n', '\n')
    return bytes(re.sub('\n+', '\n', re.sub(r"(Ref.+|[0-9]+\.|\\r|\r|(^ ))", '', newlines)), 'utf-8').decode('utf-8', 'ignore')


df = pd.read_pickle('../data/songs_akcent')
cleansed = [clean(txt) for txt in df]
with open('../data/songs_akcent.txt', 'w', encoding='utf-8') as f:
    f.write(u'\n\n'.join(cleansed))
print(cleansed[0])
