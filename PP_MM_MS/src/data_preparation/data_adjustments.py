import pandas as pd
import re


df = pd.read_csv("../data/non_disco_songs_data_tokenized.csv", names=['song'])

def add_REND_token(song):
    matched = re.finditer(pattern=r'<RBEG>', string=song)
    chorus_beg_ids = [match.start() for match in matched]

    matched = re.finditer(pattern="<EOST>", string=song)
    stanza_end_ids = [match.start() for match in matched]

    res = []
    for i in chorus_beg_ids:
        for j in stanza_end_ids:
            if j>i:
                res.append(j-1)
                break

    for idx in reversed(res):
        song = song[:idx] + "<REND>" + song[idx:]

    return song

df.song = df.song.apply(lambda x: add_REND_token(x))
df.to_csv("../data/non_disco_songs_data_tokenized2.csv", index=False)

