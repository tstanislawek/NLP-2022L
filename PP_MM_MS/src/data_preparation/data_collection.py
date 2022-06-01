import pandas as pd
import pickle
import sys
import os
sys.path.append('../')
from utils.utils import scrape_band, clean_record, Tee, clean_record_for_nlp

#url = "https://www.tekstowo.pl/piosenki_artysty,akcent_pl_.html"
#res = scrape_band(url)


# for logging purposes
sys.stdout = Tee("log_data_collection.dat", mode="a", encoding='utf-8')
# read the urls
urls = pd.read_excel("../data/non_disco_polo.xlsx")["link"]
# print(urls)

n_bands = 0
n_hits = 0
n_bands_total = len(urls)
with open("../data/non_disco_songs_data_tokenized.txt", "a", encoding='utf-8-sig') as file:
    # iterate over each band
    for band_url in urls:
        print(f"Band number {(n_bands+1)} out of {n_bands_total} bands")
        res = scrape_band(band_url, verbose=True, sleep_time=0.3)
        #res_cleaned = [clean_record(txt) for txt in res]
        res_cleaned = [clean_record_for_nlp(txt) for txt in res]
        n_bands += 1
        n_hits += len(res_cleaned)
        # write output to the data file
        file.write(u'\n\n'.join(res_cleaned))
        df = pd.DataFrame(data={"song": res_cleaned})
        df.to_csv("../data/non_disco_songs_data_tokenized.csv", sep=',', index=False, mode="a", header=False)#, encoding='utf-8')


print(f"Data was scraped for {n_hits} songs.")
print(f"Data was scraped for {n_bands} bands.")










