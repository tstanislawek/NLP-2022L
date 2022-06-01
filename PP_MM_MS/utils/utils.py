import requests
import re
from bs4 import BeautifulSoup
import sys
import time

class Tee(object):
    def __init__(self, name, mode, encoding="utf-8"):
        self.file = open(name, mode, encoding=encoding)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        try:
            self.stdout.write(data)
        except UnicodeEncodeError as err:
            self.stdout.write(f"Writing log didn't succeed due to {err}.")

    def flush(self):
        self.file.flush()


def robust_request(url, sleep_time=0.5):
    """
    A wrapper to make robust https requests.
    """
    status_code = 500  # Want to get a status-code of 200
    while status_code != 200:
        time.sleep(sleep_time)  # Don't ping the server too often
        try:
            r = requests.get(url)
            status_code = r.status_code
            if status_code != 200:
                print(f"Server Error! Response Code {r.status_code}. Retrying...")
        except:
            print("An exception has occurred, probably a momentory loss of connection. Waiting one seconds...")
            time.sleep(2)
    return r


def scrape_text(url, sleep_time):
    r = robust_request(url, sleep_time)
    soup = BeautifulSoup(r.text, "html.parser")
    tab = soup.findAll("div", attrs={"class": "inner-text"})
    # it may happen that a song won't have any text
    # then we return empty string
    if len(tab) == 0:
        return ""
    else:
        return tab[0].text


def clean_record_for_nlp(txt):
    """
    Idea: preprocess song text for NLP models. Each song is treated
    as one input.
    At the end of each line '<EOL>' token is added.
    At the beginning of chorus a '<RBEG>' token is added.
    At the end of chorus a '<REND>' token is added.
    At the end of each stanza add '<EOST>'.

    :param txt:
    :return:
    """
    txt = txt.replace('\\n', '\n')
    idx_add = 0  # after each token insertion indices of chorus beginning are changing
    # and idx_add purpose is to store the value of this shift

    matched = re.finditer(pattern=r'R(efren|EFREN|ef|EF|)(\.|:|\.:)|\[[0-9]x:\]', string=txt)
    chorus_beg_ids = [match.start() for match in matched]

    # add <RBEG> token
    for chor_start in chorus_beg_ids:
        txt = txt[:chor_start+idx_add] + "<RBEG>" + txt[chor_start+idx_add:]
        idx_add += len("<RBEG>")

    matched = re.finditer(pattern=r"\n\n", string=txt)
    stanza_end_ids = [match.end() for match in matched] + [len(txt)]
    # add <EOST> token
    idx_add = 0
    for stanza_end in stanza_end_ids:
        txt = txt[:(stanza_end+idx_add-1)] + "<EOST>" + txt[(stanza_end+idx_add-1):]
        idx_add += len("<EOST>")

    #chorus_end_id = max([i if i > chorus_beg_id else -1 for i in stanza_end_ids])
    #chorus_end_id = set([i for i in chorus_beg_id])\
    #    .intersection(set([i for i in stanza_end_ids])\
    #                  .difference(set(chorus_beg_id)))
    #stanza_end_ids = list(set(stanza_end_ids).difference(chorus_end_id))
    #chorus_end_id = []
    #for pair in stanza_end_ids
    #print(txt)
    #print("chorus_end_id", chorus_end_id)
    #print("chorus_beg_id", chorus_beg_ids)
    #print("stanza_end_ids", stanza_end_ids)

    # add <REND> token
    #txt = txt[:chorus_end_id] + "<REND>" + txt[chorus_end_id:]

    # add <EOL> tokens
    #txt_splited = txt.split()
    #txt = "<|startoftext|>" + txt + "<|endoftext|>"
    return bytes(txt.encode("utf8", errors="ignore")).decode("utf-8-sig", "ignore")


def clean_record(txt):
    newlines = txt.replace('\\n', '\n')
    return bytes(re.sub('\n+', '\n', re.sub(r"(Ref.+|[0-9]+\.|\\r|\r|(^ ))", '', newlines)), 'utf-8').decode('utf-8', 'ignore')



def scrape_band(url, verbose=True, sleep_time=0.5):
    r = robust_request(url, sleep_time)
    soup = BeautifulSoup(r.text, "html.parser")

    pages = [i.contents[0] for i in soup.findAll("a", attrs={"class": "page-link"})]
    page_nums = []
    for i in pages:
        try:
            page_nums.append(int(i))
        except ValueError:
            continue

    # if there is less than 31 hits there are no pages
    try:
        max_page_num = max(page_nums)
        # try to get active page number
        active = soup.find("li", attrs={"class": "page-item active"})
        active_page_num = int(active.find("a").contents[0])
    except ValueError:
        max_page_num = 1
        active_page_num = 1

    res = []

    # collect data for each page with music hits
    while active_page_num <= max_page_num:

        # collect all metadata
        tables = soup.findAll("div", attrs={"class": "ranking-lista"})
        tables = tables[0].findAll("a", attrs={"class": "title"})
        metadata = [(elem.get("title"), elem.get("href")) for elem in tables]

        # collect song texts
        for i, (title, link_end) in enumerate(metadata):
            # time.sleep(sleep_time)
            if verbose:
                t = time.localtime()
                current_time = time.strftime("%H:%M:%S", t)
                print(f"Song {i + 1}/30, page {active_page_num}/{max_page_num}, title = {title} (time: {current_time}).")
            full_link = "https://www.tekstowo.pl" + link_end
            scraped_text = scrape_text(full_link, sleep_time)
            if scraped_text != "":
                res.append(scraped_text)

        # update page counter
        active_page_num += 1
        # if the page wasn't the last update url,soup variables
        if active_page_num <= max_page_num:
            new_url = url+f",alfabetycznie,strona,{active_page_num}.html"
            r = robust_request(new_url, sleep_time)
            soup = BeautifulSoup(r.text, "html.parser")

    return res

