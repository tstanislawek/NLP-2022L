from transformers import AutoTokenizer, BertForSequenceClassification, TextClassificationPipeline
from nltk import ngrams
import pandas as pd
import re 

model = BertForSequenceClassification.from_pretrained("disco_vs_not_disco")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

def score_discopolo(text): 
    """Returns the score how much a song resembles a discopolo song, using a BERT classifier."""
    return [x['score'] for x in pipeline(text)[0] if x['label'] == 'LABEL_1'][0]

def score_repetability(text, N=4):
    """Returns the minus average of how many times an N-gram was repeated"""
    ngrams_result = ngrams(re.sub('\ +', ' ',text).split(" "), N)
    return -pd.Series(ngrams_result).value_counts().mean()