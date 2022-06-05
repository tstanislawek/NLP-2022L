# Test different methods on text classification task

## Tested models
Reformer (didn't fit in the memory)

BERT

DistilBERT

ALBERT

GPT2

DistilGPT2

BART (didn't fit in the memory)



## Tested dataset
IMDB

Amazon Polarity


## Steps to reproduce
1. install required libraries
```
pip install -r requirements 
```
2. (optional) pick models and dataset
```
edit code at the beggining of the train.py script to change dataset (amazon/imdb)
edit code at the end of the train.py script to pick models to train (any from https://huggingface.co/models capable of sequence classification)
```

4. run
```
python train.py
```
