# Sarcasm detection

## todo
1. analyze stopwords, lowercase of spacy tokenizer
2. use attention
3. try cnn


## Idea
1. cnn (still treat it as sentiment analysis)
2. regard it as NLI problem (sent a, sent b => predict relationship)
    * bert? think more of switching segment embeddings by language embeddings? <= talk from ves (MLM, TLM)


## baseline.py
__test accuracy__: 0.66395

## train.py
__Description__: regard it as a classification problem and use rnn 

nvidia-docker run -ti --rm --name sentiment -u 243337:100 -v /nas/longleaf/home/xiaopeng:/nas/longleaf/home/xiaopeng vincentlu073/inls690:spacy


