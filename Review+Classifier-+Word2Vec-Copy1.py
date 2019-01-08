
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

labeledData = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter='\t', quoting= 3)
unlabeledData = pd.read_csv("unlabeledTrainData.tsv", header = 0, delimiter='\t', quoting = 3)
testData =  pd.read_csv("testData.tsv", header = 0, delimiter='\t', quoting = 3)

labeledData.shape, unlabeledData.shape, testData.shape

# transform raw review
from bs4 import BeautifulSoup as bs
import re
from nltk.corpus import stopwords

# unlike bag of words, Word2Vec needs a list of words so we return the same
# instead of returning a string as done previously.
def transformSentence(rawReview, remove_stopwords = False):
    #remove punctuation marks
    noHTML = bs(rawReview, "lxml").get_text()
    
    #remove punctuation marks
    letters_only = re.sub("[^a-zA-Z0-9]", " ", noHTML)
    
    #convert to lower case and split
    words = letters_only.lower().split()
    
    #optional removing stopwords
    if remove_stopwords:
        sw = set(stopwords.words("english"))
        words = [w for w in words if w not in sw]
    
    #return list of words in the review
    return words

clean_review = transformSentence(unlabeledData.review[0])
# clean_review

unlabeledData.review[0]

import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_sentence=  tokenizer.tokenize(unlabeledData.review[0].strip())
raw_sentence

def transformReview(rawReview, tokenizer, remove_stopwords = False):
    
    #convert paragraph into sentences
    raw_sentence = tokenizer.tokenize(rawReview.strip())
    
    sentences = []
    
    #for each sentence, convert it into list of words
    for sentence in raw_sentence:
        if(len(sentence) > 0):
            sentences.append(transformSentence(sentence, remove_stopwords))
    
    #return list of sentences each broken into words i.e. a list of lists
    return sentences

s = transformReview(unlabeledData.review[0], tokenizer)
print(len(s))
print(len(s[0]))
s

# form a 3D matrix of all sentences from all reviews
sentences = []

for review in unlabeledData.review:
    sentences += transformReview(review, tokenizer)

for review in labeledData.review:
    sentences += transformReview(review, tokenizer)

len(sentences[0])

sentences[0]

## Model Training

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

nFeatures = 300
minWordCount = 40
nWorkers = 4
contextWindow = 10
downsampling = 1e-3

from gensim.models import word2vec
model = word2vec.Word2Vec(sentences, workers = nWorkers,                          size = nFeatures, min_count = minWordCount,                          window = contextWindow, sample = downsampling)

model.init_sims(replace = True)

model_name = "300features_40minwords_10context"
model.save(model_name)

# Identifying the words most dissimilar
print(model.wv.doesnt_match("man woman shot kitchen".split()))
print(model.wv.doesnt_match("man woman child kitten".split()))

# identifying differences in meaning
print(model.wv.doesnt_match("france england europe berlin".split()))

# imperfections
print(model.wv.doesnt_match("france england america berlin ".split()))

# finding most similar 
model.wv.most_similar("man")

# one pretty surprising thing occured when you find words similar to woman- results-what a wow.
model.wv.most_similar("woman")

model.wv.most_similar("queen")

model.wv.most_similar("awful")

