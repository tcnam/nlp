from cProfile import label
import nltk
from nltk.tokenize import word_tokenize
import os
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
from pprint import pprint as pretty_print

with open('intents.json') as file:
    data=json.load(file)

# pretty_print(data)

words=[]
labels=[]
docs_x=[]
docs_y=[]


for intent in data['intents']:
    for pattern in intent['patterns']:
        pattern=pattern.lower()
        word=word_tokenize(pattern)
        words.extend(word)
        docs_x.append(word)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

