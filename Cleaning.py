import pandas as pd
import numpy as np

with open ("G:/LDA/lotr.txt") as lotr:
    lotr =lotr.read()

# Cleaning the Data
import nltk
import re
from nltk.tokenize import sent_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
tokenize_sent = sent_tokenize(lotr)

filtered_split=[]
for i in range(len(tokenize_sent)):
    review = re.sub("[^A-Za-z" "'']+"," ",tokenize_sent[i])
    review = re.sub("[0-9" "' ',.]+"," ",tokenize_sent[i])
    review =review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    filtered_split.append(review)
