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

wordnet=WordNetLemmatizer()

x =  nltk.sent_tokenize(lotr)
for i in range(len(x)):
    words = nltk.word_tokenize(x[i])
    words = [wordnet.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    x[i] = ' '.join(words)   
    
# Stemming
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()

x =  nltk.sent_tokenize(lotr)

for i in range(len(x)):
    words = nltk.word_tokenize(x[i])
    words = [ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
    x[i] = ' '.join(words)      