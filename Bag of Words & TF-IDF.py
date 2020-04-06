import pandas as pd
import numpy as np

with open ("G:/LDA/lotr.txt") as lotr:
    lotr =lotr.read()

# Cleaning the Data
import nltk
import re
from nltk.tokenize import sent_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
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

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(filtered_split).toarray()


###TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

tvec_ta = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
tvec_weights_ta = tvec_ta.fit_transform(filtered_split)

weights_ta = np.asarray(tvec_weights_ta.mean(axis=0)).ravel().tolist()
weights_df_ta = pd.DataFrame({'term': tvec_ta.get_feature_names(), 'weight': weights_ta})

weights_df_ta=weights_df_ta.sort_values(by='weight', ascending=False)
weights_df_ta.head(10)
