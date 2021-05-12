import tensorflow 
import pandas as pd
import numpy as np

with open ("G:/LDA/lotr.txt") as docs:
    docs = docs.read()

from tensorflow.keras.preprocessing.text import one_hot #One Hot Representation

vocab_size = 10000 ##vocab_size is size of the dictionary (Vocabulary Size)

onehot_repr = [one_hot(words,vocab_size) for words in docs]
print(onehot_repr)

#Word Embedding Representation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential

maxsent_length = 20
embedded_docs = pad_sequences(onehot_repr,padding = 'pre', maxlen=maxsent_length )
print(embedded_docs)

dim = 15

model = Sequential()
model.add(Embedding(vocab_size,15,input_length=maxsent_length ))
model.compile(optimizer='adam',loss='mse')

model.summary()

print(model.predict(embedded_docs))
embedded_docs[0]

print(model.predict(embedded_docs)[0])