# Text Summarization

#NLTK is a leading platform for building Python programs to work with 
#human language data. It provides easy-to-use interfaces to over 50 corpora
#and lexical resources such as WordNet, along with a suite of text processing
#libraries for classification, tokenization, stemming, tagging, parsing,
#and semantic reasoning, wrappers for industrial-strength NLP libraries,
import nltk

#The counter is a sub-class available inside the dictionary class. Using the Python 
#Counter tool, you can count the key-value pairs in an object, also called a hash table object
from collections import Counter

#Tokenizers divide strings into lists of substrings.  For example,
#tokenizers can be used to find the words and punctuation in a string
#We use the method sent_tokenize() – Splitting sentences in the paragraph
#We use the method word_tokenize() to split a sentence into words.
from nltk.tokenize import sent_tokenize, word_tokenize

#A stop word is a commonly used word (such as “the”, “a”, “an”, “in”)
# that a search engine has been programmed to ignore
nltk.download('stopwords')
from nltk.corpus import stopwords

#punctuation is a pre-initialized string used as string constant
from string import punctuation

#Given a list (or set or whatever) of n elements, we want to get the k biggest ones. 
#Heap data structure is mainly used to represent a priority queue
#EX:lst = [9,1,6,4,2,8,3,7,5]
#nlargest(4, lst) # Gives [9,8,7]
from heapq import nlargest

#Sets are used to store multiple items in a single variable, and here we adding stop words
#list of punctuation are assigning to stpowords 
#
STOPWORDS = set(stopwords.words('english') + list(punctuation))

#here we are going to assigning 2 values to varibles
MIN_WORD_PROP, MAX_WORD_PROP = 0.1, 0.9

#####
#The sentence in the text variable is tokenized (divided into words) using the word_tokenize() method. 
#Next, we iterate through all the words in the counter list and checks if the word exists in the 
#stop words collection or not.
def compute_word_frequencies(word_sentences):
    words = [word for sentence in word_sentences for word in sentence if word not in STOPWORDS]
    counter = Counter(words)
    limit = float(max(counter.values()))
    word_frequencies = {word: freq/limit 
                                for word,freq in counter.items()}
    
    # Drop words if too unique or too unique in word_frequencies
    word_frequencies = {word: freq 
                            for word,freq in word_frequencies.items() 
                                if freq > MIN_WORD_PROP 
                                and freq < MAX_WORD_PROP}
    return word_frequencies
####

####
def sentence_score(word_sentence, word_frequencies):
    return sum([ word_frequencies.get(word,0) 
                    for word in word_sentence])
###
    
###
def summarize(text:str, num_sentences=3):
    #"""
    #Summarize the text, by return the most relevant sentences
    # :text the text to summarize
    # :num_sentences the number of sentences to return
   # """
    text = text.lower() # Make the text lowercase

# sent_tokenize() – Splitting sentences in the paragraph Break text into sentences
    sentences = sent_tokenize(text)  
    
    #the method Break sentences into words
    word_sentences = [word_tokenize(sentence) for sentence in sentences]
    
    # Compute the word frequencies #Find frequency of each word in a string
    word_frequencies = compute_word_frequencies(word_sentences)
    
    # Calculate the scores for each of the sentences
    scores = [sentence_score(word_sentence, word_frequencies) for word_sentence in word_sentences]
    sentence_scores = list(zip(sentences, scores))
    
    # Rank the sentences
    top_sentence_scores = nlargest(num_sentences, sentence_scores, key=lambda t: t[1])
    
    # Return the top sentences
    return [t[0] for t in top_sentence_scores]
###
    
#1st open a text file for reading by using the open() function.
#2nd read text from the text file using the file read(), readline(), 
#or readlines() method of the file object
with open('F:/360/Text Summarization/Lordoftherings.txt', 'r') as file:
    lor = file.read()


#print function is prints the specified message to the screen
print(lor)

#We use the method sent_tokenize() – Splitting sentences in the lor and checking lenghth 
#len() function returns the number of characters in the string
len(sent_tokenize(lor))

#Summarizer which helps in summarizing and shortening the text in the user feedback
summarize(lor)

#Now we will find the summary of the num_sentences in lor tex doc.
summarize(lor, num_sentences=1)


