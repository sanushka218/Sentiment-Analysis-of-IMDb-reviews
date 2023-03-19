"""
NLP- extracting emotion and behavioural features from text.
   - Also work in finding the dominant emotion
   - Plot these emotion on a graph to get a visual idea about the text
   - NLTK is a leading platform for building Python programs to work with human language data.
   - text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning
"""
# CLEANING THE TEXT:-
# 1) Create a text file and take test from it
# 2) Convert the letters into lowercase ('Apple is not equal to 'apple')
# 3) Remove punctuation like ,!?

import string
from collections import Counter
import matplotlib.pyplot as plt

text = open('testcase.txt', encoding= 'utf-8').read()
# encoding - UTF text style on internet articles
lower_case = text.lower()
#print(lower_case)
# print(string.punctuation) # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ remove these
clean_text = lower_case.translate(str.maketrans('','',string.punctuation))
# str1: specifies the list of characters that need to be replaced
# str2: specifies the list of characters with which the characters need to be replaced
# str3: specifies the list of characters that needs to be deleted
# maketrans create a transition table that helps the translate function to delete these characters
#print(clean_text)

# TOKENIZATION & STOP WORDS
tokenized_words = clean_text.split()
print(tokenized_words)

stop_words= ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
final_words = []
for words in tokenized_words:
    if words not in stop_words:
        final_words.append(words)
print(final_words)
emotion_list = []
with open('emotion.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word,emotion= clear_line.split(':')
        # print("word :"+ word + " "+ "emotion :"+ emotion)

        if word in final_words:
            emotion_list.append(emotion)
print(emotion_list)
w = Counter(emotion_list)
print(w)
fig, axl = plt.subplots()
axl.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.show()






