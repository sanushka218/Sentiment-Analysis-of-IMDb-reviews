"""
NLP- extracting emotion and behavioural features from text.
   - Also work in finding the dominant emotion
   - Plot these emotion on a graph to get a visual idea about the text
   - NLTK is a leading platform for building Python programs to work with human language data.
   - text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning
"""
# -------------------------------------------LEXICON MODEL--------------------------------------------------------------
# CLEANING THE TEXT:-
# 1) Create a text file and take test from it
# 2) Convert the letters into lowercase ('Apple is not equal to 'apple')
# 3) Remove punctuation like ,!?

import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

test= open('testcase.txt').read()
print(test)
# encoding - UTF text style on internet articles
lower_case = test.lower()
print("content in lower case done")
print(lower_case)
# print(string.punctuation) # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ remove these
clean_text = lower_case.translate(str.maketrans('','',string.punctuation))
# str1: specifies the list of characters that need to be replaced
# str2: specifies the list of characters with which the characters need to be replaced
# str3: specifies the list of characters that needs to be deleted
# maketrans create a transition table that helps the translate function to delete these characters
print("text cleaned")
print(clean_text)

# TOKENIZATION & STOP WORDS
tokenized_words= word_tokenize(clean_text, "english")
print(tokenized_words)

# STOP WORDS:-
final_words = []
for words in tokenized_words:
    if words not in stopwords.words('english'):
        final_words.append(words)
print("                                                                              ")
print("final words")
print(final_words)

# ALGORITHM FOR EMOTION AND TEXT ANALYSIS
# NLP EMOTION ALGORITHM
# 1) Check if the word in final words list is also present in emotion.txt
# - open the emotion file
# - loop through each line and clear it
# - extract the word and emotion using split
# 2) if word is present-> add it to emotion list
# 3) finally count each emotion in the emotion list

emotion_list = []

with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        # print("word :"+ word + " "+ "emotion :"+ emotion)

        if word in final_words:
            emotion_list.append(emotion)
print("                                                                                ")
print("emotion list")
print(emotion_list)
w = Counter(emotion_list)
print("                                                                                ")
print(w)

# Creating a function that measures the intensity of the emotion
def sentiment_analyse(sentiment_text):
    score= SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    print(score)
    neg=score['neg']
    pos= score['pos']
    if neg>pos:
        print("NEGATIVE REVIEW =0")
    elif pos>neg:
        print("POSITIVE REVIEW =1")

sentiment_analyse(clean_text)

fig, axl = plt.subplots()
axl.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.show()






