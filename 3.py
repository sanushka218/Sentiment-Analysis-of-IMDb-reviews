"""
"""
# ------------------------------------------CLASSIFICATION MODEL USING LOGISTIC REGRESSION----------------------------------------------------
import numpy as np
import pandas as pd
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import re

dataset= pd.read_csv('IMDB Dataset.csv')
print("DATASET SHAPE:- ", dataset.shape)
print("DATASET:- ", dataset.head())
print("Missing values in dataset:- ",dataset.isnull().sum())
# SEPARATING THE DATA & LABEL
x= dataset.drop(columns='label', axis=1)
y= dataset['label']
print("/*************************************************************************************************************/")


# print(string.punctuation) # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ remove these
# clean_text = lower_case.translate(str.maketrans('','',string.punctuation))
# print(clean_text)
# STEMMING: Reduce word to its root word

port_stem= PorterStemmer()
def stemming(review):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', review)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=''.join(stemmed_content)
    return stemmed_content
#dataset['review']= dataset['review'].apply(stemming)
x= dataset['review'].values

# what is RE.SUB() funvtion:-
# The re.sub() function stands for a substring and returns a string with replaced values.
# Multiple elements can be replaced using a list when we use this function.

print("CONTENT STEMMING DONE")

print("/*************************************************************************************************************/")
# LEMMATIZATION is a text normalization technique used in Natural Language Processing (NLP), that switches any kind of a
# word to its base root mode. Lemmatization is responsible for grouping different inflected forms of words into the root
# form,having the same meaning.
# also transforms plurals to singulars
lemmatizer= WordNetLemmatizer()
# print(lemmatizer.lemmatize("better", pos="a"))
print("LEMMATIZATION OF CONTENT DONE")
# Converting the textual data to numerical data
# tfid helps in reducing those text which are repeated and dont add much significant value
vectorizer= TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)

# TOKENIZATION & STOP WORDS
# tokenized_words= word_tokenize(clean_text, "english")
# print(tokenized_words)
# print("/**************************************************************************************/")
# # STOP WORDS:-
# final_words = []
# for words in tokenized_words:
#     if words not in stopwords.words('english'):
#         final_words.append(words)
# print("FINAL WORDS:-")
# print(final_words)
print("/************************************************************************************************************/")
# Splitting the dataset into training & testing dataset
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)

# MODEL:-
model =LogisticRegression()
model.fit(x_train, y_train)
print("X_TRAIN",x_train.shape)
print("X_TEST",x_test.shape)
print("Y_TRAIN",y_train.shape)
print("Y_TEST",y_test.shape)
# EVALUATION:-
# accuracy score on TRAIN data
x_train_prediction= model.predict(x_train)
training_data_accuracy= accuracy_score(x_train_prediction, y_train)
print("ACCURACY SCORE OF TRAINING DATA:- ",training_data_accuracy)
# accuracy score on TEST data
x_test_prediction= model.predict(x_test)
testing_data_accuracy= accuracy_score(x_test_prediction, y_test)
print("ACCURACY SCORE OF TESTING DATA:- ",testing_data_accuracy)
print("/*************************************************************************************************************/")
# MAKING A PREDICTIVE SYSTEM:-
x_new= x_test[0]
prediction = model.predict(x_new)
print(prediction)

if(prediction[0]==0):
    print("THE REVIEW IS NEGATIVE")
else:
    print("THE REVIEW IS POSITIVE")
print("/*************************************************************************************************************/")

# ALGORITHM FOR EMOTION AND TEXT ANALYSIS
# NLP EMOTION ALGORITHM
# 1) Check if the word in final words list is also present in emotion.txt
# - open the emotion file
# - loop through each line and clear it
# - extract the word and emotion using split
# 2) if word is present-> add it to emotion list
# 3) finally count each emotion in the emotion list

# emotion_list = []
#
# with open('emotions.txt', 'r') as file:
#     for line in file:
#         clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
#         word, emotion = clear_line.split(':')
#         #print("word :"+ word + " "+ "emotion :"+ emotion)
#
#         if word in final_words:
#             emotion_list.append(emotion)
# print(emotion_list)
# w = Counter(emotion_list)
# print(w)
#
# # Creating a function that measures the intensity of the emotion
# def sentiment_analyse(sentiment_text):
#     score= SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
#     print(score)
#     neg=score['neg']
#     pos= score['pos']
#     if neg>pos:
#         print("0") # NEGATIVE REVIEW
#     elif pos>neg:
#         print("1") # POSITIVE REVIEW
#
#
# sentiment_analyse(clean_text)
#
# fig, axl = plt.subplots()
# axl.bar(w.keys(), w.values())
# fig.autofmt_xdate()
# plt.show()






