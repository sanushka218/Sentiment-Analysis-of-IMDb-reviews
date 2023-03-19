import nltk
# copora- body of a text// lexicon- word:meaning
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import wordnet
from nltk.corpus import movie_reviews
import string
"""
eg = "bro the movie was awesome! i loved it. i recommend you watching it, i am not kidding and the car riding scene was epic"
# print(sent_tokenize(eg))
words=word_tokenize(eg)
print(words)
print("/**********************************/")
# for i in word_tokenize(eg):
#    print(i)
# A corpus is a collection of authentic text or audio organized into datasets. Authentic here means text written or audio
# spoken by a native of the language or dialect. A corpus can be made up of everything from newspapers, novels, recipes,
# radio broadcasts to television shows, movies, and tweets.
stop_words= set(stopwords.words("english"))
# print(stop_words)
print("/**********************************/")
preprocessed =[w for w in words if not w in stop_words]
# for i in words:
#     if i not in stop_words:
#         preprocessed.append(i)
# print(preprocessed)
print("/**********************************/")
# STEMMING
ps= PorterStemmer()
# ex= ["ride", "rides", "riding", "rode", "ridely"]
# for w in ex:
#     print(ps.stem(w))
# for w in preprocessed:
#   print(ps.stem(w))
# now stem isn't used instead WordNet is used which is a lexical database and Synset
# is a special kind of interface that is present in Nltk to look up words in WordNet

# PART OF SPEECH TAGGING
# A part-of-speech tagger, or POS-tagger, processes a sequence of words, and attaches a part of speech tag to
# each word. To do this first we have to use tokenization concept (Tokenization is the process by dividing the quantity
# of text into smaller parts called tokens.)
sample_text= "Hello, Im having a great time here, hope to see you soon!thanks bye"
train_text= "Hello, Im having a great time here, hope to see you soon!thanks bye"
custom_sent_tokenizer= PunktSentenceTokenizer(sample_text)
tokenized= custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words= nltk.word_tokenize(i)
            tagged= nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))

process_content()

# CHUNKING
# Shallow parsing- to analyze a sentence to identify the constituents(noun groups,
# verbs, verb groups). No internal structure, nor their main role in the text

# Wordnet
synonyms = []
antonyms = []
for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(set(synonyms))
print(set(antonyms))

w1= wordnet.synset("dog.n.01")
w2= wordnet.synset("table.n.01")
print(w1.wup_similarity(w2))
"""
documents= [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
print(documents[1])
all_words=[]
for w in movie_reviews.words():
    all_words.append(w.lower())
all_words= nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(all_words["hello"])
