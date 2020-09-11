#Live Visualisation

import matplotlib.pyplot as plt
from opensky_api import OpenSkyApi
import os
import conda
import os
import pandas as pd
import redis
RC = redis.Redis(host='localhost', port=6379, charset='utf-8', decode_responses=True, db=1)

import requests
import time
import random
random.seed(444)

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
from IPython import display

def coordinates():
    api = OpenSkyApi()
    lon = []
    lat = []
    j = 0
    # bbox = (min latitude, max latitude, min longitude, max longitude)
    states = api.get_states(bbox=(8.27, 33.074, 68.4, 95.63))
    for s in states.states:
        lon.append([])
        lon[j] = s.longitude
        lat.append([])
        lat[j] = s.latitude
        j+=1
    return(lon, lat)
    
print("How many Iterations?")
a = int(input())

list1 = []

for i in range(1, a + 1) :
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 20
    fig_size[1] = 20
    plt.rcParams["figure.figsize"] = fig_size
    lon, lat = coordinates()
    for i in range(0,len(lon)) :
        dict1 = { 
                 "longitude": lon[i],
                 "latitude": lat[i],
                }

        list1.append(dict1)
    print(len(list1))    
    hats = {f"hat:{random.getrandbits(32)}": i for i in (list1)}
    with RC.pipeline() as pipe:
        for h_id, hat in hats.items():
            pipe.hmset(h_id, hat)
            pipe.execute()
    print("Status: ",RC.bgsave()) 

    df = pd.DataFrame()
    for key in RC.keys():
        value = RC.hgetall(key)
        df =df.append(value,ignore_index=True)
    


    df.longitude = df.longitude.astype(float)
    df.latitude = df.latitude.astype(float)

    long = df['longitude'].tolist()

    lati = df['latitude'].tolist()

    m = Basemap(projection = 'mill', llcrnrlat = 8.1957,   urcrnrlat = 23.079, llcrnrlon = 68.933, urcrnrlon = 88.586, resolution = 'h')
    m.drawcoastlines()
    m.drawmapboundary(fill_color = '#FFFFFF')
    x, y = m(long, lati)
    plt.scatter(x, y, s = 5)
    display.clear_output(wait=True)
    display.display(plt.gcf())
      
Sentiment Analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymongo
# Comment Data
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client.airline
cursor = db.review.find({"airline_name": "emirates"})
df = pd.DataFrame(list(cursor))

# Cleaning the texts
text = []
for i in range(0, len(df)):
    comment = str(df['content'][i])
    text.append(comment)

from textblob import TextBlob
import matplotlib.pyplot as plt

def percentage(part,whole):
    return 100*float(part)/float(whole)




positive = 0
negative = 0
neutral = 0
polarity = 0


for word in text:
    analyzer = TextBlob(word)
    polarity += analyzer.sentiment.polarity
    if analyzer.sentiment.polarity > 0:
        positive += 1
    elif analyzer.sentiment.polarity < 0:
        negative += 1
    elif analyzer.sentiment.polarity == 0:
        neutral += 1
d = (positive + negative + neutral) 
positive = percentage(positive,(d))
negative = percentage(negative,(d))
neutral = percentage(neutral,(d))

positive = format(positive,'.2f')
negative = format(negative,'.2f')
neutral = format(neutral,'.2f')

print(polarity)

if (polarity > 0):
    print("Positive")
elif (polarity < 0):
    print("Negative")
elif (polarity == 0):
    print("Neutral")

labels = ['Positive ['+str(positive)+'%]', 'Negative ['+str(negative)+'%]', 
'Neutral ['+str(neutral)+'%]']
sizes = [positive, negative, neutral]
colors = ['blue','red','yellow']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.legend(patches,labels,loc="best")
plt.title("Polarity Pie Chart")
plt.axis('equal')
plt.tight_layout()
plt.show()

#Topic Modelling

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
data = df.content.values.tolist()

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

#Wordcloud

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
     
  
comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df.content: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()
