
# -*- coding: utf-8 -*-


import pandas as pd 
spotify_file= pd.read_csv('spotify3.csv')
print(spotify_file)

# to lower case

spotify_file['Review']= spotify_file['Review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
spotify_file['Review'].head()

# exract emoji 
# to be taken later
import emot

spotify_list =spotify_file['Review']
l=[]
for i in range(len(spotify_list)):
    x= emot.emoji(spotify_list[i]).get("value","none")
    l.append(x)   
    
spotify_file['emoji']=l

# punctuation removal
#to be not done
spotify_file['Review'] = spotify_file['Review'].str.replace('[^\w\s]','')
spotify_file['Review'].tail()

# stop words removal

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
stop = set(stopwords.words('english'))
spotify_file['Review'] = spotify_file['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
spotify_file['Review'].head()

# number removal

spotify_file['Review'] = spotify_file['Review'].str.replace('\d+', '')
spotify_file['Review'].head()


# Lemmatization

import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

spotify_file['Review']= spotify_file['Review'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word,'n')for word in x.split()]))
spotify_file['Review']= spotify_file['Review'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word,'v')for word in x.split()]))
print(spotify_file['Review'])

# spelling correction
from autocorrect import spell 
spotify_file['Review']= spotify_file['Review'].apply(lambda x: " ".join([spell(i) for i in x.split()]))

#replace words (depends on how word changes)

spotify_file.Review = spotify_file.Review.str.replace('app', 'application')
spotify_file.Review = spotify_file.Review.str.replace('specify', 'spotify')


# Tokenization

#spotify_file['Review'] = spotify_file.apply(lambda row: nltk.word_tokenize(row['Review']), axis=1)

#sentiments
from textblob import TextBlob
spotify_file['sentiment'] = spotify_file['Review'].apply(lambda x: TextBlob(x).sentiment[0] )
spotify_file[['Review','sentiment']].head()

#inverse document frequency (not sure)

#import numpy as np
#tf1 = (file['Review'][1:10]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
#tf1.columns = ['words','tf']

#for i,word in enumerate(tf1['words']):
#tf1.loc[i, 'idf'] = np.log(amazon_reviews.shape[0]/(len(file[file['Review'].str.contains(word)])))


# Bag of words
#from sklearn.feature_extraction.text import CountVectorizer    
#bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
#input= [input]
#train_bow = bow.fit_transform(spotify_file['Review'] [2])
#print(train_bow)
