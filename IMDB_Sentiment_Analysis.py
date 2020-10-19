# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 07:43:25 2020

@author: Ishan Nilotpal
"""

import pandas as pd
import nltk
import re
import pickle

df = pd.read_csv('Test.csv')
message = df['text']

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
ws = WordNetLemmatizer()
corpus = []

for i in range(len(message)):
    review = re.sub('[^a-zA-Z]',' ',message[i])
    review = review.lower()
    review = review.split()
    review = [ws.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    

from sklearn.feature_extraction.text import CountVectorizer
cv =  CountVectorizer()
x = cv.fit_transform(corpus).toarray()
pickle.dump(cv,open('transform.pkl','wb'))

y = df['label']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
cla = LogisticRegression()
cla.fit(x_train,y_train)

y_pred = cla.predict(x_test)

pickle.dump(cla,open('model.pkl','wb'))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
  
