#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imporing data processing libraries
import pandas as pd
import numpy as np


# In[2]:


#loading vectorizer and similarity metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


#read data and drop examples that has no answer
df=pd.read_csv('aws_faq.csv')
df.dropna(inplace=True)


# In[ ]:


#Train the vectorizer
vectorizer=TfidfVectorizer()
vectorizer.fit(np.concatenate((df.Question,df.Answer)))


# In[ ]:


#If you pass the data fram to trian the vectorizer also same as the passing the ndarray
#vectorizer1=TfidfVectorizer()
#vectorizer1.fit(df)


# In[ ]:


#vectorize the questions
question_vector=vectorizer.transform(df.Question)


# In[ ]:


#Chat with the user
print('you can start chating with me now')
while True:
    input_question=input()
    input_question_vector=vectorizer.transform([input_question])
    similarity=cosine_similarity(input_question_vector,question_vector)
    print(similarity)
    closest=np.argmax(similarity,axis=1)
    print("BOT: "+df.Answer.iloc[closest].values[0])


# In[ ]:




