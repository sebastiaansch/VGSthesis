#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../")
import nltk
import pandas as pd
import numpy as np
from helper import tokenize, dicttodf
import json


# In[2]:


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


def imagenet_flickr8k_intersection():
    path = "../data/Flickr8k_text/Flickr8k.token.txt"
    wordlist, worddictionary = tokenize("../data/Flickr8k_text/Flickr8k.token.txt")
    worddf = dicttodf(worddictionary)
    path2 = "/Users/sebastiaanscholten/Documents/speech2image-master/vgsexperiments/experiments/data/imagenet_class_index.json"

    with open(path2,"r") as json_file:
        data = json.load(json_file)
        
    wordlist1 = list(worddf["words"])
    wordlist2 = []
    for words in data.values():
        wordlist2.append(words[1])

    return intersection(wordlist1,wordlist2)


# In[3]:


x = imagenet_flickr8k_intersection()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




