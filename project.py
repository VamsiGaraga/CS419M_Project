#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("voice.csv") #Reading the voice dataset


# In[3]:


df.head()


# In[4]:


df["label"] = (df["label"] == "female").astype(int) #Converting the label from string to integer


# In[5]:


df.head()


# In[6]:


df.describe() #We can see that label has mean 0.5, i.e., 50% Female and 50% Male samples


# In[7]:


X_df = df.drop(['label'], axis = 1)
X_df = (X_df - X_df.mean()) / X_df.std()
X = np.array(X_df)
y = np.array(df['label']) 


# In[8]:


#Splitting into training and test data
phi_train, phi_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)


# In[9]:


#First SVM classifier
from sklearn.svm import SVC
model = SVC()
model.fit(phi_train, y_train)
print(f'Score without aug = {model.score(phi_test, y_test)}')


# In[10]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(phi_train, y_train)
print(f'Score without aug = {model.score(phi_test, y_test)}')


# In[11]:


# Multilayer Perceptron/ Neural network model
# Just used one hidden layer with 100 neurons and ReLU activation
from sklearn.neural_network import MLPClassifier
np.random.seed(42)
model = MLPClassifier(max_iter=1000)
model.fit(phi_train, y_train)
print(f'Score without aug = {model.score(phi_test, y_test)}')


# In[12]:


# KNN Classifer with K=5
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(phi_train, y_train)
print(f'Score without aug = {model.score(phi_test, y_test)}')


# In[13]:


# A Random Forest based classifier
from sklearn.ensemble import GradientBoostingClassifier
np.random.seed(42)
model = GradientBoostingClassifier()
model.fit(phi_train, y_train)
print(f'Score without aug = {model.score(phi_test, y_test)}')


# In[14]:


# Data augmentation by adding a little noise, noise follows normal distribution with standard deviation sigma
# Only added noise to training data, so that the test data is completely new to the model
noisy_phi_train = phi_train + np.random.normal(scale=0.008, size = phi_train.shape)
phi_train_new = np.concatenate([phi_train, noisy_phi_train], axis = 0)
y_train_new = np.concatenate([y_train, y_train])


# In[15]:


from sklearn.svm import SVC
model = SVC()
model.fit(phi_train_new, y_train_new)
print(f'Score with aug = {model.score(phi_test, y_test)}')


# In[16]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(phi_train_new, y_train_new)
print(f'Score with aug = {model.score(phi_test, y_test)}')


# In[17]:


from sklearn.neural_network import MLPClassifier
np.random.seed(42)
model = MLPClassifier(max_iter=1000)
model.fit(phi_train_new, y_train_new)
print(f'Score with aug = {model.score(phi_test, y_test)}')


# In[18]:


from sklearn.ensemble import GradientBoostingClassifier
np.random.seed(42)
model = GradientBoostingClassifier()
model.fit(phi_train_new, y_train_new)
print(f'Score with aug = {model.score(phi_test, y_test)}')


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(phi_train_new, y_train_new)
print(f'Score with aug = {model.score(phi_test, y_test)}')


# In[ ]:




