#!/usr/bin/env python
# coding: utf-8

# # Oasis Task 1 - IRIS FLOWER CLASSIFICATION

# In[29]:


#importing the needed libraies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")


# In[30]:


#importing the dataset 
df = pd.read_csv('C:/Users/preet/OneDrive/Desktop/Iris.csv')
df.head()


# In[31]:


#Checking for the null values
df.isnull().sum()


# In[32]:


#Column names
df.columns


# In[33]:


# Statistical Analysis of the dataset
df.describe()


# In[34]:


#Droping unwanted columns
df = df.drop(columns = "Id")


# In[35]:


df


# In[36]:


#Vitualizations
df['Species'].value_counts()


# In[37]:


sns.countplot(df['Species'])


# In[38]:


corr =df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="Blues", annot=True)


# In[39]:


x = df.iloc[:,:4]
y = df.iloc[:,4]


# In[40]:


x


# In[41]:


y


# In[42]:


#Spliting the data into train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[43]:


x_train.shape


# In[44]:


y_train.shape


# In[45]:


x_test.shape


# In[46]:


y_test.shape


# In[47]:


#Creating the classification model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[48]:


model.fit(x_train,y_train)


# In[49]:


y_pred = model.predict(x_test)
y_pred


# In[50]:


#Accuracy and Confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix


# In[51]:


confusion_matrix(y_test,y_pred)


# In[52]:



accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}".format(accuracy))


# In[ ]:




