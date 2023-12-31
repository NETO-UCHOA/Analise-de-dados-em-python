#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("D:\\lixo\\aluguel.csv",sep=';')


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df


# In[6]:


df.dtypes


# In[7]:


df.info()


# In[10]:


display(df.describe())


# In[9]:


df['Bairro'].describe()


# In[19]:


df.dropna(inplace=True)


# In[20]:


df.isnull().sum()


# In[21]:


df.corr()


# In[11]:


df = df[['Area','Valor']]


# In[23]:


df.corr()


# In[24]:


X = np.array(df['Area'])


# In[25]:


X = X.reshape(-1, 1)


# In[26]:


y = df['Valor']


# In[27]:


plt.scatter(X, y, color = "blue", label = "Dados Reais Históricos")
plt.xlabel("Area")
plt.ylabel("Valor")
plt.legend()
plt.show()


# In[28]:


X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[29]:


modelo = LinearRegression()


# In[31]:


modelo.fit(X_treino, y_treino)


# In[32]:


plt.scatter(X, y, color = "blue", label = "Dados Reais Históricos")
plt.plot(X, modelo.predict(X), color = "red", label = "Reta de Regressão com as Previsões do Modelo")
plt.xlabel("Area")
plt.ylabel("Valor")
plt.legend()
plt.show()


# In[33]:


score = modelo.score(X_teste, y_teste)
print(f"Coeficiente R^2: {score:.2f}")


# In[34]:


modelo.intercept_


# In[35]:


modelo.coef_


# In[ ]:




