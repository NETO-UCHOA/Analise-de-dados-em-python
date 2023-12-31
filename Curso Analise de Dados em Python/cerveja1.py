#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.compat import lzip


# Todo processo começa com uma análise dos dados, o conhecimento da base a ser analisada é de fundamental importância, assim como qual é o objetivo (previsão) que queremos chegar.
# 
# Os dados (amostra) foram coletados em São Paulo — Brasil, em uma área universitária, onde acontecem algumas festas com turmas de alunos de 18 a 28 anos (média). O conjunto de dados utilizado para esta atividade possui 7 variáveis, sendo uma variável dependente, com período de um ano.
# 
# Nossa variável dependente(objetivo) é o consumo de cerveja em litros.

# In[2]:


df = pd.read_csv("D:\\lixo\\Consumo_cerveja.csv")


# In[3]:


display(df)
# o comando display(df) pode ser reproduzido também com 
#df.head()
#df.tail()
#df.shape


# In[4]:


df.info()
# o comando df,info() pode ser reproduzido também com
#df.isna().sum()
#df.dtypes


# In[5]:


df.describe()


# Uma informação muito inportante para a análise é saber como as variáveis estão correlacionadas entre si, em especial como a variável dependente se correlaciona com cada uma das variáveis independentes

# In[6]:


df.corr(numeric_only=True)


# Como se espera a quantidade de dias que não são sinal de semana é maior que a quantidade de dias que são final de semana

# In[7]:


plt.figure(figsize=(15,7))
sns.countplot(x='Final de Semana', data=df)
plt.xlabel('Final de Semana')
plt.ylabel('');


# In[8]:


df[['Temperatura Media (C)', 
         'Temperatura Minima (C)',
         'Temperatura Maxima (C)']].plot(figsize=(15,7));
plt.title('Séries de temperaturas máximas, Médias e Mínimas', size=15);


# In[9]:


df['Precipitacao (mm)'].plot(figsize=(15,7))
plt.title('Precipitação em mm',size=15);


# In[10]:


df['Consumo de cerveja (litros)'].plot(figsize=(15,7), color='black');
plt.title('Consumo de cerveja',size=15);


# In[11]:


plt.figure(figsize=(15,7))
sns.heatmap(df.corr(numeric_only=True), annot = True, cmap= "RdYlGn");
plt.title('Correlação de Pearson',size=15);


# In[12]:


plt.figure(figsize=(15,7))
plt.title('Correlação de Spearman',size=15)
sns.heatmap(df.corr('spearman',numeric_only=True), annot = True, cmap= "Miscellaneous");


# In[13]:


df[['Temperatura Media (C)', 'Temperatura Minima (C)',
       'Temperatura Maxima (C)', 'Precipitacao (mm)',
       'Consumo de cerveja (litros)']].plot.box(figsize=(15,7));


# In[14]:


df[['Temperatura Media (C)', 'Temperatura Minima (C)',
       'Temperatura Maxima (C)', 'Precipitacao (mm)',
       'Consumo de cerveja (litros)']].hist(figsize=(15,7), bins=50);


# In[15]:


fig,ax = plt.subplots(2,2, figsize=(15,7))
sns.scatterplot(x='Consumo de cerveja (litros)',y='Temperatura Media (C)',data = df,ax=ax[0][0]);
sns.scatterplot(x='Consumo de cerveja (litros)',y='Temperatura Minima (C)',data = df,ax=ax[0][1]);
sns.scatterplot(x='Consumo de cerveja (litros)',y='Temperatura Maxima (C)',data = df,ax=ax[1][0]);
sns.scatterplot(x='Consumo de cerveja (litros)',y='Precipitacao (mm)',data = df,ax=ax[1][1]);


# ### Regressão linear múltipla em Python
# Existem três formas de estimar o modelo de regressão múltipla em Python:
# 
# Biblioteca Scikit Learn : que é mais voltada para resolução de problemas de machine learning e por esse motivo é limitada, principalmente para gerar modelos inferenciais;
# 
# Biblioteca Pingouin : é mais sofisticada que a Scikit Learn gerando mais estatísticas e resultados do modelo;
# 
# Biblioteca Statsmodels : mais completa para gerar o modelo, permitindo a realização de testes para análise e diagnóstico.
# 
# Para fins de demonstração será realizado, primeiramente, o procedimento com a biblioteca Scikit Learn e depois com a Statsmodels.
# 
# Antes será realizada a separação entre a variável dependente e as variáveis independentes.

# In[16]:


#Variáveis independentes
X = df.drop(['Consumo de cerveja (litros)','Data'],axis=1)
#Variável dependentes
y = df['Consumo de cerveja (litros)']


# ## Criando modelo com a Scikit Learn

# In[17]:


modelo = LinearRegression()
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 42)
modelo.fit(X_treino, y_treino)
LinearRegression() #esse é o objeto criado


# In[18]:


# Coeficiente de Determinação (R²)
print(modelo.score(X_teste, y_teste))

# Intercepto (ou constante) do modelo.(β1)
print(modelo.intercept_)

# Coeficientes (ou parâmetros) do modelo. (β2....)
print(modelo.coef_)


# O que é o 𝑅² ?
# 
# O coeficiente de determinação é uma proporção que ajuda a entender o quanto as variáveis explicativas explicam a variação da média do consumo de cerveja. No sumário do modelo com intercepto, o coeficiente de determinação foi de 72.3%; já no modelo sem intercepto, o valor foi de 99.1%.
# 
# O 𝑅² é um métrica que varia de 0 a 1, se o modelo tiver intercepto; caso contrário usa-se o 𝑅² não centrado (caso do modelo 2).
# 
# O 𝑅² varia entre 0 e 1, então quanto maior o 𝑅² melhor é o modelo de regressão, pois teria uma maior a capacidade de explicação.
# 
# Uma limitação dessa medida é que com a inserção de regressores ao modelo o 𝑅² tende a aumentar.

# # Criando modelo com a Statsmodels

# Nessa etapa são gerado dois modelos novamente: um com intercepto (ou constante) e um sem. A presença ou não do intercepto gera mudanças consideráveis nas estatísticas geradas.

# In[20]:


modelo1 = (sm.OLS(y,sm.add_constant(X)).fit())
modelo1.summary(title='Sumário do modelo com intercepto')


# In[21]:


modelo2 = sm.OLS(y,X).fit()
modelo2.summary(title='Sumário do modelo sem intercepto')


# Ver documento WORD 

# Para os diagnósticos dos modelos serão gerados os resíduos, que são a diferença entre o real e o predito pelo modelo, conforme código abaixo

# In[22]:


modelo1.resid
modelo2.resid


# In[23]:


Predicoes = pd.DataFrame(modelo1.predict(), columns=['Predições 1'])
Predicoes['Predições 2'] = modelo2.predict()
Predicoes['Consumo de cerveja (litros)']=df['Consumo de cerveja (litros)']


# In[24]:


plt.figure()
Predicoes[['Predições 1','Consumo de cerveja (litros)']].plot(figsize=(15,7), color=['b','g']);


# In[26]:


plt.figure()
Predicoes[['Predições 2','Consumo de cerveja (litros)']].plot(figsize=(15,7), color=['r','g']);


# In[33]:


residuos1 = modelo1.resid
fig, ax = plt.subplots(2,2,figsize=(15,6))
residuos1.plot(title="Resíduos do modelo 1", ax=ax[0][0])
sns.histplot(residuos1,ax=ax[0][1])
plot_acf(residuos1,lags=40, ax=ax[1][0])
qqplot(residuos1,line='s', ax=ax[1][1]);


# In[34]:


residuos2 = modelo2.resid
fig, ax = plt.subplots(2,2,figsize=(15,6))
residuos2.plot(title="Resíduos do modelo 2", ax=ax[0][0])
sns.histplot(residuos2,ax=ax[0][1])
plot_acf(residuos2,lags=40, ax=ax[1][0])
qqplot(residuos2,line='s', ax=ax[1][1]);


# Calculando o teste Omnibus para os modelos.

# In[39]:


nome1 = ['Estatística', 'Probabilidade']
teste = sms.omni_normtest(modelo1.resid)
lzip(nome1, teste)


# In[40]:


nome2 = ['Estatística', 'Probabilidade']
teste2 = sms.omni_normtest(modelo2.resid)
lzip(nome2, teste2)


# Multicolinearidade

# In[41]:


print('Número condição do modelo 1 :',np.linalg.cond(modelo1.model.exog))

print('Número condição do modelo 2 :',np.linalg.cond(modelo2.model.exog))


# In[42]:


df1 = df
df1['residuos1'] = modelo1.resid
df1['residuos2'] = modelo2.resid


# In[44]:


fig, ax = plt.subplots(1,2,figsize=(20,7))
sns.regplot(x='Consumo de cerveja (litros)',y='residuos1',data=df1, ax=ax[0])
sns.regplot(x='Consumo de cerveja (litros)',y='residuos2',data=df1, ax=ax[1]);


# In[ ]:




