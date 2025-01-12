# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 14:24:27 2025

@author: Gabriel Prisco
Problema: Detectar Fake News """

#%% Importar Bibliotecas:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk as nlp
import sklearn as sk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
from nltk.stem import WordNetLemmatizer  
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
#%% Reading datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")
# Rows and columns of fake news dataset
fake_df.shape
fake_df.info()
fake_df.describe()
fake_df.head()
fake_df.isnull().sum()
# Rows and columns of fake news dataset
true_df.shape
true_df.info()
true_df.describe()
true_df.head()
true_df.isnull().sum()
# Adding 'Fake' column to our datasets then join them together
fake_df['Fake']=1
true_df['Fake']=0
concat_df=pd.concat([fake_df,true_df],axis = 0)
#%% Processamento Textual
concat_df['full_text'] = concat_df['title'] + ' ' + concat_df['subject']


# Funções para processamento de texto  
def process_text(text):  
    # Tokenização  
    tokens = word_tokenize(text.lower())  
    # Remover stopwords e não alfabéticos  
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]  
    # Lematização  
    lemmatizer = WordNetLemmatizer()  
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]  
    return ' '.join(lemmatized)  

# Aplicar o processamento no texto completo  
concat_df['processed_text'] = concat_df['full_text'].apply(process_text) 


#%% VETORIZAÇÃO
# dividindo target e features
y = concat_df['Fake']
x = concat_df['processed_text'] 
#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
# vetorizando
vectorizer = TfidfVectorizer()  
x_train_vectorized = vectorizer.fit_transform(x_train)  
x_test_vectorized = vectorizer.transform(x_test)  
# Inicializar o modelo e treinar  
model = MultinomialNB()  
model.fit(x_train_vectorized, y_train)  #treinando
# Prever no conjunto de teste  
y_pred = model.predict(x_test_vectorized)  

#%% Avaliar o modelo  
print("Acurácia:", round(accuracy_score(y_test, y_pred)*100,2),'%')
print(classification_report(y_test, y_pred))  
print(sk.metrics.confusion_matrix(y_test, y_pred))
# %% Visualização:
# Calcular a matriz de confusão  
cm = sk.metrics.confusion_matrix(y_test, y_pred)  
# Criar um heatmap  
plt.figure(figsize=(20, 11))  
sns.heatmap(cm, annot=True,fmt='d',
            xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'], cmap='viridis',
            annot_kws={"size": 24})  
plt.title('Matriz de Confusão',fontsize=25)  
plt.xlabel('Previsões',fontsize=25)  
plt.ylabel('Valores Reais',fontsize=25)  
plt.tight_layout()
plt.savefig('Confusionmatrix_FakeNews')
plt.show()     


