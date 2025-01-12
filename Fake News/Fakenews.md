# Detecção de Fake News

Este código realiza a detecção de fake news utilizando um modelo de Naive Bayes. O processo inclui leitura de dados, processamento de texto, vetorização e avaliação do modelo.

## Importar Bibliotecas

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk as nlp
import sklearn as sk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
from nltk.stem import WordNetLemmatizer  
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

## Leitura dos Dados
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Visualizando as informações do dataset de fake news
fake_df.shape
fake_df.info()
fake_df.describe()
fake_df.head()
fake_df.isnull().sum()

# Visualizando as informações do dataset de true news
true_df.shape
true_df.info()
true_df.describe()
true_df.head()
true_df.isnull().sum()

# Adicionando a coluna 'Fake' nos datasets e juntando-os
fake_df['Fake'] = 1
true_df['Fake'] = 0
concat_df = pd.concat([fake_df, true_df], axis=0)
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

# Aplicando o processamento no texto completo  
concat_df['processed_text'] = concat_df['full_text'].apply(process_text)

## Vetorização e Divisão dos Dados




# Dividindo as variáveis de target e features
y = concat_df['Fake']
x = concat_df['processed_text'] 

# Dividindo em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# Vetorizando os textos
vectorizer = TfidfVectorizer()  
x_train_vectorized = vectorizer.fit_transform(x_train)  
x_test_vectorized = vectorizer.transform(x_test)  

# Inicializando o modelo e treinando
model = MultinomialNB()  
model.fit(x_train_vectorized, y_train)  # treinando

# Prevendo os resultados no conjunto de teste  
y_pred = model.predict(x_test_vectorized)  

# Acurácia e relatório de classificação
print("Acurácia:", round(accuracy_score(y_test, y_pred)*100, 2), '%')
print(classification_report(y_test, y_pred))  
print(sk.metrics.confusion_matrix(y_test, y_pred))

# Calculando a matriz de confusão
cm = sk.metrics.confusion_matrix(y_test, y_pred)

# Criando o heatmap
plt.figure(figsize=(20, 11))  
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'], cmap='viridis',
            annot_kws={"size": 24})  
plt.title('Matriz de Confusão', fontsize=25)  
plt.xlabel('Previsões', fontsize=25)  
plt.ylabel('Valores Reais', fontsize=25)  
plt.tight_layout()
plt.savefig('Confusionmatrix_FakeNews')
plt.show()

