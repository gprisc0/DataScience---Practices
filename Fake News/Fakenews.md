# Detecção de Fake News com Naive Bayes

Neste projeto, vamos construir um modelo de detecção de fake news utilizando o algoritmo Naive Bayes.

## 1. Importação das Bibliotecas

Primeiro, importamos as bibliotecas necessárias para a análise.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
```
---
## 2. Leitura de Dados
Aqui, vamos carregar os datasets de notícias falsas e verdadeiras.
```python
Copiar código
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Exibir informações dos datasets
print(fake_df.shape)
print(fake_df.info())
print(fake_df.describe())
print(fake_df.head())
print(fake_df.isnull().sum())

print(true_df.shape)
print(true_df.info())
print(true_df.describe())
print(true_df.head())
print(true_df.isnull().sum())
```
---
## 3. Pré-processamento de Texto
Aqui, aplicamos o pré-processamento nos textos das notícias, como tokenização, remoção de stopwords e lematização.
```python
def process_text(text):
    # Tokenização
    tokens = word_tokenize(text.lower())
    # Remover stopwords e palavras não alfabéticas
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    # Lematização
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized)

# Aplicando o processamento ao texto
concat_df['processed_text'] = concat_df['full_text'].apply(process_text)
```
---
## 4. Treinamento do Modelo
Neste passo, vamos treinar o modelo Naive Bayes para classificar as notícias como verdadeiras ou falsas.
```python
Copiar código
# Dividindo target e features
y = concat_df['Fake']
x = concat_df['processed_text']

# Dividindo os dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vetorização dos textos com TF-IDF
vectorizer = TfidfVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

# Inicializando o modelo Naive Bayes
model = MultinomialNB()
model.fit(x_train_vectorized, y_train)
```
---
## 5. Avaliação do Modelo
Após treinar o modelo, vamos avaliá-lo calculando a acurácia e exibindo a matriz de confusão.

```python
# Prevendo os resultados no conjunto de teste
y_pred = model.predict(x_test_vectorized)

# Avaliação do modelo
print("Acurácia:", round(accuracy_score(y_test, y_pred) * 100, 2), '%')
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
```
---
## 6. Visualização da Matriz de Confusão
Por fim, vamos visualizar a matriz de confusão com um heatmap.

```python
# Calculando a matriz de confusão
cm = sk.metrics.confusion_matrix(y_test, y_pred)

# Criando o heatmap da matriz de confusão
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['True', 'Fake'], 
            yticklabels=['True', 'Fake'], 
            annot_kws={"size": 16})
plt.title('Matriz de Confusão', fontsize=18)
plt.xlabel('Previsões', fontsize=14)
plt.ylabel('Valores Reais', fontsize=14)
plt.tight_layout()
plt.show()
```


