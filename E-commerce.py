# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 09:58:38 2025

@author: Gabriel Prisco
Problema: Segmentar clientes de um E-commerce através de Clusters
Tarefas:
1) Carregue e prepare o dataset, preprocesse os dados.
2) Construa uma rede neural CNN e a treine.
3) Avalie o modelo no conjunto teste."""
# %% importe as Bibliotecas
import sklearn as sk
from sklearn import preprocessing,cluster,metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#%% Carregamento do dataset
df = pd.read_csv('data.csv',encoding="ISO-8859-1")

# Verificar informações do DataFrame  
df.info()
df.shape
df.head(10)
df.describe()
#%% Pré-processamento dos dados  
df['Amount']=df['UnitPrice']*df['Quantity']
df.isna().sum()
df.dropna(inplace=True)
# Selecionar características relevantes para clustering  
# Exemplo: usando 'Amount' e 'Frequency'
X=df.groupby(['CustomerID']).agg({'Amount':'sum'})
X['Frequency']=df['CustomerID'].value_counts()
# Tirar Outliers
# Cálculo do Z-score  
z_scores = np.abs(stats.zscore(X))  
X = X[(z_scores < 2).all(axis=1)]  

# Normalizar os dados  
scaler = sk.preprocessing.StandardScaler()  
X_scaled = scaler.fit_transform(X)   
# Escolher o número de clusters  
inertias = []  
for i in range (2,10):
    kmeans = sk.cluster.KMeans(n_clusters=i,random_state=42)  
    kmeans.fit(X_scaled) 
    score = sk.metrics.silhouette_score(X_scaled, kmeans.labels_)
    db_index = sk.metrics.davies_bouldin_score(X_scaled, kmeans.labels_)
    ch_index = sk.metrics.calinski_harabasz_score(X_scaled, kmeans.labels_)
    inertias.append(kmeans.inertia_) 
    print("For n_clusters =", i, "Silhouette_score is:", score,', Inertia:',kmeans.inertia_,
          "Davies-Bouldin",db_index,"and CH_index",ch_index)

#%% Método do cotovelo:    
# Plotar o gráfico do cotovelo  
plt.plot(range(2, 10), inertias, marker='o')  
plt.title('Método do Cotovelo')  
plt.xlabel('Número de Clusters (k)')  
plt.show()  

#%% Considerando Todos os Critérios escolhemos o melhor K ( no caso 4 foi o melhor)
kmeans = sk.cluster.KMeans(n_clusters=4,random_state=42)
kmeans.fit(X_scaled) 
#%% Adicionar os rótulos dos clusters ao DataFrame original  
X['Cluster'] = kmeans.labels_  
clusters = kmeans.labels_
#%% VALORES Escalados(0-1)
# Visualização
centroids = kmeans.cluster_centers_   
plt.figure(figsize=(10, 6))  
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, cmap='cividis') 
plt.scatter(centroids[:, 0], centroids[:, 1], c='darkred', s=300, alpha=0.75, marker='X', label='Centróides')    
# Adicionar título e rótulos  
plt.title('Clusters de Clientes')  
plt.xlabel('Amount')  
plt.ylabel('Frequency')  
plt.legend()  
plt.show()  
#%% VALORES NÃO Escalados
#Visualização 
centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_))
plt.figure(figsize=(10, 6))  
plt.scatter(X['Amount'], X['Frequency'], c=X['Cluster'], cmap='cividis') 
plt.scatter(centroids[[0]], centroids[[1]], c='darkred', s=300, alpha=0.75, marker='X', label='Centróides')    
# Adicionar título e rótulos  
plt.title('Clusters de Clientes')  
plt.xlabel('Amount')  
plt.ylabel('Frequency')  
plt.legend()  
plt.savefig('Clusters de Clientes')
plt.show()  









