# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:14:45 2025

@author: Gabriel Prisco


Problema: Reconhecer dígitos de imagens ( algarismos de 0 a 9)
Tarefas:
1) Carregue e prepare o dataset, preprocesse os dados.
2) Construa uma rede neural CNN e a treine.
3) Avalie o modelo no conjunto teste."""
# %% importe as Bibliotecas
from tensorflow.keras import layers, models
from keras.layers import Activation,Conv2D,MaxPooling2D,Flatten,Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%% Carregamento do dataset
df_train = pd.read_csv('digit-recognizer/train.csv')
df_train.head(10)
df_test = pd.read_csv('digit-recognizer/test.csv')
x_train = np.asarray(df_train.iloc[:,1:]).reshape(df_train.shape[0],28,28,1) #imagens
y_train = df_train.iloc[:,0] #labels 
x_test = np.asarray(df_test).reshape(df_test.shape[0],28,28,1) #imagens

#%% Normalizar os dados (opcional)  
x_train = x_train.astype('float32') / 255.0 #(255 é o valor máximo do byte) 
x_test = x_test.astype('float32') / 255.0 
#%% Definição da Rede Neural 

modelo = models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]),
    MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.3),
    Flatten(),
    Dense(64,activation='relu'),
    layers.Dropout(0.3),
    Dense(10,activation='softmax')
    ])
modelo.summary()
modelo.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#%% Treino do Modelo com método fit
Result= modelo.fit(x_train,y_train,batch_size=128,epochs=10,validation_split=0.3)
# %% Avaliação do Modelo com Evaluate do módulo Models score[0] = loss e score[1]=acc
score = modelo.evaluate(x_train,y_train, verbose=1)

print(round(score[1]*100,2),'%')
predictions = modelo.predict(x_test)  
predicted_classes = np.argmax(predictions, axis=1)
num_images = 20  
rows = 4  
cols = 5  
plt.figure(figsize=(15, 10))  # Tamanho da figura ajustado para melhorar a visualização  
for i in range(num_images):  
    plt.subplot(rows, cols, i + 1)  
    plt.imshow(x_test[i].reshape(28, 28), cmap='winter')  
    plt.title(f'{predicted_classes[i]}')  
    plt.axis('off')  

plt.tight_layout()  
plt.savefig('Digit Recognition')
plt.show()  





