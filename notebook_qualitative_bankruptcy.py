#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:01:36 2023

@author: douglvv
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### PREPARANDO O DATASET ###

# lendo os dados
data = pd.read_csv('qualitative_bankruptcy.txt', header=None)

# nomeando as colunas
data.columns = ['Industrial_Risk', 'Management_Risk', 'Financial_Flexibility', 'Credibility', 'Competitiveness', 'Operating_Risk', 'Class']

# mapeando a classe
class_mapping = {'B': 1, 'NB': 0}
data['Class'] = data['Class'].map(class_mapping)

# usando label encoder nos valores categoricos
encoder = LabelEncoder()
for col in data.columns[:-1]:
    data[col] = encoder.fit_transform(data[col])

# dividindo os dados em treinamento e teste
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# separando os dados de treinamento e classe (X e y)
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']

# base de treinamento normalizada
print(X_train)
print(y_train)



### NAIVE BAYES ###
from sklearn.naive_bayes import GaussianNB

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)

# previsao na base de teste
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']
predictions = naive_bayes_model.predict(X_test)

# testando a quantidade de acertos
accuracy = accuracy_score(y_test, predictions)
print(f'Acertos Naive Bayes: {accuracy:.2f}')

 # base de teste normalizada
#print(X_test)
#print(y_test)

### DECISION TREE ###
from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# previsao na base de teste
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']
predictions = decision_tree_model.predict(X_test)

# testando a quantidade de acertos
accuracy = accuracy_score(y_test, predictions)
print(f'Acertos Decision Tree: {accuracy:.2f}')



### RANDOM FOREST ###
from sklearn.ensemble import RandomForestClassifier

# criando o modelo random forest
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# previsao na base de teste
predictions = random_forest_model.predict(X_test)

# testando a quantidade de acertos
accuracy = accuracy_score(y_test, predictions)
print(f'Acertos Random Forest: {accuracy:.2f}')



### KNN ###
from sklearn.neighbors import KNeighborsClassifier

# criando o modelo KNN
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train, y_train)

# previsao na base de teste
predictions = knn_model.predict(X_test)

# testando a quantidade de acertos
accuracy = accuracy_score(y_test, predictions)
print(f'Acertos KNN: {accuracy:.2f}')


### SVM ###
from sklearn.svm import SVC

# criando o model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# previsao na base de teste
predictions = svm_model.predict(X_test)

# testando a quantidade de acertos
accuracy = accuracy_score(y_test, predictions)
print(f'AAcertos SVM (kernel rbf): {accuracy:.2f}')


### Redes Neurais ###





## Confusion matrix para gerar o plot gráfico dos erros e acertos
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)

#pip install yellowbrick
from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(random_forest_model)
cm.fit(X_train, y_train)
cm.score(X_test,y_test)

from sklearn.metrics import classification_report

classification_report(y_test, predictions)






