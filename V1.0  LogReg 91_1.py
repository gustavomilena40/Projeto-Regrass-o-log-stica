# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:21:26 2020

@author: Gustavo
"""
#importações
import pandas as pd
import sklearn as sk 
import numpy as np
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.datasets import load_digits
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from unit_matplotlib import *

stpw = set(stopwords.words('portuguese')) # seta var de stopwords

col_names = ['DESCRICAO_DI','NCM'] #colunas da planilha

pima = pd.read_csv("91_1.csv", header=None, names=col_names, delimiter=';')#ler planilha

pima.DESCRICAO_DI = pima.DESCRICAO_DI.str.lower()#tranforma a planilha em minúsculo

pima.DESCRICAO_DI=pima.DESCRICAO_DI.astype(str) #transforma tudo em string

#tokeniza a planilha e joga os tokens na coluna 'token'
pima['token']  = pima["DESCRICAO_DI"].apply(nltk.word_tokenize)
print(pima.DESCRICAO_DI[1],'\n')
print(pima.token[1],'\n')

#tira as stops words e joga na coluna 'tokens_sem_Stpw'
pima['token_sem_stop']  = pima["token"].apply(lambda x: [w for w in x  if not w in stpw])
print(pima.token_sem_stop[1],'\n')
#tranforma todos os valores da coluna em string de novo, coreção erro 'float' object has no attribute 'lower'
pima.token_sem_stop=pima.token_sem_stop.astype(str)

#faz a vetorização TFIDF transformando cada palavra em valores numericos
from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer() #instancia
vector.fit(pima.token_sem_stop) #joga coluna token_sem_stop no TfidfVectorizer.fit

tfidf = vector.fit_transform(pima.token_sem_stop) #faz efetivamente o tfifd 

pima['tfidf']=list(tfidf)#jogando o TFIFD em uma coluna apenas para validação
print(pima.tfidf[1],'\n')
'''fim processamento de texto'''

#define X e y, e separa os dados em treino e teste
X = tfidf # Features
y = pima.NCM # Target variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#define algoritmo de machine learning
logreg = LogisticRegression(C=4.7)

#joga dados no algoritmo faz repetição e os fit
logreg.fit(X_train,y_train)



#faz a predição dos dados no conjunto de teste
y_pred=logreg.predict(X_test)

'''fim machine learning'''




'''analise do resultados'''
p_ncm0= list(logreg.predict_proba(X)[:,0])
p_ncm1= list(logreg.predict_proba(X)[:,1])
p_ncm2= list(logreg.predict_proba(X)[:,2])
p_ncm3= list(logreg.predict_proba(X)[:,3])



#resultado do modelo
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
#print(y_pred)
#print(logreg.predict_proba(X)[:,0])
#print(logreg.predict_proba(X)[:,1],'\n')

validacao=pd.DataFrame(list(X_test),columns=['teste'])
print('Posição 100 conjunto teste: \n', validacao.teste[90],'\n')
print('Palavras: \n')
print(vector.get_feature_names()[649])
print(vector.get_feature_names()[1296])
print(vector.get_feature_names()[288])
print(vector.get_feature_names()[607],'\n')
print('Probabilidade da descrição ser NCM 49019900: ',p_ncm0[90])
print('Probabilidade da descrição ser NCM 49111010: ',p_ncm1[90],'\n')


print(cnf_matrix)
print(classification_report(y_test, y_pred, zero_division=1))

classes = ['49019900', '49029000', '49089000', '49111010', '49111090', '71069290', '71129100', '71131900', '71159000']
plot_confusion_matrix(cnf_matrix,classes)

#plot_grafico_X_Y(p_ncm0,p_ncm1,'49111010','49019900')



#cv = ShuffleSplit(n_splits=20, test_size=0.1, random_state=0)
#plot_learning_curve(logreg, 'title', X, y, ylim=(0.7, 1.01),
 #                  cv=cv, n_jobs=4)
