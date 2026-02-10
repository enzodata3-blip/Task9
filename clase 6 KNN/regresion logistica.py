# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:07:43 2024

@author: Carlos-Javier

algoritpomos de clasificacion KNN 
datos categoricos pero la clasificacion ya no es lineal
para comenzar se puede un nuevo dato, para identificar a que lado se va acercar
es la teoria del vecino mas cercano 
loq ue hace este algoritmo es:
1. eligue un numero de K vecinos se recomienda que sea impar, nota el default de ptyhon es 5 
2. tomar lps k vecinos ,mas cercanos del nuevo dato. segun la distancia eulclidea
3. entre esos vecinos, contar el mimero de puntos qie pertenece a cada categoria 
4. asignar el nuevo dato a la categoria con mas vecinos en ella 
solo utiliza cualitativas




aqui entra la probabilidad 
esta entre 0 y 1 y se responde a aciones muy censillas por eso no se usa las cuantitativas sino cualitativas 
un modelo de calsificacion que se llama knn 
cuando predigo variables categoricas 
debe a´parecer la palabra clasificacion 
las variables comparables se escla 
y escalar es ponerles en el mismo rango 


"""

###importacion de librerias 
import numpy as np ## Libraría para tratamiento de números y herramientas matemáticas (np Alias)
import matplotlib.pyplot as plt  ## Librería para representación gráfica (pyplot es una sublibrearía - gráficos) (plt Alias)
import pandas as pd  ## Librería para manipular datos (pd Alias)

## Importar el Dataset
dataset  = pd.read_csv("Social_Network_Ads.csv")

##matris de caracteristca 
X = dataset.iloc[:,[2,3]]#variable independientes en este caso la edad y el salario estimado
y = dataset.iloc[:,-1]# variable dependientes 

#dividir el dataset en conjunto de entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 

#escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

##ajuasdtar el modelo de regresion logistioca en el conjutno de entrenmamiento
## aprender a predecir las clasificaciones 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
## cuando p e 1 es lineal y cuando es 2 es eclidiana
knn.fit(X_train,y_train)

##prediccion de los resultados con el conjutno de test
y_pred = knn.predict(X_test)

##analisis de resultados con la matris
##saver si la prediccion es b uena 
## se clacula la matriz de confusion sobre el modelo de testing 
## cuantas son las predicciones correctas 

from sklearn.metrics import confusion_matrix
conmet = confusion_matrix(y_test, y_pred)

##65 no comprAN Y 24 SI 
## 8 Y 3 CORRECVTAS 
##   cual es el porcenmtaje 
##para la comprobacion se suma en diagonal

##visualizae el algoritmo de train graficamente con los resudltados 
### Visualizar el algotirmo de train graficamente con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2 , knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("K-NN (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
 
 
### Visualizar el algotirmo de test graficamente con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2 , knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("K-NN (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


























