
"""
Created on Wed Nov  6 19:07:43 2024

@author: Carlos-Javier

modelos de clasificacion bosuqes aleatorios (ensammble - learning )-ramdom forest
es la cantidad de arboles que yo necesite crear 
clasificacion de votos dpor mayoria 

proceso:
    1- slecciona un nuemro aleatorio de k de puntos del conjunto de entrenamiento 
    2- de estos puntos se construye un arbol de decision a esos k puntos de datos 
    3- elegimos un numero Ntree de arboles que queremos construir y repetir el paso 1 y 2
    4- para clasificar un nuevo punto, hacer que cada uno de los Ntree arboles elabore la prediccion
    a que categoria pertenece y asignar el nuevo punto a la categoria con mas votos 
    
    este modelo tiende al sobreajuste tener cuidado con eso 
regresion-> produce varibles numericas 

siempre se usa .values para los arboles 

"""

###importacion de librerias 
import numpy as np ## Libraría para tratamiento de números y herramientas matemáticas (np Alias)
import matplotlib.pyplot as plt  ## Librería para representación gráfica (pyplot es una sublibrearía - gráficos) (plt Alias)
import pandas as pd  ## Librería para manipular datos (pd Alias)

## Importar el Dataset
dataset  = pd.read_csv("Social_Network_Ads.csv")

##matris de caracteristca 
X = dataset.iloc[:,[2,3]].values#variable independientes en este caso la edad y el salario estimado
y = dataset.iloc[:,4].values# variable dependientes 

#dividir el dataset en conjunto de entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 

#escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#akjuste del calsificador en ekl conjunto de entrenamiento 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100,criterion = "entropy", random_state = 0)## el 100 representa cuantos arboles se va ocupar en este caso 100 puede ser 10 tambien
##criterio entropia indice gini es el default
## si la entropia 0 es y logro clasificar 100%
rf.fit(X_train, y_train)

##prediccion de los resultados con el conjutno de test
y_pred = rf.predict(X_test)

##analisis de resultados con la matris
##saver si la prediccion es b uena 
## se clacula la matriz de confusion sobre el modelo de testing 
## cuantas son las predicciones correctas 

from sklearn.metrics import confusion_matrix
conmet = confusion_matrix(y_test, y_pred)

##visualiza el algoritmo de train graficando los resultados 
##setp --->de año en año y de $500
##   cual es el porcenmtaje 
##para la comprobacion se suma en diagonal

##visualizae el algoritmo de train graficamente con los resudltados 
### Visualizar el algotirmo de train graficamente con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2 , rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("bosuqe aleatorios - ramdom forest (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
 
 
### Visualizar el algotirmo de test graficamente con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),##aqui puedo poner en el 0.01 antes del step para los años 
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))## el 0.01 para que salga en la grafica 
plt.contourf(X1, X2 , rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Bosque aleatorios - ramdom forest (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()



























