# -*- coding: utf-8 -*-
"""
MÚLTIPLE REGRESIÓN LINEAL
Restricciones para usar la RM:
    - Linealidad: relación lineal entre variables independientes y dependiente.
    - Homocedasticidad: variabilidad constante de errores.
    - Normalidad: las variables independientes tienen que ser normales.
    - Independencia de errores: errores sin relación entre sí.
    - Sin multicolinealidad: evitar "competencia" entre variables dummy.
"""

'''
5 Métodos para construir modelos de regresión múltiple:
    - All-in (meter todas las variables de una): A veces funciona, pero hay que conocer bien las variables para que no se cuele alguna irrelevante.
    - Eliminación hacia atrás: quitamos la variable menos significativa una por una.
    - Selección hacia adelante: añadimos la variable más significativa cada vez.
    - Eliminación bidireccional: combinamos hacia adelante y hacia atrás, es tedioso pero más preciso.
    - Comparación de Scores: probamos todas las combinaciones y nos quedamos con el mejor score (Akaike).
'''

# Importamos las librerías necesarias para el análisis.
import numpy as np  # Para manejar los datos numéricos
import matplotlib.pyplot as plt  # Para graficar los datos
import pandas as pd  # Para manipular los datos

# Cargamos el dataset con datos de startups
dataset = pd.read_csv("50_Startups.csv")

# Extraemos las variables independientes (X) y la dependiente (y).
X = dataset.iloc[:, :-1].values  # Todas menos la última columna
y = dataset.iloc[:, 4].values  # Última columna como nuestra variable a predecir

## Tratamiento de valores faltantes (NaN)
from sklearn.impute import SimpleImputer  # para rellenar valores NaN
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])  # Aplicamos la media en las columnas con NaN (edad y sueldo).

# Codificación de variables categóricas (esto convierte palabras en números).
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()  # Codifica la primera columna de categorías
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(onehotencoder.fit_transform(X), dtype=float)  # Para manejar las categorías de forma binaria (dummy variables)

# Evitamos variables ficticias extra (dummy variables).
X = X[:, 1:]  # Eliminamos la primera columna dummy.

# Dividimos el dataset en entrenamiento y prueba.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Escalamos los datos (estandarización) para evitar que valores grandes influyan más.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  # Los datos de prueba se escalan con los mismos parámetros del entrenamiento.

# MODELO DE REGRESIÓN LINEAL MÚLTIPLE
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)  # Entrenamos el modelo con los datos de entrenamiento

# Probamos el modelo con los datos de prueba
y_pred = regression.predict(X_test)

# Usamos eliminación hacia atrás para simplificar el modelo.
import statsmodels.api as sm
X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)  # Añadimos una columna de unos (intercepto)

SL = 0.05  # Nivel de significancia
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(y, X_opt).fit()
print(regression_OLS.summary())  # Mostramos un resumen estadístico del modelo

# Eliminamos variables menos significativas y repetimos el proceso para obtener el mejor modelo posible.
X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(y, X_opt).fit()
print(regression_OLS.summary())

# Visualizamos resultados del entrenamiento
plt.scatter(X_train[:, 0], y_train, color="red")
plt.plot(X_train[:, 0], regression.predict(X_train), color="blue")
plt.title("Sueldo vs. Años de Experiencia (Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()

# Visualización de resultados de prueba
plt.scatter(X_test[:, 0], y_test, color="red")
plt.plot(X_test[:, 0], regression.predict(X_test), color="blue")
plt.title("Sueldo vs. Años de Experiencia (Prueba)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()
