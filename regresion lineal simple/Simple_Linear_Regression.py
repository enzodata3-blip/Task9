
# Ahora vamos a importar las librerías necesarias.
import numpy as np  # Trabaja con operaciones numéricas, alias np.
import matplotlib.pyplot as plt  # Para gráficos, usamos la sublib 'pyplot' (plt).
import pandas as pd  # Para manipular datos, alias pd.

# Importamos el dataset
dataset  = pd.read_csv("50_Startups.csv")

# Variables independientes "X" y la que queremos predecir "Y"
X = dataset.iloc[:,:-1].values  # Todas las filas, menos la última columna (que es la que vamos a predecir).
y = dataset.iloc[:, 4].values  # Columna que queremos predecir.

'''
# Lidiamos con datos faltantes (NAN) usando Imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")  # Rellenamos NANs con el promedio
imputer = imputer.fit(X[:, 1:3])  # Solo en columnas de edad y sueldo
X[:, 1:3] = imputer.transform(X[:, 1:3])  # Transformamos y sustituimos en X.
'''

# Codificamos las variables categóricas
from sklearn.preprocessing import OneHotEncoder, LabelEncoder  # LabelEncoder y OneHot para dummies.
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()  # Creamos un LabelEncoder para categorías
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])  # Convertimos texto a números

# Variable Dummy: convertimos categorías en varias columnas (uno para cada categoría)
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')
X = np.array(onehotencoder.fit_transform(X), dtype=float)

# Para evitar multicolinealidad, quitamos una de las variables dummy
X = X[:,1:]

# Dividimos los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 80% entrenamiento, 20% prueba

# Escalado de variables (standarización y normalización)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()  # Creamos un escalador
X_train = sc_X.fit_transform(X_train)  # Ajustamos con los datos de entrenamiento
X_test  = sc_X.transform(X_test)  # Aplicamos el escalado a los datos de prueba

# Creamos el modelo de Regresión Lineal Múltiple
from sklearn.linear_model import LinearRegression
regression = LinearRegression()  # Creamos el objeto regresión
regression.fit(X_train, y_train)  # Entrenamos con los datos

# Hacemos predicciones con el modelo
y_pred = regression.predict(X_test)  # Predicciones de los datos de prueba

# Eliminación hacia atrás (para optimizar el modelo)
import statsmodels.api as sm 
X = np.append(arr=np.ones((X.shape[0],1)).astype(int), values=X, axis=1)  # Añadimos una columna de unos para la constante
SL = 0.05  # Nivel de significancia

# Probamos distintas combinaciones de variables, quitando las menos significativas
X_opt = X[:, [0,1,2,3,4,5]]
regression_OLS = sm.OLS(y, X_opt).fit()
print(regression_OLS.summary())

# Repetimos quitando variables
X_opt = X[:, [0,1,3,4,5]]
regression_OLS = sm.OLS(y, X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:, [0,3,4,5]]
regression_OLS = sm.OLS(y, X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:, [0,3,5]]
regression_OLS = sm.OLS(y, X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:, [0,3]]
regression_OLS = sm.OLS(y, X_opt).fit()
print(regression_OLS.summary())

# Visualizamos resultados
plt.scatter(X_train[:, 0], y_train, color="red")  # Datos reales (entrenamiento)
plt.plot(X_train[:, 0], regression.predict(X_train), color="blue")  # Línea de predicción
plt.title("Ganancias vs I+D (Entrenamiento)")
plt.xlabel("Gasto en I+D")
plt.ylabel("Ganancias ($)")
plt.show()

plt.scatter(X_test[:, 0], y_test, color="red")  # Datos reales (prueba)
plt.plot(X_test[:, 0], regression.predict(X_test), color="blue")  # Línea de predicción
plt.title("Ganancias vs I+D (Prueba)")
plt.xlabel("Gasto en I+D")
plt.ylabel("Ganancias ($)")
plt.show()
