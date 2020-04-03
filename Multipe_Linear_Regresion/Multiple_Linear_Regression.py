#Regresión lineal multiple

#Pre procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression

#Importar el dataset
dataset = pd.read_csv('50_Startups.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Codificar datos categoricos 
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3])

#Codificar datos categoricos en variables dummy
onehotencoder = make_column_transformer((OneHotEncoder(),[3]), remainder = "passthrough")
x = onehotencoder.fit_transform(x)

#Evitar la trampa de las variables ficticias (Se elimino la columna California)

x = x[:, 1:]

#Dividir el dataset entrenamiento/testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Ajustar el modelo de regresión lineal multiple con el conjunto de entrenamiento
regression = LinearRegression()
regression.fit(x_train, y_train)

#Predicción de resultados en el conjunto de test
y_pred = regression.predict(x_test)