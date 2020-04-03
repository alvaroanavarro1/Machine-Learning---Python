#Regresión linear simple

#Pre procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Importar el dataset
dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Dividir el dataset entrenamiento/testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#Crear modelo de regresión simple con el conjunto de entrenamiento
regression = LinearRegression()
regression.fit(x_train, y_train)

#Predecir el conjunto de prueba
y_pred = regression.predict(x_test)

#Visualización de datos de entrenamiento
plt.scatter(x_train, y_train, color = "red", label = "Train data")
plt.scatter(x_test, y_test, color = "green", label = "Test data")
plt.plot(x_train, regression.predict(x_train), color = "blue", label = "Regression line")
plt.title("Sueldo vs A�os de experiencia")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo ($)")
plt.legend()
plt.show()