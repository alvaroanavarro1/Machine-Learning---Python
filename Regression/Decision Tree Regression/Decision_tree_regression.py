#Regresión con arboles de decisión

#Pre procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

#Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values


#Ajustar la regresión con el dataset
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(x, y)

#Predicción de nuestro modelo con Arbol de regresión
y_pred = regression.predict([[6.5]])


#Visualizacion de los resultados del modelo SVR
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regression.predict(x_grid), color = 'blue')
plt.title('Modelo de Regresión con Arbol')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo en $')
plt.show()