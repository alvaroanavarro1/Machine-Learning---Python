#Arboles de regresión aleatoria

#Pre procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#Ajustar la regresión con el dataset
regression = RandomForestRegressor(n_estimators = 300, random_state = 0)
regression.fit(x, y)

#Predicción de nuestro modelo con random forest
y_pred = regression.predict([[6.5]])

#Visualizacion de los resultados del modelo random forest
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y, color = 'red')
plt.plot(x_grid, regression.predict(x_grid), color = 'blue')
plt.title('Modelo de Regresión Random Forest')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo en $')
plt.show()
