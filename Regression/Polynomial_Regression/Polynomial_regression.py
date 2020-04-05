#Regresion lineal polinomica

#Pre procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#Ajustar la regresion lineal con el dataset
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Ajustar la regresion polinomica con el dataset
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#Visualizacion de los resultados del modelo lineal
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Modelo de Regresión Lineal')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo en $')
plt.show()

#Visualizacion de los resultados del modelo polinomico
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Modelo de Regresión Polinómica')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo en $')
plt.show()

#Predicción de nuestros modelos
print('Para el modelo lineal, la predicción de un puesto 6.5 es:' + str(lin_reg.predict([[6.5]])) + '$\n')
print('Para el modelo polinomico, la predicción de un puesto 6.5 es:' + str(lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))) + '$')

