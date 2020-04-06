#SVR

#Pre procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

#Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#Escalado de variables
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#Ajustar la regresión con el dataset
regression = SVR(kernel = 'rbf')
regression.fit(x, y)

#Predicción de nuestro modelo con SVR
y_pred = regression.predict(sc_x.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)

#Visualizacion de los resultados del modelo SVR
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x_grid), sc_y.inverse_transform(regression.predict(x_grid)), color = 'blue')
plt.title('Modelo de Regresión (SVR)')
plt.xlabel('Posición del empleado')
plt.ylabel('Sueldo en $')
plt.show()


