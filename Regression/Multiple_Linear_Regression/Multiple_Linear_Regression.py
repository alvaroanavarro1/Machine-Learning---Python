#Regresión lineal multiple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#Pre procesado de datos
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

#Construir el modelo optimo utilizando la eliminacion hacia atras
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
sl = 0.05
x_opt = np.array(x[:,[0, 1, 2, 3, 4, 5]], dtype=float)

#Funcion para eliminación hacia atras
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range (0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if(regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x 

#Prueba de la funcio eliminación hacia atras
x_Modeled = backwardElimination(x_opt, sl)
                    
"""
Metodo manual para el calculo de la x_Modeled utilizando eliminación hacia atras
#Primera corrida
x_opt = np.array(x[:,[0, 1, 2, 3, 4, 5]], dtype=float)
regression_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regression_OLS.summary())

#Segunda corrida
x_opt = np.array(x[:,[0, 1, 3, 4, 5]], dtype=float)
regression_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regression_OLS.summary())

#Tercera corrida
x_opt = np.array(x[:,[0, 3, 4, 5]], dtype=float)
regression_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regression_OLS.summary())

#Cuarta corrida
x_opt = np.array(x[:,[0, 3, 5]], dtype=float)
regression_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regression_OLS.summary())

#Quinta corrida para ser extrictos 
x_opt = np.array(x[:,[0, 3]], dtype=float)
regression_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regression_OLS.summary())"""


#Ajustar el modelo de regresión lineal multiple con el conjunto de entrenamiento para y y x_modeled para x
regression = LinearRegression()
regression.fit(x_Modeled, y)

#Predicción de resultados en el conjunto de test
y_pred2 = regression.predict(x_Modeled)


#Visualizacion de datos
#Ajuste de variables a plotear
x_Modeled = x_Modeled[:,1]
x = x[:,3]
#Modelo Final luego de eliminación hacia atras
plt.plot(x,y,'o', color = "red", label = "Real data")
plt.plot(x_Modeled, y_pred2, color = "blue", label = "Regression Line")
plt.title("Starups Profit vs R&D spend  ($) (Optimum)")
plt.ylabel("Profit ($)")
plt.xlabel("R&D spend")
plt.legend()
plt.show()

#
plt.plot(y_pred,'o', color = "red", label = "Real data")
plt.plot(y_test,'o', color = "blue", label = "Predicted data")
plt.title("Starups Profit vs Expenses ($)")
plt.ylabel("Profit ($)")
plt.xlabel("Expenses")
plt.xticks([])
plt.legend()
plt.show()
