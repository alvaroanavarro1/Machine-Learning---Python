#Pre procesado de datos

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#Importar el dataset
dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Codificar datos categoricos 
#labelencoder_x = LabelEncoder()
#x[:,0] = labelencoder_x.fit_transform(x[:,0])

#Dividir el dataset entrenamiento/testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Escalado de variables
"""sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""
 
