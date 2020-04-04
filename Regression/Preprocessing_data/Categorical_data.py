#Plantilla de datos categoricos

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

#Importar el dataset
dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Codificar datos categoricos 
#labelencoder_x = LabelEncoder()
#x[:,0] = labelencoder_x.fit_transform(x[:,0])

#Codificar datos categoricos en variables dummy
onehotencoder = make_column_transformer((OneHotEncoder(),[0]), remainder = "passthrough")
x = onehotencoder.fit_transform(x)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
