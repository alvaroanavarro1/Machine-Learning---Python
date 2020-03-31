#Plantilla de datos faltantes

import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import pandas as pd

#Importar el dataset
dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Tratamiento de los datos faltantes
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
imputer = imputer.fit(x[:, 1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

