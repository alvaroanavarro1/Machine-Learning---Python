#Pre procesado de datos

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset
dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values