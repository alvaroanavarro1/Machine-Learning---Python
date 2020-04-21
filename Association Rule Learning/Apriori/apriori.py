# Apriori

#Pre procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

#Importar el dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

#Transformar los datos por transacción 
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Entrenar el algoritmo de Apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualización de resultados
results = list(rules)

print(results[0])
print(results[1])