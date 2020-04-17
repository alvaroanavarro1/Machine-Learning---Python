#Clustering Jerarquico

#Pre procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

#Importar el dataset
dataset = pd.read_csv('Mall_Customers.csv')

x = dataset.iloc[:, 3:5].values

# Utilizar el dendrograma para encontrar el número óptimo de clusters
dendrogram = sch.dendrogram(sch.linkage(x, method = "ward"))
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")
plt.show()

# Ajustar el clustetring jerárquico a nuestro conjunto de datos
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(x)

# Visualización de los clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 50, c = "red", label = "Cautos")
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 50, c = "blue", label = "Estandard")
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 50, c = "green", label = "Objetivo")
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 50, c = "cyan", label = "Descuidados")
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 50, c = "magenta", label = "Conservadores")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de Gastos (1-100)")
plt.legend()
plt.show()