# UCB

#Pre procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
#Importar el dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Algoritmo de Upper Confidence Bound
N = 10000 #Numero de usuarios
d = 10 #Numero de anuncios'

number_of_selections = [0]*d
sums_of_rewards = [0]*d
ads_selected = []
total_reward = 0

for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if (number_of_selections[i]>0):
            # Calculo de recompensa media
            avg_reward = sums_of_rewards[i] / number_of_selections[i]
            # Calculo de delta sub i
            delta_i = math.sqrt(3/2*math.log(n+1) / number_of_selections[i])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
#Visualización de resultados
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()
