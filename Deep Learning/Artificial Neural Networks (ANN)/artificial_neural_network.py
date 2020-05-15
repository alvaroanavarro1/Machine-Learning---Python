# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
#tf.__version__
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Preprocesado de datos

# Importar el dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Codificar datos categoricos
# Codificar la columna "Gender"
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)

# One Hot Encoding para la columna "Geography"
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Escalado de variables
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)

# Separar en conjunto de test y de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Building the ANN

# Inicializamos la ANN
ann = tf.keras.models.Sequential()

# Añadir las capas de entrada y primera capa oculta
ann.add(tf.keras.layers.Dense(units=6, kernel_initializer= 'uniform',  activation='relu'))

# Añadir la segunda capa oculta
ann.add(tf.keras.layers.Dense(units=6, kernel_initializer= 'uniform',  activation='relu'))

# Añadir la capa de salida
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compilar la ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Entrenamiento de la ANN

# Entrenamiento de la ANN con el conjunto de entrenamiento
ann.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicciones y evaluacion del modelo

# Prediccion del conjunto de prueba
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
