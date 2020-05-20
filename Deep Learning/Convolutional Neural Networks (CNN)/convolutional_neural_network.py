# Convolutional Neural Network

# Importar librerias
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
#tf.__version__

# Pre-procesado de datos

# Generar imagenes para el Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 2,
                                                 class_mode = 'binary')

# Creating the Test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 2,
                                            class_mode = 'binary')

# Construir la CNN

# Inicializar CNN
cnn = tf.keras.models.Sequential()

# Convulución
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))

# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Segunda capa de convolución
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Capa de salida
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Entrenamiento CNN

# Compilado de CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Entrenando la CNN y verificación de resultados con test_set
cnn.fit_generator(training_set,
                  steps_per_epoch = 20,
                  epochs = 20,
                  validation_data = test_set,
                  validation_steps = 20)