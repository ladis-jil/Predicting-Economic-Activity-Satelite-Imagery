
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

def load_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = True
        
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(200, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mae')
    return model

def main():
    # Usar CPU para el procesamiento
    tf.config.set_visible_devices([], 'GPU')
    
    images = np.load("preprocessed_images.npy")
    print("Loaded images shape:", images.shape)
    # images = images / 255.0  # Normalizar imágenes

    
    # df = pd.read_csv('../raw_data/images_without_transparency.csv')
    # y = df["cons_pc"].values
    
    # # Utilizar un conjunto de datos reducido para pruebas iniciales
    # X_train = images[:100]
    # y_train = y[:100]
    # X_test = images[100:130]
    # y_test = y[100:130]
    
    # print("types", type(X_train), type(X_test), type(y_train), type(y_test))
    # print("size X_train", X_train.shape)
    # print("size y_train", y_train.shape)
    # print("size X_test", X_test.shape)
    # print("size y_test", y_test.shape)
    
    # # Convertir np.arrays a tensores de TensorFlow en partes más pequeñas
    # X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    # y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    # X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    # y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    
    # print("Tensor conversion successful.")
    
    # model = load_model((112, 112, 3))
    # model.fit(X_train, y_train, epochs=100, batch_size=2, validation_split=0.2, callbacks=[EarlyStopping(patience=10)])

if __name__ == "__main__":
    main()