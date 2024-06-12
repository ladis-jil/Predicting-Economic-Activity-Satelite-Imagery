
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import r2_score

def load_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False
        
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(200, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(80, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mae')
    return model

def main():
    # Usar CPU para el procesamiento
    # tf.config.set_visible_devices([], 'GPU')
    
    images = np.load("../preprocessed_images.npy")
    print("Loaded images shape:", images.shape)
    images = images / 255.0  # Normalizar imágenes

    df = pd.read_csv('../raw_data/images_without_transparency.csv')
    y = df["cons_pc"].values
    
    # Utilizar un conjunto de datos reducido para pruebas iniciales
    X_train = images[:2432]
    y_train = y[:2432]
    X_test = images[2432:]
    y_test = y[2432:]
    
    images, y = shuffle(images, y)
    
    print("types", type(X_train), type(X_test), type(y_train), type(y_test))
    print("size X_train", X_train.shape)
    print("size y_train", y_train.shape)
    print("size X_test", X_test.shape)
    print("size y_test", y_test.shape)
    
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float16)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float16)
    
    print("Tensor conversion successful.")
    
    model = load_model()
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    print("R2 score:", r2)
    model.save("training_model_0.h5")
    
if __name__ == "__main__":
    main()