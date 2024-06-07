
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def preprocess(X):
    X_tensor = tf.convert_to_tensor(X)
    return X_tensor


def load_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # for layer in base_model.layers:
    #     layer.trainable = False
        
    model = models.Sequential([
        base_model,
        # add CNN layers
        layers.Dense(32, activation='relu'),
        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam',
                loss='mae',
                metrics=['mae'])

    return model

# def main():
#     # open a file .npz with the data
#     data = np.load('data.npz')
#     print(data)

#     model = load_model()
#     model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, callbacks=[EarlyStopping(patience=5)])
    

