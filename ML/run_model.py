
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
    

def load_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

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

    model.compile(optimizer='adam',
                loss='mae')

    return model


def load_npz(max_i):
    tensor_list = ""
    for i in range(0,max_i):
        path = f"../download_preprocess/list_tensors_{i}.npz"
        X = np.load(path)
        images = [tf.convert_to_tensor(X[image]) for image in X.files]
        if i==0:
            tensor_list = images
        else:
            tensor_list.extend(images)
    return tf.convert_to_tensor(tensor_list)

def main():

    images = load_npz(5)
    df = pd.read_csv('../download_preprocess/img_info.csv', index_col=0)
    
    df = df[0:4495]
    y = df["cons_pc"]
    X_train = images[0:3596]
    y_train = y[0:3596]
    X_test = images[3596:]
    y_test = y[3596:]
    model = load_model()
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=10)])
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"test mean{np.mean(y_test)}, pred mean {np.mean(y_pred)}")
    print(r2)
main()