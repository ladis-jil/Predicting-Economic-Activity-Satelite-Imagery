import numpy as np
import os
import tensorflow as tf
from scipy.ndimage import rotate
import cv2
import time
start = time.time()
def load_npz(i):
    path = f'image_download/image_{i}.npz'
    X = np.load(path)
    image = np.array([X[row] for row in X.files])
    return image

def convert_npz_to_bgr(i):
    # Load the .npz file
    image = load_npz(i)
    # Iterate over all keys in the file
    if image.shape[0] != 224 or image.shape[1] != 224:
            image = tf.image.resize(image, (224, 224))
    if image.shape[2] == 3:  # Check if the image has 3 color channels
            image = tf.reverse(image, axis=[-1])
    if image.shape != (224, 224, 3):
            raise ValueError(f"Image shape is {image.shape}, expected (224, 224, 3)")
    image = tf.cast(image, tf.float32)
    # Preprocess the image using VGG16's preprocessing function
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image
    # for key in data.keys():
    #     # Extract the image data
    #     image = data[key]
    #     image = np.expand_dims(image, axis=0)
    #     # Check if the image needs to be resized
    #     if image.shape[0] != 224 or image.shape[1] != 224:
    #         image = tf.image.resize(image, (224, 224))
    #     # Convert the image from RGB to BGR (if necessary)
    #     # VGG16 expects the image in BGR format.
    #     if image.shape[2] == 3:  # Check if the image has 3 color channels
    #         image = tf.reverse(image, axis=[-1])  # Reversing channels from RGB to BGR
    #     # Ensure the image is in the shape (224, 224, 3)
    #     if image.shape != (224, 224, 3):
    #         raise ValueError(f"Image shape is {image.shape}, expected (224, 224, 3)")
    #     # Convert the image to float32, as VGG16 expects inputs in float32
    #     image = tf.cast(image, tf.float32)
    #     # Preprocess the image using VGG16's preprocessing function
    #     image = tf.keras.applications.vgg16.preprocess_input(image)
    #     # Store the preprocessed image
    #     bgr_images[key] = image
    # return bgr_images


index_list = []

path='image_download/'
for filename in os.listdir(path):
    if filename.endswith('.npz'):
        # file_path = os.path.join(path, filename)
        index = int(filename.split('_')[-1].split('.')[0])
        index_list.append(index)

# batches = np.arange(8,22)
# for batch in batches:
# img_list = []
# for index in index_list[19778:19794]:
#     img_list.append(convert_npz_to_bgr(index))
# np.savez(f'download_preprocess/list_tensors_22', *np.array(img_list))
# print(f'saved list_tensors_22')
        
# print('It took', time.time()-start, 'seconds.')

