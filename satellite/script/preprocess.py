import tensorflow as tf

def convert_npz_to_bgr(image):
    # Load the .npz file
    # image = load_npz(i)
    # Iterate over all keys in the file
    image = image[:, :, :3]
    if image.shape[0] != 224 or image.shape[1] != 224:
            image = tf.image.resize(image, (224, 224))
    if image.shape[2] == 3:  # Check if the image has 3 color channels
            image = tf.reverse(image, axis=[-1])
    if image.shape != (224, 224, 3):
            raise ValueError(f"Image shape is {image.shape}, expected (224, 224, 3)")
    image = tf.cast(image, tf.float16)
    # Preprocess the image using VGG16's preprocessing function
    image = tf.keras.applications.vgg16.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    print('tipoooo', type(image))
    image = tf.convert_to_tensor(image)
    print('sizeee', image)
    return image