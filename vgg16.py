import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.preprocessing import image

# Change input shape
input_shape = (64, 64, 3)  # Assuming 3 channels for RGB images
new_input = Input(shape=input_shape)

# Load the pre-trained VGG16 model
#vgg16 = VGG16(weights="imagenet", input_tensor=new_input, include_top=False)
vgg16 = VGG16(weights="imagenet", input_tensor=new_input, include_top=False, pooling='avg')
flatten_layer = Flatten()(vgg16.output)
model = Model(inputs=vgg16.input, outputs=flatten_layer)

for layer in model.layers:
    layer.trainable = False


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def get_vgg_embeddings(img_path):
    img_array = preprocess_image(img_path)
    embeddings = model.predict(img_array)
    return np.squeeze(embeddings)  # remove the extra dimension


img_path = "sign_data/Test/00000.png"
embeddings = get_vgg_embeddings(img_path)

print(model.summary())
np.save("vgg16_embeddings.npy", embeddings)
print(embeddings)
print(embeddings.shape)
