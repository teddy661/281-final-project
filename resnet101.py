import numpy as np
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image

# Change input shape
input_shape = (64, 64, 3)  # Assuming 3 channels for RGB images
new_input = Input(shape=input_shape)

# Load the pre-trained ResNet-101 model
model = ResNet101(weights="imagenet", input_tensor=new_input, include_top=False)
# resnet101 = ResNet101(weights='imagenet', input_tensor=new_input, include_top=False, pooling='avg')
for layer in model.layers:
    layer.trainable = False


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def get_resnet_embeddings(image_path):
    img_array = preprocess_image(img_path)
    embeddings = model.predict(img_array)
    return np.squeeze(embeddings)  # remove the extra dimension


img_path = "sign_data/Test/00000.png"
embeddings = get_resnet_embeddings(img_path)

print(model.summary())
np.save("embeddings.npy", embeddings)
print(embeddings)
print(embeddings.shape)
