###############################################################################
##
## This file will live outside the feature creation pipeline since we don't
## want to continually start and stop the model to predict against images.
## This file will read an existing features.parquet file and add the VGG16
## feature column to it.
##
## Batch load all images into numpy array
## then predict!
from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Flatten, Input

# Read the parquet file, this takes a while. Leave it here
train_features_file = Path("data/train_features.parquet")
if not train_features_file.exists():
    print("No train features file found. Please run the create_features_table first")
    exit(1)

test_features_file = Path("data/test_features.parquet")
if not test_features_file.exists():
    print("No train features file found. Please run the create_features_table first")
    exit(1)

dft = pl.read_parquet(test_features_file, use_pyarrow=True, memory_map=True)
image_df = dft.select(["Image"])
image_df = image_df.with_columns(
    pl.col("Image").map_elements(lambda x: np.load(BytesIO(x))).alias("NumPy")
)
input_images = np.asarray(image_df["NumPy"].to_list())
preprocessed_data = preprocess_input(input_images)
dataset = tf.data.Dataset.from_tensor_slices(preprocessed_data)
dataset = dataset.batch(100)

# Change input shape
input_shape = (64, 64, 3)  # Assuming 3 channels for RGB images
new_input = Input(shape=input_shape)

# Load the pre-trained VGG16 model
# vgg16 = VGG16(weights="imagenet", input_tensor=new_input, include_top=False)
vgg16 = VGG16(
    weights="imagenet", input_tensor=new_input, include_top=False, pooling="avg"
)
flatten_layer = Flatten()(vgg16.output)
model = Model(inputs=vgg16.input, outputs=flatten_layer)

for layer in model.layers:
    layer.trainable = False

embeddings = model.predict(dataset)

print(embeddings)
print(embeddings.shape)
