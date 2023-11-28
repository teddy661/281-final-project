###############################################################################
##
## This file will live outside the feature creation pipeline since we don't
## want to continually start and stop the model to predict against images.
## This file will read an existing features.parquet file and add the VGG16
## feature column to it.
##
## Batch load all images into numpy array
## then predict!
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import polars as pl
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Flatten, Input

tf.get_logger().setLevel("ERROR")


def process_features(input_df: pl.DataFrame) -> pl.DataFrame:
    image_df = input_df.select(["Image"])
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
    vgg16_embed_df = pl.DataFrame({"VGG16": [row.tolist() for row in embeddings]})
    df_final = pl.concat([input_df, vgg16_embed_df], how="horizontal")
    return df_final


def preprocess_features(feature_file: Path) -> pl.DataFrame:
    print("Begin Reading Feature Parquet", file=sys.stderr)
    start_time = datetime.now()
    feature_df = pl.read_parquet(feature_file, use_pyarrow=True, memory_map=True)
    if "VGG16" in feature_df.columns:
        print("Dropping Existing VGG16 Column", file=sys.stderr)
        feature_df = feature_df.drop("VGG16")
    end_time = datetime.now()
    print(f"End Reading Feature Parquet:\t\t\t{end_time-start_time}", file=sys.stderr)
    return feature_df


def main():
    # Read the parquet file, this takes a while. Leave it here
    train_features_file = Path("data/train_features.parquet")
    if not train_features_file.exists():
        print(
            "No train features file found. Please run the create_features_table first"
        )
        exit(1)

    test_features_file = Path("data/test_features.parquet")
    if not test_features_file.exists():
        print(
            "No train features file found. Please run the create_features_table first"
        )
        exit(1)

    print("Updating Test Feature file with VGG16 Embeddings", file=sys.stderr)
    script_start_time = datetime.now()
    test_feature_df = preprocess_features(test_features_file)
    print("Begin Calculating Features", file=sys.stderr)
    start_time = datetime.now()
    updated_test_df = process_features(test_feature_df)
    end_time = datetime.now()
    print(f"End Calculating Features:\t\t\t{end_time-start_time}", file=sys.stderr)
    print("Begin Writing Test feature data", file=sys.stderr)
    test_features_file.unlink()
    start_time = datetime.now()
    updated_test_df.write_parquet(
        test_features_file,
        compression="zstd",
        compression_level=5,
        use_pyarrow=True,
    )
    end_time = datetime.now()
    print(
        f"End Writing Test feature data:\t\t\t{end_time - start_time}",
        file=sys.stderr,
    )

    print("Updating Train Feature file with VGG16 Embeddings", file=sys.stderr)
    script_start_time = datetime.now()
    train_feature_df = preprocess_features(train_features_file)
    print("Begin Calculating Features", file=sys.stderr)
    start_time = datetime.now()
    updated_train_df = process_features(train_feature_df)
    end_time = datetime.now()
    print(f"End Calculating Features:\t\t\t{end_time-start_time}", file=sys.stderr)
    print("Begin Writing Train feature data", file=sys.stderr)
    train_features_file.unlink()
    start_time = datetime.now()
    updated_train_df.write_parquet(
        train_features_file,
        compression="zstd",
        compression_level=5,
        use_pyarrow=True,
    )
    end_time = datetime.now()
    print(
        f"End Writing Train feature data:\t\t\t{end_time - start_time}",
        file=sys.stderr,
    )

    script_end_time = datetime.now()
    print(80 * "=", file=sys.stderr)
    print(
        f"Total Elapsed Time:\t\t\t\t{script_end_time - script_start_time}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
