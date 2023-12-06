###############################################################################
##
## This file will live outside the feature creation pipeline since we don't
## want to continually start and stop the model to predict against images.
## This file will read an existing features.parquet file and add the RESNET101
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
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.layers import Flatten, Input

tf.get_logger().setLevel("ERROR")


def process_features(input_df: pl.DataFrame) -> pl.DataFrame:
    image_df = input_df.select(["ImageNet_Scaled_Image"])
    image_df = image_df.with_columns(
        pl.col("ImageNet_Scaled_Image").map_elements(lambda x: np.load(BytesIO(x))).alias("NumPy")
    )
    input_images = np.asarray(image_df["NumPy"].to_list())
    input_images_tf = tf.convert_to_tensor(input_images)
    # resizing and cropping happened in the feature creation pipeline
    preprocessed_data = preprocess_input(input_images)  # Normalization step is in here.
    dataset = tf.data.Dataset.from_tensor_slices(preprocessed_data)
    dataset = dataset.batch(100)

    # Load the pre-trained ResNet-101 model
    resnet101 = ResNet101(weights="imagenet", include_top=False, pooling="avg")
    model = Model(inputs=resnet101.input, outputs=resnet101.output)

    for layer in model.layers:
        layer.trainable = False

    embeddings = model.predict(dataset)

    resnet101_embed_df = pl.DataFrame(
        {"RESNET101": [row.tolist() for row in embeddings]}
    )
    df_final = pl.concat([input_df, resnet101_embed_df], how="horizontal")
    return df_final


def preprocess_features(feature_file: Path) -> pl.DataFrame:
    print("Begin Reading Feature Parquet", file=sys.stderr)
    start_time = datetime.now()
    feature_df = pl.read_parquet(feature_file, memory_map=True)
    if "RESNET101" in feature_df.columns:
        print("Dropping Existing RESNET101 Column", file=sys.stderr)
        feature_df = feature_df.drop("RESNET101")
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

    print("Updating Test Feature file with RESNET101 Embeddings", file=sys.stderr)
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
        # use_pyarrow=True,
    )
    end_time = datetime.now()
    print(
        f"End Writing Test feature data:\t\t\t{end_time - start_time}",
        file=sys.stderr,
    )

    print("Updating Train Feature file with RESNET101 Embeddings", file=sys.stderr)
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
        # use_pyarrow=True,
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
