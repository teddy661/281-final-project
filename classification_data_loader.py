import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl

TRAIN_FEATURES_FILE = Path("data/train_features.parquet")
TEST_FEATURES_FILE = Path("data/test_features.parquet")
META_FILE = Path("data/meta_full.parquet")

SAMPLES_PER_CLASS = 210
RANDOM_SEED = 42
TRAIN_PERCENTAGE = 0.8

pl.set_random_seed(RANDOM_SEED)


def load_raw_dataframes() -> (pl.DataFrame, pl.DataFrame, pl.DataFrame):
    """
    Loads the raw dataframes from the parquet files
    :return: raw_train_df, raw_test_df, raw_meta_df
    """
    if TRAIN_FEATURES_FILE.exists:
        raw_train_df = pl.read_parquet(TRAIN_FEATURES_FILE, memory_map=True)
    else:
        print(f"Train Features File is missing {TRAIN_FEATURES_FILE}", file=sys.stderr)
        exit()

    if TEST_FEATURES_FILE.exists:
        raw_test_df = pl.read_parquet(TEST_FEATURES_FILE, memory_map=True)
    else:
        print(f"Test Features File is missing {TEST_FEATURES_FILE}", file=sys.stderr)
        exit()

    if META_FILE.exists:
        raw_meta_df = pl.read_parquet(META_FILE, memory_map=True)
    else:
        print(f"Meta File is missing {META_FILE}", file=sys.stderr)
        exit()

    return raw_train_df, raw_test_df, raw_meta_df


def get_sampled_data(raw_train_df: pl.DataFrame) -> pl.DataFrame:
    """
    Samples the dataframe to get the same number of samples per class our smallest class
    has 210 examples so we'll use that as our sample size. We'll also shuffle the dataframe
    :param df: The dataframe to sample
    :return: The sampled dataframe
    """
    pl.set_random_seed(RANDOM_SEED)
    train_equal_sample_df = pl.concat(
        [
            x.sample(SAMPLES_PER_CLASS, with_replacement=False, shuffle=False)
            for x in raw_train_df.partition_by("ClassId")
        ]
    )
    return train_equal_sample_df.sample(shuffle=True, fraction=1)


def split_data(df: pl.DataFrame) -> (pl.DataFrame, pl.DataFrame):
    """
    Splits the dataframe into a train and validation set
    :param df: The dataframe to split
    :return: The train and validation dataframes
    """
    num_train_rows = int(df.height * TRAIN_PERCENTAGE)

    train_df = df.slice(0, num_train_rows)
    validation_df = df.slice(num_train_rows, df.height)
    return train_df, validation_df


def get_lbp_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame, validation_df: pl.DataFrame
) -> (np.array, np.array, np.array):
    numpy_stage1_train = train_df.with_columns(
        pl.col("LBP_Image").map_elements(lambda x: np.load(BytesIO(x))).alias("NumPy")
    )
    X_train_LBP = np.asarray(numpy_stage1_train["NumPy"].to_list())
    del numpy_stage1_train
    X_train_LBP = np.reshape(X_train_LBP, (X_train_LBP.shape[0], -1)).copy()

    numpy_stage1_test = test_df.with_columns(
        pl.col("LBP_Image").map_elements(lambda x: np.load(BytesIO(x))).alias("NumPy")
    )
    X_test_LBP = np.asarray(numpy_stage1_test["NumPy"].to_list())
    del numpy_stage1_test
    X_test_LBP = np.reshape(X_test_LBP, (X_test_LBP.shape[0], -1)).copy()

    numpy_stage1_validation = validation_df.with_columns(
        pl.col("LBP_Image").map_elements(lambda x: np.load(BytesIO(x))).alias("NumPy")
    )
    X_validation_LBP = np.asarray(numpy_stage1_validation["NumPy"].to_list())
    del numpy_stage1_validation
    X_validation_LBP = np.reshape(
        X_validation_LBP,
        (X_validation_LBP.shape[0], -1),
    ).copy()
    return X_train_LBP, X_test_LBP, X_validation_LBP


def get_hog_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame, validation_df: pl.DataFrame
) -> (np.array, np.array, np.array):
    X_train_HOG = np.asarray(train_df["HOG_Features"].to_list())
    X_test_HOG = np.asarray(test_df["HOG_Features"].to_list())
    X_validation_HOG = np.asarray(validation_df["HOG_Features"].to_list())
    return X_train_HOG, X_test_HOG, X_validation_HOG


def get_hue_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame, validation_df: pl.DataFrame
) -> (np.array, np.array, np.array):
    X_train_Hue = np.asarray(train_df["Hue_Hist"].to_list())
    X_test_Hue = np.asarray(test_df["Hue_Hist"].to_list())
    X_validation_Hue = np.asarray(validation_df["Hue_Hist"].to_list())

    return X_train_Hue, X_test_Hue, X_validation_Hue


def get_saturation_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame, validation_df: pl.DataFrame
) -> (np.array, np.array, np.array):
    X_train_Saturation = np.asarray(train_df["Saturation_Hist"].to_list())
    X_test_Saturation = np.asarray(test_df["Saturation_Hist"].to_list())
    X_validation_Saturation = np.asarray(validation_df["Saturation_Hist"].to_list())

    return X_train_Saturation, X_test_Saturation, X_validation_Saturation


def get_value_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame, validation_df: pl.DataFrame
) -> (np.array, np.array, np.array):
    X_train_Value = np.asarray(train_df["Value_Hist"].to_list())
    X_test_Value = np.asarray(test_df["Value_Hist"].to_list())
    X_validation_Value = np.asarray(validation_df["Value_Hist"].to_list())

    return X_train_Value, X_test_Value, X_validation_Value


def get_template_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame, validation_df: pl.DataFrame
) -> (np.array, np.array, np.array):
    X_train_Template = np.asarray(train_df["Template_Pattern"].to_list())
    X_test_Template = np.asarray(test_df["Template_Pattern"].to_list())
    X_validation_Template = np.asarray(validation_df["Template_Pattern"].to_list())

    return X_train_Template, X_test_Template, X_validation_Template


def get_hog_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame, validation_df: pl.DataFrame
) -> (np.array, np.array, np.array):
    X_train_HOG = np.asarray(train_df["HOG_Features"].to_list())
    X_test_HOG = np.asarray(test_df["HOG_Features"].to_list())
    X_validation_HOG = np.asarray(validation_df["HOG_Features"].to_list())

    return X_train_HOG, X_test_HOG, X_validation_HOG


def get_resnet101_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame, validation_df: pl.DataFrame
) -> (np.array, np.array, np.array):
    X_train_Resnet101 = np.asarray(train_df["RESNET101"].to_list())
    X_test_Resnet101 = np.asarray(test_df["RESNET101"].to_list())
    X_validation_Resnet101 = np.asarray(validation_df["RESNET101"].to_list())

    return X_train_Resnet101, X_test_Resnet101, X_validation_Resnet101


def get_vgg16_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame, validation_df: pl.DataFrame
) -> (np.array, np.array, np.array):
    X_train_VGG16 = np.asarray(train_df["VGG16"].to_list())
    X_test_VGG16 = np.asarray(test_df["VGG16"].to_list())
    X_validation_VGG16 = np.asarray(validation_df["VGG16"].to_list())

    return X_train_VGG16, X_test_VGG16, X_validation_VGG16
