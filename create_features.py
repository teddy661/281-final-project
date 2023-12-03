import argparse
import multiprocessing as mp
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import psutil

from feature_utils import compute_hsv_histograms  # TemplateMatching,
from feature_utils import (
    compute_lbp_image_and_histogram,
    compute_sift,
    convert_numpy_to_bytesio,
    create_hog_features,
    parallelize_dataframe,
)


def compute_hsv_histograms_wapper(image: bytes) -> tuple:
    """
    Wrapper function for compute_hsv_histograms. Must return a list to avoid polars poor
    handling of numpy arrays
    """

    float32_image = np.load(BytesIO(image))
    (
        hue_hist,
        hue_edges,
        sat_hist,
        sat_edges,
        val_hist,
        val_edges,
    ) = compute_hsv_histograms(float32_image)
    return (
        list(hue_hist),
        list(hue_edges),
        list(sat_hist),
        list(sat_edges),
        list(val_hist),
        list(val_edges),
    )


def compute_lbp_image_and_histogram_wrapper(image: bytes) -> tuple:
    """
    Wrapper function for compute_lbp_image_and_histogram. Must return a list to avoid polars poor
    handling of numpy arrays
    """
    float32_image = np.load(BytesIO(image))
    lbp_image, lbp_hist, lbp_edges = compute_lbp_image_and_histogram(float32_image)
    return convert_numpy_to_bytesio(lbp_image), list(lbp_hist), list(lbp_edges)


def compute_sift_features_wrapper(image: bytes) -> tuple:
    """
    Wrapper function for compute_sift_features. Must return a list to avoid polars poor
    handling of numpy arrays
    """
    float32_image = np.load(BytesIO(image))
    sift_features = compute_sift(float32_image)
    if sift_features is None:
        # There are images where sift doesn't work. Return an empty feature vector
        sift_features = np.zeros((1, 128), dtype=np.float32)  # Empty feature vector
    return convert_numpy_to_bytesio(sift_features)


def compute_hog_features_wrapper(image: bytes) -> tuple:
    """
    Wrapper function for compute_hog_features. Must return a list to avoid polars poor
    handling of numpy arrays
    """
    float32_image = np.load(BytesIO(image))
    hog_features, hog_image = create_hog_features(float32_image)

    return list(hog_features), convert_numpy_to_bytesio(hog_image)


def hsv_parallel_wrapper(df: pl.DataFrame, meta_df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one is for the hsv histograms
    """
    df = df.with_columns(
        pl.col("Image")
        .map_elements(
            lambda x: dict(
                zip(
                    (
                        "Hue_Hist",
                        "Hue_Edges",
                        "Saturation_Hist",
                        "Saturation_Edges",
                        "Value_Hist",
                        "Value_Edges",
                    ),
                    compute_hsv_histograms_wapper(x),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    df = df.drop("Hue_Edges", "Saturation_Edges", "Value_Edges")
    return df


def lbp_parallel_wrapper(df: pl.DataFrame, meta_df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one is for the lbp image and histogram
    """
    df = df.with_columns(
        pl.col("Image")
        .map_elements(
            lambda x: dict(
                zip(
                    ("LBP_Image", "LBP_Hist", "LBP_Edges"),
                    compute_lbp_image_and_histogram_wrapper(x),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    df = df.drop("LBP_Edges")
    return df


def sift_parallel_wrapper(df: pl.DataFrame, meta_df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one is for the sift features
    """
    df = df.with_columns(
        pl.col("Image")
        .map_elements(
            lambda x: compute_sift_features_wrapper(x),
        )
        .alias("SIFT_Features")
    )
    return df


def hog_parallel_wrapper(df: pl.DataFrame, meta_df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one is for the hog features
    """
    df = df.with_columns(
        pl.col("Image")
        .map_elements(
            lambda x: dict(
                zip(
                    ("HOG_Features", "HOG_Image"),
                    compute_hog_features_wrapper(x),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    # df = df.drop("HOG_Image") # Keep the hog image for debugging
    return df


def adjust_columns(
    df: pl.DataFrame, target_image: str, source_image_columns: list
) -> pl.DataFrame:
    """
    The feature dataset will always have the columns "Width", "Height", "Image", "Resolution", "ClassId"
    which is the basis for feature generation.
    """
    if target_image == source_image_columns[0]:
        drop_columns = [
            "Width",
            "Height",
            "Roi.X1",
            "Roi.Y1",
            "Roi.X2",
            "Roi.Y2",
            "Path",
            "Resolution",
            "Image",
            "Cropped_Width",
            "Cropped_Height",
            "Cropped_Image",
            "Cropped_Resolution",
        ]
        rename_columns = {
            "Scaled_Image": "Image",
            "Scaled_Resolution": "Resolution",
            "Scaled_Width": "Width",
            "Scaled_Height": "Height",
        }
    else:
        raise ValueError("Unknown target image")

    print("\tBegin Adjusting Columns", file=sys.stderr)
    df = df.drop(drop_columns)
    df = df.rename(rename_columns)
    print("\tEnd Adjusting Columns", file=sys.stderr)
    return df


def compute_template_wrapper(image_bytes: bytes, meta_df: pl.DataFrame) -> np.array:
    """
    Wrapper function for compute_template. Must return a list to avoid polars poor
    handling of numpy arrays
    """
    # Pad the image to make sure it's larger than the template
    pad_rows = 4
    pad_cols = 4

    uint8_image = (np.load(BytesIO(image_bytes)) * 255.0).astype(np.uint8)
    gray_image = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2GRAY)
    gray_image = np.pad(
        gray_image,
        ((pad_rows, pad_rows), (pad_cols, pad_cols)),
        mode="constant",
        constant_values=0,
    )

    i = 0
    for row in meta_df.iter_rows(named=True):
        template_bytes = row["Scaled_Meta_Image"]
        uint8_template = (np.load(BytesIO(template_bytes)) * 255.0).astype(np.uint8)
        gray_template = cv2.cvtColor(uint8_template, cv2.COLOR_RGB2GRAY)
        template_match = cv2.matchTemplate(
            gray_image, gray_template, cv2.TM_CCOEFF_NORMED
        )
        if i == 0:
            final_array = (
                template_match.ravel()
            )  # Flatten the array These are all 9x9 due to scaling and padding
        else:
            final_array = np.concatenate((final_array, template_match.ravel()))
        i += 1
    return list(final_array)


def template_parallel_wrapper(df: pl.DataFrame, meta_df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one is for the template
    """
    df = df.with_columns(
        pl.struct(["Image", "Scaled_Meta_Image"])
        .map_elements(
            lambda x: compute_template_wrapper(x["Image"], meta_df),
        )
        .alias("Template_Pattern")
    )
    return df


def process_features(
    df: pl.DataFrame, meta_df: pl.DataFrame, num_cpus: int
) -> pl.DataFrame:
    """
    Need to wrap everything so we can process multiple files
    """

    # HSV Histograms
    print("\tBegin Calculating HSV Histograms", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, meta_df, hsv_parallel_wrapper, num_cpus)
    end_time = datetime.now()
    print(
        f"\tEnd Calculating HSV Histograms:\t\t{end_time - start_time}",
        file=sys.stderr,
    )

    # LBP Image and Histogram
    print("\tBegin Calculating LBP Histograms", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, meta_df, lbp_parallel_wrapper, num_cpus)
    end_time = datetime.now()
    print(
        f"\tEnd Calculating LBP Histograms:\t\t{end_time - start_time}",
        file=sys.stderr,
    )

    # HOG Features
    print("\tBegin Calculating HOG Features", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, meta_df, hog_parallel_wrapper, num_cpus)
    end_time = datetime.now()
    print(
        f"\tEnd Calculating HOG Features:\t\t{end_time - start_time}",
        file=sys.stderr,
    )

    # SIFT Features - Disable SIFT for now
    # print("\tBegin Calculating SIFT Features", file=sys.stderr)
    # start_time = datetime.now()
    # df = parallelize_dataframe(df, sift_parallel_wrapper, num_cpus)
    # end_time = datetime.now()
    # print(
    #     f"\tEnd Calculating SIFT Features:\t\t{end_time - start_time}",
    #     file=sys.stderr,
    # )

    # Template Feature
    print("\tBegin Calculating Template Features", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, meta_df, template_parallel_wrapper, num_cpus)
    end_time = datetime.now()
    print(
        f"\tEnd Calculating Template Features:\t{end_time - start_time}",
        file=sys.stderr,
    )

    return df


def main():
    try:
        __file__
    except NameError:
        __file__ = None
    if __file__ is not None:
        script_name = Path(__file__)
    else:
        script_name = Path("./process_raw_dataset.py")
    script_dir = script_name.parent

    parser = argparse.ArgumentParser(
        description="Create Feature Tables from Training Data"
    )
    parser.add_argument(
        "-n",
        dest="num_cpus",
        help="number of cpus to use for parallel processing",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-f",
        dest="force",
        help="force overwrite of existing parquet files",
        action="store_true",
    )
    args = parser.parse_args()
    prog_name = parser.prog
    print(80 * "=", file=sys.stderr)
    script_start_time = datetime.now()
    if args.num_cpus is not None:
        num_cpus = args.num_cpus
    else:
        num_cpus = psutil.cpu_count(logical=False)

    if num_cpus > 8 and args.num_cpus is None:
        print(f"Number of cpus might be too high: {num_cpus}", file=sys.stderr)
        print("Forcing to 8 cpus", file=sys.stderr)
        print(
            "Re-Run and set number of cpus with -n option to override", file=sys.stderr
        )
        num_cpus = 8

    train_parquet = script_dir.joinpath("data/train.parquet")
    train_features_parquet = script_dir.joinpath("data/train_features.parquet")

    test_parquet = script_dir.joinpath("data/test.parquet")
    test_features_parquet = script_dir.joinpath("data/test_features.parquet")

    meta_parquet = script_dir.joinpath("data/meta_full.parquet")

    if not train_parquet.exists():
        print(f"FATAL: Training file is missing: {train_parquet}", file=sys.stderr)
        exit(1)

    if not test_parquet.exists():
        print(f"FATAL: Training file is missing: {test_parquet}", file=sys.stderr)
        exit(1)

    if not meta_parquet.exists():
        print(f"FATAL: Meta file is missing: {meta_parquet}", file=sys.stderr)
        exit(1)

    if (
        train_features_parquet.exists() or test_features_parquet.exists()
    ) and not args.force:
        print(
            "FATAL: Features parquet files already exist. Use -f to force overwrite.",
            file=sys.stderr,
        )
        exit(1)

    if args.force:
        print("Force removing existing files.", file=sys.stderr)
        train_features_parquet.unlink(missing_ok=True)
        test_features_parquet.unlink(missing_ok=True)

    print(f"Multiprocessing on {num_cpus} CPUs", file=sys.stderr)

    ############################################################################
    ## Make sure to select the correct image for feature generation
    ##
    ## This step is manually done here by editing this file. There is no
    ## command line option to select the image. "Stretched_Histogram_Image" is
    ## the default. If you want to use "Scaled_Image" then change the line below
    ## to:
    ## target_image = source_image_columns[0]
    source_image_columns = ["Scaled_image"]
    target_image = source_image_columns[0]  # 0 should be the default
    print(
        f"Using {target_image} as source image for feature generation", file=sys.stderr
    )

    meta_df = pl.read_parquet(meta_parquet, memory_map=True)
    meta_image_df = meta_df.select(["ClassId", "Scaled_Meta_Image"]).sort(by="ClassId")
    del meta_df

    print("Begin Reading Test Parquet", file=sys.stderr)
    start_time = datetime.now()
    test_feature_df = pl.read_parquet(test_parquet, memory_map=True)
    end_time = datetime.now()
    print(f"End Reading Test Parquet:\t\t\t{end_time-start_time}", file=sys.stderr)
    test_feature_df = adjust_columns(
        test_feature_df, target_image, source_image_columns
    )
    test_feature_df = test_feature_df.join(meta_image_df, on="ClassId")

    ############################################################################
    ##
    ## DataFrames need to be chunked out since LBP takes so much memory.
    ##
    num_rows_per_partition = test_feature_df.height + 1
    print(
        "Processing Test DataFrame in chunks of: ",
        num_rows_per_partition,
        file=sys.stderr,
    )
    test_feature_df_list = (
        test_feature_df.with_row_count("id")
        .with_columns(
            pl.col("id").map_elements(lambda i: int(i / num_rows_per_partition))
        )
        .partition_by("id")
    )

    del test_feature_df

    i = 0
    num_partitions = len(test_feature_df_list)
    while test_feature_df_list:
        print(
            f"\tProcessing Test partition {i+1:2d} of {num_partitions:2d}",
            file=sys.stderr,
        )
        dfp = process_features(test_feature_df_list.pop(0), meta_image_df, num_cpus)
        if i == 0:
            test_feature_df = dfp.drop(["id', 'Scaled_Meta_Image"])
        else:
            test_feature_df = pl.concat(
                [test_feature_df, dfp.drop(["id', 'Scaled_Meta_Image"])]
            )
        i += 1

    # Write the Test parquet file
    print("\tBegin Writing Test feature data", file=sys.stderr)
    start_time = datetime.now()
    test_feature_df.write_parquet(
        test_features_parquet,
        compression="zstd",
        compression_level=5,
        # use_pyarrow=True,
    )
    end_time = datetime.now()
    print(
        f"\tEnd Writing Test feature data:\t\t{end_time - start_time}", file=sys.stderr
    )

    del test_feature_df, test_feature_df_list  # Free up memory

    print("Begin Reading Training Parquet", file=sys.stderr)
    start_time = datetime.now()
    train_feature_df = pl.read_parquet(train_parquet, memory_map=True)
    end_time = datetime.now()
    print(f"End Reading Training Parquet:\t\t\t{end_time-start_time}", file=sys.stderr)
    train_feature_df = adjust_columns(
        train_feature_df, target_image, source_image_columns
    )
    train_feature_df = train_feature_df.join(meta_image_df, on="ClassId")

    ############################################################################
    ##
    ## DataFrames need to be chunked out since LBP takes so much memory.
    ##
    num_rows_per_partition = train_feature_df.height + 1
    print(
        "Processing Train DataFrame in chunks of: ",
        num_rows_per_partition,
        file=sys.stderr,
    )
    train_feature_df_list = (
        train_feature_df.with_row_count("id")
        .with_columns(
            pl.col("id").map_elements(lambda i: int(i / num_rows_per_partition))
        )
        .partition_by("id")
    )

    del train_feature_df  # Free up memory

    i = 0
    num_partitions = len(train_feature_df_list)
    while train_feature_df_list:
        print(
            f"\tProcessing Train partition {i+1:2d} of {num_partitions:2d}",
            file=sys.stderr,
        )
        dfp = process_features(train_feature_df_list.pop(0), meta_image_df, num_cpus)
        if i == 0:
            train_feature_df = dfp.drop(["id', 'Scaled_Meta_Image"])
        else:
            train_feature_df = pl.concat(
                [train_feature_df, dfp.drop(["id', 'Scaled_Meta_Image"])]
            )
        i += 1

    # Write the Training parquet file
    print("\tBegin Writing Training feature data", file=sys.stderr)
    start_time = datetime.now()
    train_feature_df.write_parquet(
        train_features_parquet,
        compression="zstd",
        compression_level=5,
        # use_pyarrow=True,
    )
    end_time = datetime.now()
    print(
        f"\tEnd Writing Training feature data:\t{end_time - start_time}",
        file=sys.stderr,
    )

    script_end_time = datetime.now()
    print(80 * "=", file=sys.stderr)
    print(
        f"Total Elapsed Time:\t\t\t\t{script_end_time - script_start_time}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
