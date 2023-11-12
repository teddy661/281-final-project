import argparse
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import psutil
from skimage.feature import canny, hog, local_binary_pattern
from skimage.filters import farid, gaussian

from create_feature_tools import (
    compute_hsv_histograms,
    compute_lbp_image_and_histogram,
    create_hog_features,
    normalize_histogram,
    restore_image_from_list,
)


def parallelize_dataframe(df, func, n_cores=4):
    """
    Enable parallel processing of a dataframe by splitting it by the number of cores
    and then recombining the results.
    """
    rows_per_dataframe = df.height // n_cores
    remainder = df.height % n_cores
    num_rows = [rows_per_dataframe] * (n_cores - 1)
    num_rows.append(rows_per_dataframe + remainder)
    start_pos = [0]
    for n in num_rows:
        start_pos.append(start_pos[-1] + n)
    df_split = []
    for start, rows in zip(start_pos, num_rows):
        df_split.append(df.slice(start, rows))
    pool = Pool(n_cores)
    new_df = pl.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return new_df


def compute_hsv_histograms_wapper(width: int, height: int, image: list) -> tuple:
    """
    Wrapper function for compute_hsv_histograms. Must return a list to avoid polars poor
    handling of numpy arrays
    """

    uint8_image = restore_image_from_list(width, height, image)
    (
        hue_hist,
        hue_edges,
        sat_hist,
        sat_edges,
        val_hist,
        val_edges,
    ) = compute_hsv_histograms(uint8_image)
    return (
        list(hue_hist),
        list(hue_edges),
        list(sat_hist),
        list(sat_edges),
        list(val_hist),
        list(val_edges),
    )


def compute_lbp_image_and_histogram_wrapper(
    width: int, height: int, image: list
) -> tuple:
    """
    Wrapper function for compute_lbp_image_and_histogram. Must return a list to avoid polars poor
    handling of numpy arrays
    """
    uint8_image = restore_image_from_list(width, height, image)
    lbp_image, lbp_hist, lbp_edges = compute_lbp_image_and_histogram(uint8_image)
    return list(lbp_image.ravel()), list(lbp_hist), list(lbp_edges)


def compute_hog_features_wrapper(width: int, height: int, image: list) -> tuple:
    """
    Wrapper function for compute_hog_features. Must return a list to avoid polars poor
    handling of numpy arrays
    """
    uint8_image = restore_image_from_list(width, height, image)
    hog_features, hog_image = create_hog_features(uint8_image)

    return list(hog_features), list(hog_image.ravel())


def hsv_parallel_wrapper(df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one is for the hsv histograms
    """
    df = df.with_columns(
        pl.struct(["Width", "Height", "Image"])
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
                    compute_hsv_histograms_wapper(
                        x["Width"],
                        x["Height"],
                        x["Image"],
                    ),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    df = df.drop("Hue_Edges", "Saturation_Edges", "Value_Edges")
    return df


def lbp_parallel_wrapper(df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one is for the lbp image and histogram
    """
    df = df.with_columns(
        pl.struct(["Width", "Height", "Image"])
        .map_elements(
            lambda x: dict(
                zip(
                    ("LBP_Image", "LBP_Hist", "LBP_Edges"),
                    compute_lbp_image_and_histogram_wrapper(
                        x["Width"],
                        x["Height"],
                        x["Image"],
                    ),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    df = df.drop("LBP_Edges")
    return df


def hog_parallel_wrapper(df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one is for the hog features
    """
    df = df.with_columns(
        pl.struct(["Width", "Height", "Image"])
        .map_elements(
            lambda x: dict(
                zip(
                    ("HOG_Features", "HOG_Image"),
                    compute_hog_features_wrapper(
                        x["Width"],
                        x["Height"],
                        x["Image"],
                    ),
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
            "Scaled_Image",
        ]
        rename_columns = {
            "Stretched_Histogram_Image": "Image",
            "Scaled_Resolution": "Resolution",
            "Scaled_Width": "Width",
            "Scaled_Height": "Height",
            "Scaled_Resolution": "Resolution",
        }
    elif target_image == source_image_columns[1]:
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
            "Stretched_Histogram_Image",
        ]
        rename_columns = {
            "Scaled_Image": "Image",
            "Scaled_Resolution": "Resolution",
            "Scaled_Width": "Width",
            "Scaled_Height": "Height",
            "Scaled_Resolution": "Resolution",
        }
    else:
        raise ValueError("Unknown target image")

    print("\tBegin Adjusting Columns", file=sys.stderr)
    df = df.drop(drop_columns)
    df = df.rename(rename_columns)
    print("\tEnd Adjusting Columns", file=sys.stderr)
    return df


def process_features(df: pl.DataFrame, num_cpus: int) -> pl.DataFrame:
    """
    Need to wrap everything so we can process multiple files
    """
    # HSV Histograms
    print(f"\tBegin Calculating HSV Histograms", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, hsv_parallel_wrapper, num_cpus)
    end_time = datetime.now()
    print(
        f"\tEnd Calculating HSV Histograms:\t\t{end_time - start_time}", file=sys.stderr
    )

    # LBP Image and Histogram
    print(f"\tBegin Calculating LBP Histograms", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, lbp_parallel_wrapper, num_cpus)
    end_time = datetime.now()
    print(
        f"\tEnd Calculating LBP Histograms:\t\t{end_time - start_time}", file=sys.stderr
    )

    # HOG Features
    print(f"\tBegin Calculating HOG Features", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, hog_parallel_wrapper, num_cpus)
    end_time = datetime.now()
    print(
        f"\tEnd Calculating HOG Features:\t\t{end_time - start_time}", file=sys.stderr
    )
    return df


def main():
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

    if num_cpus > 12 and args.num_cpus is None:
        print(f"Number of cpus might be too high: {num_cpus}", file=sys.stderr)
        print(f"Forcing to 12 cpus", file=sys.stderr)
        print(
            f"Re-Run and set number of cpus with -n option to override", file=sys.stderr
        )
        num_cpus = 12

    train_parquet = Path("Train.parquet")
    train_features_parquet = Path("train_features.parquet")

    test_parquet = Path("Test.parquet")
    test_features_parquet = Path("test_features.parquet")

    if not train_parquet.exists():
        print(f"FATAL: Training file is missing: {train_parquet}", file=sys.stderr)
        exit(1)

    if not test_parquet.exists():
        print(f"FATAL: Training file is missing: {test_parquet}", file=sys.stderr)
        exit(1)

    if (
        train_features_parquet.exists() or test_features_parquet.exists()
    ) and not args.force:
        print(
            f"FATAL: Features parquet files already exist. Use -f to force overwrite.",
            file=sys.stderr,
        )
        exit(1)

    if args.force:
        print(f"Force removing existing files.", file=sys.stderr)
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
    ## target_image = source_image_columns[1]
    source_image_columns = ["Stretched_Histogram_Image", "Scaled_image"]
    target_image = source_image_columns[0]  # 0 should be the default
    print(
        f"Using {target_image} as source image for feature generation", file=sys.stderr
    )

    print(f"Begin Reading Test Parquet", file=sys.stderr)
    start_time = datetime.now()
    test_feature_df = pl.read_parquet(train_parquet, use_pyarrow=True, memory_map=True)
    end_time = datetime.now()
    print(f"End Reading Test Parquet:\t\t\t{end_time-start_time}", file=sys.stderr)
    test_feature_df = adjust_columns(
        test_feature_df, target_image, source_image_columns
    )
    test_feature_df = process_features(test_feature_df, num_cpus)

    # Write the Test parquet file
    print(f"\tBegin Writing Test feature data", file=sys.stderr)
    start_time = datetime.now()
    test_feature_df.write_parquet(
        test_features_parquet,
        compression="zstd",
        compression_level=5,
        use_pyarrow=True,
    )
    end_time = datetime.now()
    print(
        f"\tEnd Writing Test feature data:\t\t{end_time - start_time}", file=sys.stderr
    )

    del test_feature_df  # Free up memory

    print(f"Begin Reading Training Parquet", file=sys.stderr)
    start_time = datetime.now()
    train_feature_df = pl.read_parquet(train_parquet, use_pyarrow=True, memory_map=True)
    end_time = datetime.now()
    print(f"End Reading Training Parquet:\t\t\t{end_time-start_time}", file=sys.stderr)
    train_feature_df = adjust_columns(
        train_feature_df, target_image, source_image_columns
    )
    train_feature_df = process_features(train_feature_df, num_cpus)

    # Write the Training parquet file
    print(f"\tBegin Writing Training feature data", file=sys.stderr)
    start_time = datetime.now()
    train_feature_df.write_parquet(
        train_features_parquet,
        compression="zstd",
        compression_level=5,
        use_pyarrow=True,
    )
    end_time = datetime.now()
    print(
        f"\tEnd Writing Training feature data:\t\t{end_time - start_time}",
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
