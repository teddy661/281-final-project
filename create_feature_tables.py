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
    df = df.drop("HOG_Image")
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
    args = parser.parse_args()
    prog_name = parser.prog
    print(60 * "=", file=sys.stderr)
    script_start_time = datetime.now()
    if args.num_cpus is not None:
        num_cpus = args.num_cpus
    else:
        num_cpus = psutil.cpu_count(logical=False)

    if num_cpus > 12 and args.num_cpus is None:
        print(f"Number of cpus might be too high: {num_cpus}")
        print(f"Forcing to 12 cpus")
        print(f"Re-Run and set number of cpus with -n option to override")
        num_cpus = 12

    print(f"Multiprocessing on {num_cpus} CPUs")
    # Read the parquet file, this takes a while. Leave it here
    print(f"Begin Reading Parquet", file=sys.stderr)
    df = pl.read_parquet("Train.parquet", use_pyarrow=True, memory_map=True)
    print("End Reading Parquet", file=sys.stderr)

    source_image_columns = ["Stretched_Histogram_Image", "Scaled_image"]
    target_image = source_image_columns[0]  # 0 should be the default
    print(f"Using {target_image} as source image for feature generation")

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

    # The dataset used for training will always have the columns
    # "Width", "Height", "Image", "Resolution", "ClassId"
    # which is the basis for feature generation
    print("Begin Adusting Columns")
    df = df.drop(drop_columns)
    df = df.rename(rename_columns)
    print("End Adusting Columns")
    # df = df.sample(10, with_replacement=False)  # debugging

    # HSV Histograms
    print(f"Begin Calulating HSV Histograms", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, hsv_parallel_wrapper, num_cpu)
    end_time = datetime.now()
    print(f"End Calulating HSV Histograms:\t{end_time - start_time}", file=sys.stderr)

    # LBP Image and Histogram
    print(f"Begin Calulating LBP Histograms", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, lbp_parallel_wrapper, num_cpu)
    end_time = datetime.now()
    print(f"End Calulating LBP Histograms:\t{end_time - start_time}", file=sys.stderr)

    # HOG Features
    print(f"Begin Calulating HOG Features", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, hog_parallel_wrapper, num_cpu)
    end_time = datetime.now()
    print(f"End Calulating HOG Features:\t{end_time - start_time}", file=sys.stderr)

    # Write the parquet file
    print(f"Begin Writing feature data to parquet file.", file=sys.stderr)
    df.write_parquet(
        "Features.parquet",
        compression="zstd",
        compression_level=5,
        use_pyarrow=True,
    )
    print(f"End Writing feature data to parquet file.", file=sys.stderr)
    script_end_time = datetime.now()
    print(60 * "=", file=sys.stderr)
    print(f"Script Duration:\t\t{script_end_time - script_start_time}", file=sys.stderr)


if __name__ == "__main__":
    main()
