import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from skimage.feature import canny, hog, local_binary_pattern
from skimage.filters import farid, gaussian

from create_feature_tools import (
    compute_hsv_histograms,
    compute_lbp_image_and_histogram,
    create_hog_features,
    normalize_histogram,
    restore_image_from_list,
)


def main():
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
    df = df.sample(100)
    print(f"\tBegin Calulating HSV Histograms", file=sys.stderr)
    start_time = datetime.now()
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
                    compute_hsv_histograms(
                        restore_image_from_list(x["Width"], x["Height"], x["Image"])
                    ),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    end_time = datetime.now()
    print(f"\End Calulating HSV Histograms:\t{end_time - start_time}", file=sys.stderr)
    print(df.head())
    print(df.columns)


if __name__ == "__main__":
    main()
