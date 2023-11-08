from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from skimage.feature import canny, hog, local_binary_pattern
from skimage.filters import farid, gaussian


def main():
    # Read the parquet file, this takes a while. Leave it here
    train_df = pl.read_parquet("Train.parquet", use_pyarrow=True, memory_map=True)

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

    train_df = train_df.drop(drop_columns)
    train_df = train_df.rename(rename_columns)


if __name__ == "__main__":
    main()
