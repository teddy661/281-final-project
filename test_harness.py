import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path
import cv2
from io import BytesIO
from feature_utils import parallelize_dataframe, restore_image_from_list


def stretch_histogram_hsv_wrapper(df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one stretches the histogram of the image to the full range of 0-255
    """
    df = df.with_columns(
        pl.struct(["Scaled_Width", "Scaled_Height", "Scaled_Image"])
        .map_elements(
            lambda x: restore_image_from_list(
                x["Scaled_Width"], x["Scaled_Height"], x["Scaled_Image"]
            )
        )
        .alias("Unpacked_Image")
        .cast(pl.Binary)
    )
    return df


def do_convert(width: int, height, x: list):
    mem_file = BytesIO()
    img = restore_image_from_list(width, height, x)
    np.save(mem_file, img)
    return mem_file.getvalue()


def convert_to_bytes(df: pl.DataFrame) -> pl.DataFrame:
    mem_file = BytesIO()
    df = df.with_columns(
        pl.struct(["Scaled_Width", "Scaled_Height", "Scaled_Image"])
        .map_elements(
            lambda x: do_convert(
                x["Scaled_Width"], x["Scaled_Height"], x["Scaled_Image"]
            )
        )
        .alias("BytesIo")
        .cast(pl.Binary)
    )
    return df


def main():
    num_cpus = 8
    test_parquet = Path("data/test.parquet.lists")
    test_df = pl.read_parquet(test_parquet, memory_map=True)
    print(test_df.head())
    print(test_df.columns)
    print(80 * "-")
    test_df = parallelize_dataframe(test_df, convert_to_bytes, num_cpus)
    print(test_df.head())
    print(test_df.columns)

    print(type(test_df["BytesIo"][0]))
    bytes = BytesIO(test_df["BytesIo"][0])
    image = np.load(bytes)
    # sample = np.load(BytesIO(image))


if __name__ == "__main__":
    main()
