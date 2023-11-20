import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path
import cv2
from io import BytesIO


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


def restore_image_from_list(
    width: int, height: int, image: list, num_channels: int = 3
) -> np.array:
    """
    Images are stored as lists in the parquet file. This function creates a numpy,
    array reshapes it to the correct size using the width and height of the image.
    Expects the original image to be a 3 channel image. We ensure the images
    are always written to disk as float32. This function reshapes it back to an image
    the dtype is not altered.

    There's something really odd in here. When I run this in ipython everything works
    when I run the script it always returns a float64. We'll just force it to float32
    """
    return np.asarray(image, dtype=np.float32).reshape((height, width, num_channels))


def restore_image_from_list(
    width: int, height: int, image: list, num_channels: int = 3
) -> np.array:
    """
    Images are stored as lists in the parquet file. This function creates a numpy,
    array reshapes it to the correct size using the width and height of the image.
    Expects the original image to be a 3 channel image. We ensure the images
    are always written to disk as float32. This function reshapes it back to an image
    the dtype is not altered.

    There's something really odd in here. When I run this in ipython everything works
    when I run the script it always returns a float64. We'll just force it to float32
    """
    return np.asarray(image, dtype=np.float32).reshape((height, width, num_channels))


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


def do_convert(width, height, x):
    mem_file = BytesIO()
    img = restore_image_from_list(width, height, x)
    np.save(mem_file, img)
    return mem_file.getvalue()


def convert_to_bytes(df):
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
    test_parquet = Path("data/test.parquet")
    test_df = pl.read_parquet(test_parquet, use_pyarrow=True, memory_map=True)
    print(test_df.head())
    print(test_df.columns)
    print(80 * "-")
    test_df = parallelize_dataframe(test_df, convert_to_bytes, num_cpus)
    print(test_df.head())
    print(test_df.columns)

    print(type(test_df["BytesIo"][0]))
    bytes = BytesIO(test_df["BytesIo"][0])
    image = np.load(bytes)
    print(image.shape)


if __name__ == "__main__":
    main()
