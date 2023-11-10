import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
import psutil
from PIL import Image
from skimage.transform import rescale, rotate


def restore_image_from_list(width: int, height: int, image: list) -> np.array:
    return np.array(image).reshape((height, width, 3)).astype(np.uint8)


def rescale_image(width: int, height: int, image: list, standard=64) -> tuple:
    """
    Rescale the image to a standard size. Median for our dataset is 35x35.
    Use order = 5 for (Bi-quintic) #Very slow Super high quality result.
    Settle on 64x64 for our standard size after discussion with professor.
    There will be some cropping of the image, but we'll center the crop.
    Returns an image with the pixel values normalized betwen 0-1
    """
    image = restore_image_from_list(width, height, image)
    scale = standard / min(image.shape[:2])
    image = rescale(image, scale, order=5, anti_aliasing=True, channel_axis=2)
    image = image[
        int(image.shape[0] / 2 - standard / 2) : int(image.shape[0] / 2 + standard / 2),
        int(image.shape[1] / 2 - standard / 2) : int(image.shape[1] / 2 + standard / 2),
        :,
    ]
    scaled_image_height = image.shape[0]
    scaled_image_width = image.shape[1]
    image = np.clip((image * 255.0), 0, 255).astype(
        np.uint8
    )  # put it back to uint8 after scaling
    return scaled_image_width, scaled_image_height, list(image.ravel())


def do_something(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.struct(["Scaled_Width", "Scaled_Height", "Scaled_Image"])
        .map_elements(
            lambda x: dict(
                zip(
                    ("NewWidth", "NewHeight", "NewImage2"),
                    rescale_image(
                        x["Scaled_Width"], x["Scaled_Height"], x["Scaled_Image"]
                    ),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    return df


def parallelize_dataframe(df, func, n_cores=4):
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
    new_df = pl.concat(df_split)
    pool = Pool(n_cores)
    df = pl.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return new_df


def main():
    script_start_time = start_time = datetime.now()

    try:
        __file__
    except NameError:
        __file__ = None
    if __file__ is not None:
        script_name = Path(__file__)
    else:
        script_name = Path("./try_parallel_polars.py")
    script_dir = script_name.parent
    test_parquet = script_dir.joinpath("Test.parquet")
    df = pl.read_parquet(test_parquet, use_pyarrow=True, memory_map=True)
    df2 = df.sample(1062, with_replacement=False)
    num_cpus = psutil.cpu_count(logical=False)
    num_cpus = 8

    result_dataframe = parallelize_dataframe(df2, do_something, 4)
    print(result_dataframe.columns)
    print(f"Result Shape: {result_dataframe.shape}")
    print(f"Original Shape: {df2.shape}")
    # check = (df2 == result_dataframe)
    # newcheck = (check == False).sum()
    # print((check == False).sum())
    # for col in check.columns:
    #    print(f'{col}: {newcheck[col]}')


if __name__ == "__main__":
    main()
