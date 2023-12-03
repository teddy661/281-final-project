import argparse
import multiprocessing as mp
import sys
from datetime import datetime
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import psutil
from PIL import Image
from skimage.transform import rescale, rotate

from feature_utils import convert_numpy_to_bytesio, parallelize_dataframe


def prelim_validate_dataset_dir(root_dir: Path) -> bool:
    prelim_ready = True
    files = ["Meta_full.csv", "Meta.csv", "Test.csv", "Train.csv"]
    dirs = ["Meta", "Test", "Train"]
    for c_file in files:
        if not root_dir.joinpath(c_file).exists:
            print(f"Required File is Missing: {c_file}", file=sys.stderr)
            prelim_ready = False
    for c_dir in dirs:
        if not root_dir.joinpath(c_dir).exists:
            print(f"Required Directory is Missing: {c_dir}", file=sys.stderr)
            prelim_ready = False
        elif c_dir == "Train":
            for d in range(43):
                if not root_dir.joinpath(c_dir).joinpath(str(d)).exists:
                    print("Required Directory is Missing: {d}", file=sys.stderr)
                    prelim_ready = False
    return prelim_ready


def update_path(path: Path, root_dir: Path) -> Path:
    return str(root_dir.joinpath(path).resolve())


def crop_to_roi(
    width: int, height: int, y1: int, y2: int, x1: int, x2: int, image: bytes
) -> tuple:
    """
    Crops the image to the Roi values provided in the dataset.
    Does not alter the image dtype
    """
    image = np.load(BytesIO(image))
    cropped_image = image[y1 : y2 + 1, x1 : x2 + 1, :]
    cropped_image_height = cropped_image.shape[0]
    cropped_image_width = cropped_image.shape[1]
    return (
        cropped_image_width,
        cropped_image_height,
        convert_numpy_to_bytesio(cropped_image),
    )


def rescale_image(image: bytes, standard: int = 64) -> tuple:
    """
    Rescale the image to a standard size. Median for our dataset is 35x35.
    Use order = 5 for (Bi-quintic) #Very slow Super high quality result.
    Settle on 64x64 for our standard size after discussion with professor.
    There will be some cropping of the image, but we'll center the crop.
    This function will take an input image for type uint8 or float(64,32)
    and always return an image of dtype float64 which we truncate back to float32
    which is our standard image format due to cvtColor limitations
    """
    image = np.load(BytesIO(image))
    scale = standard / min(image.shape[:2])
    image = rescale(image, scale, order=5, anti_aliasing=True, channel_axis=2)
    image = image[
        int(image.shape[0] / 2 - standard / 2) : int(image.shape[0] / 2 + standard / 2),
        int(image.shape[1] / 2 - standard / 2) : int(image.shape[1] / 2 + standard / 2),
        :,
    ]
    scaled_image_height = image.shape[0]
    scaled_image_width = image.shape[1]
    return (
        scaled_image_width,
        scaled_image_height,
        convert_numpy_to_bytesio(image),
    )


def stretch_histogram(image: bytes) -> list:
    """
    Input a float32 image
    Stretch the histogram of the image to the full range of 0-100
    For our data this appears to work much better than equalizing the histogram
    We are only stretching the L channel of the LAB color space. This should
    preserve the possible the color information in the image. The highest precision
    supported by cvtColor. Abandon this in favor of HSV
    """
    restored_image = np.load(BytesIO(image))
    restored_uint8 = (restored_image * 255.0).astype(np.uint8)
    lab_image = cv2.cvtColor(restored_uint8, cv2.COLOR_RGB2LAB)
    l_channel = lab_image[:, :, 0]
    min_val = np.min(l_channel)
    max_val = np.max(l_channel)
    new_min = 0
    new_max = 100
    # stretched_l_channel = ((l_channel - min_val) / (max_val - min_val)) * (
    #    new_max - new_min
    # ) + new_min
    stretched_l_channel = np.interp(
        l_channel, (min_val, max_val), (new_min, new_max)
    ).astype(lab_image.dtype)
    stretched_lab_image = np.stack(
        (stretched_l_channel, lab_image[:, :, 1], lab_image[:, :, 2]), axis=2
    )
    rgb_image = cv2.cvtColor(stretched_lab_image, cv2.COLOR_LAB2RGB)
    # convert back to float32 to store it to disk
    return list((rgb_image.astype(np.float32) / 255.0).ravel())


def stretch_histogram_hsv(image: bytes) -> list:
    """
    Input a float32 image
    Stretch the histogram of the image to the full range of 0-1
    For our data this appears to work much better than equalizing the histogram
    We are only stretching the V channel of the HSV color space. This should
    preserve the possible the color information in the image. The highest precision
    supported by cvtColor. Prefer this over LAB stretching for the image data
    we have
    """
    restored_image = np.load(BytesIO(image))
    restored_uint8 = (restored_image * 255.0).astype(np.uint8)
    hsv_image = cv2.cvtColor(restored_uint8, cv2.COLOR_RGB2HSV)
    brightness = hsv_image[:, :, 2]
    min_val = np.min(brightness)
    max_val = np.max(brightness)
    new_min = 0
    new_max = 255
    stretched_v_channel = np.interp(
        brightness, (min_val, max_val), (new_min, new_max)
    ).astype(brightness.dtype)
    stretched_hsv_image = np.stack(
        (hsv_image[:, :, 0], hsv_image[:, :, 1], stretched_v_channel), axis=2
    )
    new_rgb_image = cv2.cvtColor(stretched_hsv_image, cv2.COLOR_HSV2RGB)
    return list((new_rgb_image.astype(np.float32) / 255.0).ravel())


def apply_clahe(image: bytes) -> list:
    """
    Input a float32 image
    see if better results with CLAHE
    CLAHE introduces a lot of noise in the image even though
    the contrast is better. Not using it for now
    """
    restored_image = sample = np.load(BytesIO(image))
    restored_uint8 = (restored_image * 255.0).astype(np.uint8)
    hsv_image = cv2.cvtColor(restored_uint8, cv2.COLOR_RGB2HSV)
    brightness = hsv_image[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(brightness)
    new_hsv_image = np.stack((hsv_image[:, :, 0], hsv_image[:, :, 1], cl1), axis=2)
    new_rgb_image = cv2.cvtColor(new_hsv_image, cv2.COLOR_HSV2RGB)
    return list((new_rgb_image.astype(np.float32) / 255.0).ravel())


def pad_cropped_image_to_original(original_image, cropped_image) -> np.array:
    """
    Put the samller image in the top left corner and pad out to the
    right and bottom with zeros to match the original image size.
    Does not alter image dtype
    """
    target_shape = original_image.shape

    # Create a new array of zeros with the target shape
    padded_array = np.zeros(target_shape, dtype=cropped_image.dtype)

    # Copy the smaller array into the top-left corner of the padded array
    padded_array[: cropped_image.shape[0], : cropped_image.shape[1]] = cropped_image
    return_array = np.concatenate((original_image, padded_array), axis=1)
    return return_array


def read_image_wrapper(df: pl.DataFrame, meta_df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one reads an image from disk and stores it as a flattened list in a column.
    We convert the image to float32 and normalize it to the range of 0-1. cvtColor is which
    we use extensively expects thing in uint8 format. We'll convert back to float32.
    The meta images are png with 4 channels add .convert('RGB') to convert to 3 channels
    doesn't affect the existing jpg
    """
    mem_file = BytesIO()
    df = df.with_columns(
        pl.col("Path")
        .map_elements(
            lambda x: convert_numpy_to_bytesio(
                np.array(Image.open(x).convert("RGB"), dtype=np.float32) / 255.0
            )
        )
        .alias("Image")
    )
    return df


def crop_to_roi_wrapper(df: pl.DataFrame, meta_df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one crops the image to the Roi values provided in the dataset
    """
    df = df.with_columns(
        pl.struct(["Width", "Height", "Roi.Y1", "Roi.Y2", "Roi.X1", "Roi.X2", "Image"])
        .map_elements(
            lambda x: dict(
                zip(
                    ("Cropped_Width", "Cropped_Height", "Cropped_Image"),
                    crop_to_roi(
                        x["Width"],
                        x["Height"],
                        x["Roi.Y1"],
                        x["Roi.Y2"],
                        x["Roi.X1"],
                        x["Roi.X2"],
                        x["Image"],
                    ),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    df = df.with_columns(
        (pl.col("Cropped_Width") * pl.col("Cropped_Height")).alias("Cropped_Resolution")
    )
    return df


def rescale_image_wrapper(df: pl.DataFrame, meta_df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one rescales the image to our standard size which is 64x64
    """
    df = df.with_columns(
        pl.col("Cropped_Image")
        .map_elements(
            lambda x: dict(
                zip(
                    ("Scaled_Width", "Scaled_Height", "Scaled_Image"),
                    rescale_image(x),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    df = df.with_columns(
        (pl.col("Scaled_Width") * pl.col("Scaled_Height")).alias("Scaled_Resolution")
    )
    return df


def stretch_histogram_lab_wrapper(
    df: pl.DataFrame, meta_df: pl.DataFrame
) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one stretches the histogram of the image to the full range of 0-255
    """
    df = df.with_columns(
        pl.struct(["Scaled_Width", "Scaled_Height", "Scaled_Image"])
        .map_elements(
            lambda x: stretch_histogram(
                x["Scaled_Width"], x["Scaled_Height"], x["Scaled_Image"]
            )
        )
        .alias("Stretched_LAB_Histogram_Image")
    )
    return df


def stretch_histogram_hsv_wrapper(
    df: pl.DataFrame, meta_df: pl.DataFrame
) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one stretches the histogram of the image to the full range of 0-255
    """
    df = df.with_columns(
        pl.struct(["Scaled_Width", "Scaled_Height", "Scaled_Image"])
        .map_elements(
            lambda x: stretch_histogram_hsv(
                x["Scaled_Width"], x["Scaled_Height"], x["Scaled_Image"]
            )
        )
        .alias("Stretched_Histogram_Image")
    )
    return df


def calculate_meta_image_stats(df: pl.DataFrame, meta_df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one calculates the mean and standard deviation of the image
    """
    df = df.with_columns(
        pl.col("Meta_Image")
        .map_elements(
            lambda x: dict(
                zip(("Meta_Width", "Meta_Height"), np.load(BytesIO(x)).shape)
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    df = df.with_columns(
        (pl.col("Meta_Width") * pl.col("Meta_Height")).alias("Meta_Resolution")
    )
    return df


def rescale_meta_image_wrapper(df: pl.DataFrame, meta_df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one rescales the image to our standard size which is 64x64
    """
    df = df.with_columns(
        pl.col("Meta_Image")
        .map_elements(
            lambda x: dict(
                zip(
                    ("Scaled_Meta_Width", "Scaled_Meta_Height", "Scaled_Meta_Image"),
                    rescale_image(x),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    df = df.with_columns(
        (pl.col("Scaled_Meta_Width") * pl.col("Scaled_Meta_Height")).alias(
            "Scaled_Meta_Resolution"
        )
    )
    return df


def process_meta_csv(csv_file: Path, root_dir: Path, num_cpus: int) -> pl.DataFrame:
    """
    Read the csv file into a polars dataframe.
    Read the image into a numpy array and store it in BytesIO object in numpy format
    This dataframe can be joined with the train and test dataframes
    """
    df = pl.read_csv(csv_file)
    # Update the path to be absolute so we're not passing around relative paths
    # this makes the parquet file machine dependent
    df = df.with_columns(
        pl.col("Path").map_elements(lambda x: update_path(x, root_dir))
    )
    print(f"\tBegin Reading Images", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, None, read_image_wrapper, num_cpus)
    df = df.rename({"Image": "Meta_Image"})
    df = parallelize_dataframe(df, None, calculate_meta_image_stats, num_cpus)
    df = parallelize_dataframe(df, None, rescale_meta_image_wrapper, num_cpus)
    end_time = datetime.now()
    print(f"\tEnd Reading Images:\t\t{end_time - start_time}", file=sys.stderr)
    return df


def process_csv(
    csv_file: Path, root_dir: Path, meta_df: pl.DataFrame, num_cpus: int
) -> pl.DataFrame:
    """
    Read the csv file into a polars dataframe.
    Read the image into a numpy array and store it in BytesIO object in numpy format
    This allows pyarrow to store it correctly in the parquet file
    Our images are in scale of 0 to 255, so we'll divide by 255 to normalize
    Crop the image to the Roi values provided in the dataset
    Rescale the image to our standard size which is 64x64
    """
    df = pl.read_csv(csv_file)
    df = df.with_columns((pl.col("Width") * pl.col("Height")).alias("Resolution"))
    # Update the path to be absolute so we're not passing around relative paths
    # this makes the parquet file machine dependent
    df = df.with_columns(
        pl.col("Path").map_elements(lambda x: update_path(x, root_dir))
    )

    # Read the image into a numpy array and store it in BytesIO object in numpy format
    # This allows pyarrow to store it correctly in the parquet file.
    # Change the Entire pipline back to storing the image as a list of float64
    # Deal with conversion whenever we deal with cvtColor
    # PIL is faster than cv2 in reading images
    print(f"\tBegin Reading Images", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, meta_df, read_image_wrapper, num_cpus)
    end_time = datetime.now()
    print(f"\tEnd Reading Images:\t\t{end_time - start_time}", file=sys.stderr)

    # Crop the image to the Roi values provided in the dataset
    print(f"\tBegin Cropping Images", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, meta_df, crop_to_roi_wrapper, num_cpus)
    end_time = datetime.now()
    print(f"\tEnd Cropping Images:\t\t{end_time - start_time}", file=sys.stderr)

    # Rescale the image to our standard size which is 64x64
    print(f"\tBegin Re-scaling Images", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, meta_df, rescale_image_wrapper, num_cpus)
    end_time = datetime.now()
    print(f"\tEnd Re-scaling Images:\t\t{end_time - start_time}", file=sys.stderr)

    ##
    # Turns out histogram stretching and re-assembly is bad according to Rachel. Do not use
    #
    # Stretch the histogram of the image to the full range of 0-100 in L Channel
    # This was abandoned in favor of stretching the V channel in HSV
    # print(f"\tBegin Histogram Stretching", file=sys.stderr)
    # start_time = datetime.now()
    # df = parallelize_dataframe(df, stretch_histogram_wrapper, num_cpus)
    # end_time = datetime.now()
    # print(f"\tEnd Histogram Stretching:\t{end_time - start_time}", file=sys.stderr)

    # Stretch the histogram of the image to the full range of 0-255 in V Channel
    # print(f"\tBegin HSV Histogram Stretching", file=sys.stderr)
    # start_time = datetime.now()
    # df = parallelize_dataframe(df, stretch_histogram_hsv_wrapper, num_cpus)
    # end_time = datetime.now()
    # print(f"\tEnd HSV Histogram Stretching:\t{end_time - start_time}", file=sys.stderr)

    return df


def main():
    script_start_time = start_time = datetime.now()
    print(80 * "=", file=sys.stderr)
    try:
        __file__
    except NameError:
        __file__ = None
    if __file__ is not None:
        script_name = Path(__file__)
    else:
        script_name = Path("./process_raw_dataset.py")
    script_dir = script_name.parent

    train_parquet = script_dir.joinpath("data/train.parquet")
    test_parquet = script_dir.joinpath("data/test.parquet")
    meta_parquet = script_dir.joinpath("data/meta_full.parquet")

    parser = argparse.ArgumentParser(description="Parse GTSRB dataset")
    parser.add_argument(
        "-r", dest="root_dir", help="dataset root directory", type=str, required=True
    )
    parser.add_argument(
        "-f",
        dest="force",
        help="force overwrite of existing parquet files",
        action="store_true",
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

    root_dir = Path(args.root_dir)
    if not root_dir.exists:
        print(f"Directory does not exit: {root_dir} ", file=sys.stderr)
        exit(1)
    prelim_ready = prelim_validate_dataset_dir(root_dir)
    if not prelim_ready:
        print(f"Preliminary Directory Check Failed", file=sys.stderr)
        exit(1)
    else:
        print(f"Preliminary Dataset Check Succeeded", file=sys.stderr)

    if (
        train_parquet.exists() or test_parquet.exists() or meta_parquet.exists()
    ) and not args.force:
        print(
            f"FATAL: Parquet files already exist. Use -f to force overwrite.",
            file=sys.stderr,
        )
        exit(1)

    if args.force:
        print(f"Force removing existing files.", file=sys.stderr)
        train_parquet.unlink(missing_ok=True)
        test_parquet.unlink(missing_ok=True)
        meta_parquet.unlink(missing_ok=True)

    if args.num_cpus is not None:
        num_cpus = args.num_cpus
    else:
        num_cpus = psutil.cpu_count(logical=False)

    if num_cpus > 8 and args.num_cpus is None:
        print(f"Number of cpus might be too high: {num_cpus}", file=sys.stderr)
        print(f"Forcing to 8 cpus", file=sys.stderr)
        print(f"Set number of cpus with -n option to override", file=sys.stderr)
        num_cpus = 8

    print(f"Multiprocessing on {num_cpus} CPUs", file=sys.stderr)
    print(f"Begin Processing Meta data.", file=sys.stderr)
    meta_start_time = datetime.now()
    meta_csv = root_dir.joinpath("Meta_full.csv")
    meta_df = process_meta_csv(meta_csv, root_dir, num_cpus)
    print(f"\tBegin Writing meta data", file=sys.stderr)
    start_time = datetime.now()
    meta_df.write_parquet(
        meta_parquet,
        compression="zstd",
        compression_level=5,
        # use_pyarrow=True,
    )
    end_time = datetime.now()
    print(
        f"\tEnd Writing meta data:\t\t{end_time - start_time}",
        file=sys.stderr,
    )
    meta_end_time = datetime.now()
    print(
        f"End Processing meta data:\t\t{meta_end_time - meta_start_time}",
        file=sys.stderr,
    )
    print(meta_df.head())
    meta_df = meta_df.select(["ClassId", "Scaled_Meta_Image"])
    print("Begin Processing test data.", file=sys.stderr)
    train_start_time = datetime.now()
    test_csv = root_dir.joinpath("Test.csv")
    test_df = process_csv(test_csv, root_dir, meta_df, num_cpus)

    print(f"\tBegin Writing test data", file=sys.stderr)
    start_time = datetime.now()
    test_df.write_parquet(
        test_parquet,
        compression="zstd",
        compression_level=5,
        # use_pyarrow=True,
    )
    end_time = datetime.now()
    print(
        f"\tEnd Writing test data:\t\t{end_time - start_time}",
        file=sys.stderr,
    )
    train_end_time = datetime.now()
    print(
        f"End Processing test data:\t\t{train_end_time - train_start_time}",
        file=sys.stderr,
    )
    print(test_df.head())
    del test_df  # free up some memory

    print(f"Begin Processing train data.", file=sys.stderr)
    test_start_time = datetime.now()
    train_csv = root_dir.joinpath("Train.csv")
    train_df = process_csv(train_csv, root_dir, meta_df, num_cpus)
    print(f"\tBegin Writing train data.", file=sys.stderr)
    start_time = datetime.now()
    train_df.write_parquet(
        train_parquet,
        compression="zstd",
        compression_level=5,
        # use_pyarrow=True,
    )
    end_time = datetime.now()
    print(
        f"\tEnd Writing train data:\t\t{end_time - start_time}",
        file=sys.stderr,
    )
    test_end_time = datetime.now()
    print(
        f"End Processing train data:\t\t{test_end_time - test_start_time}",
        file=sys.stderr,
    )
    print(train_df.head())
    script_end_time = datetime.now()
    print(80 * "=", file=sys.stderr)
    print(
        f"Total Elapsed Time:\t\t\t{script_end_time - script_start_time}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
