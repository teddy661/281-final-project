import argparse
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import psutil
from PIL import Image
from skimage.transform import rescale, rotate


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


def restore_image_from_list(width: int, height: int, image: list) -> np.array:
    return np.array(image).reshape((height, width, 3)).astype(np.uint8)


def crop_to_roi(
    width: int, height: int, y1: int, y2: int, x1: int, x2: int, image: list
) -> tuple:
    image = restore_image_from_list(width, height, image)
    cropped_image = image[y1 : y2 + 1, x1 : x2 + 1, :]
    cropped_image_height = cropped_image.shape[0]
    cropped_image_width = cropped_image.shape[1]
    return cropped_image_width, cropped_image_height, list(cropped_image.ravel())


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


def stretch_histogram(width, height, image: np.array) -> list:
    """
    Input a uint8 image
    Stretch the histogram of the image to the full range of 0-255
    For our data this appears to work much better than equalizing the histogram
    We are only stretching the L channel of the LAB color space. This should
    preserve the possible the color information in the image.
    """
    restored_image = restore_image_from_list(width, height, image)
    lab_image = cv2.cvtColor(restored_image, cv2.COLOR_RGB2LAB)
    l_channel = lab_image[:, :, 0]
    min_val = np.min(l_channel)
    max_val = np.max(l_channel)
    new_min = 0
    new_max = 255
    stretched_l_channel = ((l_channel - min_val) / (max_val - min_val)) * (
        new_max - new_min
    ) + new_min
    stretched_lab_image = np.stack(
        (stretched_l_channel, lab_image[:, :, 1], lab_image[:, :, 2]), axis=2
    ).astype(np.uint8)
    rgb_image = cv2.cvtColor(stretched_lab_image, cv2.COLOR_LAB2RGB)
    return list(rgb_image.ravel())


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


def read_image_wrapper(df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one reads an image from disk and stores it as a flattened list in a column
    """
    df = df.with_columns(
        pl.col("Path")
        .map_elements(lambda x: list(np.array(Image.open(x)).ravel()))
        .alias("Image")
    )
    return df


def crop_to_roi_wrapper(df: pl.DataFrame) -> pl.DataFrame:
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


def rescale_image_wrapper(df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one rescales the image to our standard size which is 64x64
    """
    df = df.with_columns(
        pl.struct(["Cropped_Width", "Cropped_Height", "Cropped_Image"])
        .map_elements(
            lambda x: dict(
                zip(
                    ("Scaled_Width", "Scaled_Height", "Scaled_Image"),
                    rescale_image(
                        x["Cropped_Width"],
                        x["Cropped_Height"],
                        x["Cropped_Image"],
                    ),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    df = df.with_columns(
        (pl.col("Scaled_Width") * pl.col("Scaled_Height")).alias("Scaled_Resolution")
    )
    return df


def stretch_histogram_wrapper(df: pl.DataFrame) -> pl.DataFrame:
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
        .alias("Stretched_Histogram_Image")
    )
    return df


def process_csv(csv_file: Path, root_dir: Path, num_cpus: int) -> pl.DataFrame:
    """
    Read the csv file into a polars dataframe.
    Read the image into a numpy array and store it as a flattened list
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

    # Read the image into a numpy array and store it as a flattened list
    # This allows pyarrow to store it correctly in the parquet file
    # Our images are in scale of 0 to 255, so we'll divide by 255 to normalize
    # Turns out this is stupid. Leave them all a 0-255
    # PIL is faster than cv2 in reading images
    print(f"\tBegin Reading Images", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, read_image_wrapper, num_cpus)
    end_time = datetime.now()
    print(f"\tEnd Reading Images:\t\t{end_time - start_time}", file=sys.stderr)

    # Crop the image to the Roi values provided in the dataset
    print(f"\tBegin Cropping Images", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, crop_to_roi_wrapper, num_cpus)
    end_time = datetime.now()
    print(f"\tEnd Cropping Images:\t\t{end_time - start_time}", file=sys.stderr)

    # Rescale the image to our standard size which is 64x64
    print(f"\tBegin Re-scaling Images", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, rescale_image_wrapper, num_cpus)
    end_time = datetime.now()
    print(f"\tEnd Re-scaling Images:\t\t{end_time - start_time}", file=sys.stderr)

    # Stretch the histogram of the image to the full range of 0-255
    print(f"\tBegin Histogram Stretching", file=sys.stderr)
    start_time = datetime.now()
    df = parallelize_dataframe(df, stretch_histogram_wrapper, num_cpus)
    end_time = datetime.now()
    print(f"\tEnd Histogram Stretching:\t{end_time - start_time}", file=sys.stderr)
    return df


def main():
    script_start_time = start_time = datetime.now()

    try:
        __file__
    except NameError:
        __file__ = None
    if __file__ is not None:
        script_name = Path(__file__)
    else:
        script_name = Path("./process_raw_dataset.py")
    script_dir = script_name.parent

    train_parquet = script_dir.joinpath("Train.parquet")
    test_parquet = script_dir.joinpath("Test.parquet")

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

    if train_parquet.exists() and test_parquet.exists() and not args.force:
        print(
            f"Parquet files already exist. Use -f to force overwrite.", file=sys.stderr
        )
        exit(1)

    if args.force:
        print(f"Force removing existing files.", file=sys.stderr)
        train_parquet.unlink(missing_ok=True)
        test_parquet.unlink(missing_ok=True)

    if args.num_cpus is not None:
        num_cpus = args.num_cpus
    else:
        num_cpus = psutil.cpu_count(logical=False)

    if num_cpus > 12 and args.num_cpus is None:
        print(f"Number of cpus might be too high: {num_cpus}", file=sys.stderr)
        print(f"Forcing to 12 cpus", file=sys.stderr)
        print(f"Set number of cpus with -n option to override", file=sys.stderr)
        num_cpus = 12

    print(f"Multiprocessing on {num_cpus} CPUs", file=sys.stderr)
    print("Begin Processing test data.", file=sys.stderr)
    train_start_time = datetime.now()
    test_csv = root_dir.joinpath("Test.csv")
    test_df = process_csv(test_csv, root_dir, num_cpus)

    print(f"\tBegin Writing test data", file=sys.stderr)
    start_time = datetime.now()
    test_df.write_parquet(
        "Test.parquet",
        compression="zstd",
        compression_level=5,
        use_pyarrow=True,
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
    train_df = process_csv(train_csv, root_dir, num_cpus)
    print(f"\tBegin Writing train data.", file=sys.stderr)
    start_time = datetime.now()
    train_df.write_parquet(
        "Train.parquet",
        compression="zstd",
        compression_level=5,
        use_pyarrow=True,
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
    print(
        f"\n\nTotal Time elapsed:\t\t\t{script_end_time - script_start_time}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
