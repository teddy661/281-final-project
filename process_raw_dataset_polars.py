import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from skimage.transform import rescale, rotate


def prelim_validate_dataset_dir(root_dir: Path) -> bool:
    prelim_ready = True
    files = ["Meta_full.csv", "Meta.csv", "Test.csv", "Train.csv"]
    dirs = ["Meta", "Test", "Train"]
    for c_file in files:
        if not root_dir.joinpath(c_file).exists:
            print(f"Required File is Missing: {c_file}")
            prelim_ready = False
    for c_dir in dirs:
        if not root_dir.joinpath(c_dir).exists:
            print(f"Required Directory is Missing: {c_dir}")
            prelim_ready = False
        elif c_dir == "Train":
            for d in range(43):
                if not root_dir.joinpath(c_dir).joinpath(str(d)).exists:
                    print("Required Directory is Missing: {d}")
                    prelim_ready = False
    return prelim_ready


def update_path(path: Path, root_dir: Path) -> Path:
    return str(root_dir.joinpath(path).resolve())


def restore_image_from_list(width: int, height: int, image: list) -> np.array:
    return np.array(image).reshape((height, width, 3))


def crop_to_roi(
    width: int, height: int, y1: int, y2: int, x1: int, x2: int, image: list
) -> tuple:
    image = restore_image_from_list(width, height, image)
    cropped_image = image[y1 : y2 + 1, x1 : x2 + 1, :]
    cropped_image_height = cropped_image.shape[0]
    cropped_image_width = cropped_image.shape[1]
    return cropped_image_width, cropped_image_height, list(cropped_image.ravel())


def rescale_image(width: int, height: int, image: list, standard=64):
    # rescale short side to standard size, then crop center
    # median for our dataset is 35x35
    # we're doing geometric shapes for our image dataset so we'll scale these
    # way up to 64x64
    # This is going to return an image with the pixels normalized to 0-1
    image = restore_image_from_list(width, height, image)
    scale = standard / min(image.shape[:2])
    image = rescale(image, scale, anti_aliasing=True, channel_axis=2)
    image = image[
        int(image.shape[0] / 2 - standard / 2) : int(image.shape[0] / 2 + standard / 2),
        int(image.shape[1] / 2 - standard / 2) : int(image.shape[1] / 2 + standard / 2),
        :,
    ]
    scaled_image_height = image.shape[0]
    scaled_image_width = image.shape[1]
    return scaled_image_width, scaled_image_height, list(image.ravel())


def pad_cropped_image_to_original(original_image, cropped_image):
    target_shape = original_image.shape

    # Create a new array of zeros with the target shape
    padded_array = np.zeros(target_shape, dtype=cropped_image.dtype)

    # Copy the smaller array into the top-left corner of the padded array
    padded_array[: cropped_image.shape[0], : cropped_image.shape[1]] = cropped_image
    return_array = np.concatenate((original_image, padded_array), axis=1)
    return return_array


def process_csv(csv_file: Path, root_dir: Path) -> pl.DataFrame:
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
    print(f"\tBegin Reading Images", file=sys.stderr)
    start_time = datetime.now()
    df = df.with_columns(
        pl.col("Path")
        .map_elements(lambda x: list((np.array(Image.open(x)) / 255.0).ravel()))
        .alias("Image")
    )
    end_time = datetime.now()
    print(f"\tEnd Reading Images. Total Time: {end_time - start_time}", file=sys.stderr)

    # Crop the image to the Roi values provided in the dataset
    print(f"\tBegin Cropping Images", file=sys.stderr)
    start_time = datetime.now()
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
    end_time = datetime.now()
    print(
        f"\tEnd Cropping Images. Total Time: {end_time - start_time}", file=sys.stderr
    )

    # Rescale the image to our standard size which is 64x64
    print(f"\tBegin Rescaling Images", file=sys.stderr)
    start_time = datetime.now()
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
    end_time = datetime.now()
    print(
        f"\tEnd Rescaling Images. Total Time: {end_time - start_time}", file=sys.stderr
    )
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

    parser = argparse.ArgumentParser(description="Parse GTSRB dataset")
    parser.add_argument(
        "-r", dest="root_dir", help="dataset root directory", type=str, required=True
    )
    args = parser.parse_args()
    prog_name = parser.prog

    root_dir = Path(args.root_dir)
    if not root_dir.exists:
        print(f"Directory does not exit: {root_dir} ")
        exit(1)
    prelim_ready = prelim_validate_dataset_dir(root_dir)
    if not prelim_ready:
        print(f"Preliminary Directory Check Failed")
        exit(1)
    else:
        print(f"Preliminary Dataset Check Succeeded")

    print("Begin Processing test data.", file=sys.stderr)
    train_start_time = datetime.now()
    test_csv = root_dir.joinpath("Test.csv")
    test_df = process_csv(test_csv, root_dir)
    train_end_time = datetime.now()
    print(
        f"End Processing test data. Total Time: {train_end_time - train_start_time}",
        file=sys.stderr,
    )
    print(f"Begin Writing test data to parquet file", file=sys.stderr)
    test_df.write_parquet(
        "Test.parquet",
        compression="zstd",
        compression_level=5,
        use_pyarrow=True,
    )
    print(test_df.head())
    del test_df  # free up some memory
    print(f"End Writing test data to parquet file.", file=sys.stderr)

    print(f"Begin Processing train data.", file=sys.stderr)
    test_start_time = datetime.now()
    train_csv = root_dir.joinpath("Train.csv")
    train_df = process_csv(train_csv, root_dir)
    test_end_time = datetime.now()
    print(
        f"End Processing train data. Total Time: {test_end_time - test_start_time}",
        file=sys.stderr,
    )
    print(train_df.head())
    print(f"Begin Writing train data to parquet file.", file=sys.stderr)
    train_df.write_parquet(
        "Train.parquet",
        compression="zstd",
        compression_level=5,
        use_pyarrow=True,
    )
    print(f"End Writing train data to parquet file.", file=sys.stderr)

    script_end_time = datetime.now()
    print(
        f"\n\nTotal Time elapsed: {script_end_time - script_start_time}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
