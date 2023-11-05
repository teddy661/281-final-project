import argparse
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


def main():
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

    test_csv = root_dir.joinpath("Test.csv")
    test_df = pl.read_csv(root_dir.joinpath("Test.csv"))

    test_df = test_df.with_columns(
        pl.col("Path").map_elements(lambda x: update_path(x, root_dir))
    )

    test_df = test_df.with_columns(
        pl.col("Path")
        .map_elements(lambda x: list(np.array(Image.open(x)).ravel()))
        .alias("Image")
    )

    test_df = test_df.with_columns(
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

    test_df = test_df.with_columns(
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

    print(test_df.head(5))


if __name__ == "__main__":
    main()
