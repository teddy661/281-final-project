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


def crop_to_roi(width: int, height: int, y1: int, y2: int, x1: int, x2: int, image: list) -> tuple:
    image = np.array(image).reshape((height, width, 3))
    cropped_image = image[y1:y2+1, x1:x2+1, :]
    cropped_image_height = cropped_image.shape[0]
    cropped_image_width = cropped_image.shape[1]
    return pl.DataFrame({"Cropped_Height":cropped_image_height, "Cropped_Width":cropped_image_width, "Cropped_Image":list(cropped_image.ravel())})


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
        pl.col("Path").map_elements(lambda x: list(np.array(Image.open(x)).ravel())).alias("Image")
    )

    thing = test_df.map_elements(
            lambda x: crop_to_roi(x['Width'], x['Height'], x["Roi.Y1"], x["Roi.Y2"], x["Roi.X1"], x["Roi.X2"], x["Image"])
        ).explode()


    print(thing.head(5))


if __name__ == "__main__":
    main()
