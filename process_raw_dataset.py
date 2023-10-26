import sys
import warnings
from datetime import datetime
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from PIL import Image
from skimage.transform import rescale, rotate

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

try:
    __file__
except NameError:
    __file__ = None
if __file__ is not None:
    script_name = Path(__file__)
else:
    script_name = Path("./process_raw_dataset.py")

target_dir = script_name.parent.joinpath("sign_data")

if not target_dir.exists():
    print(f"Failed to find target directory: {target_dir}")
    exit(1)


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def update_path(df):
    df["Path"] = df["Path"].map(lambda x: str((target_dir.joinpath(x))))
    return df


def read_image_into_numpy(df):
    df["Image"] = df["Path"].map(lambda x: np.array(Image.open(x)))
    return df


def do_convert(x):
    mem_file = BytesIO()
    np.save(mem_file, x)
    return mem_file.getvalue()


def convert_to_bytes(df):
    mem_file = BytesIO()
    df["Bytesio"] = df["Image"].map(lambda x: do_convert(x))
    return df


def get_store():
    store = pd.HDFStore(get_store_path())
    return store


def get_store_path():
    return target_dir.parent.joinpath("sign_data.h5")


def store_exists():
    return get_store_path().exists()


def crop_to_roi(df):
    df["Cropped_Image"] = df.apply(
        lambda x: x["Image"][
            x["Roi.Y1"] : x["Roi.Y2"] + 1, x["Roi.X1"] : x["Roi.X2"] + 1
        ],
        axis=1,
    )
    return df


def get_cropped_dimensions(df):
    df["Cropped_Height"] = df["Cropped_Image"].apply(lambda x: x.shape[0])
    df["Cropped_Width"] = df["Cropped_Image"].apply(lambda x: x.shape[1])
    return df


def rescale_image(img, standard=35):
    # rescale short side to standard size, then crop center
    # median for our dataset is 35x35
    scale = standard / min(img.shape[:2])
    img = rescale(img, scale, anti_aliasing=True, channel_axis=2)
    img = img[
        int(img.shape[0] / 2 - standard / 2) : int(img.shape[0] / 2 + standard / 2),
        int(img.shape[1] / 2 - standard / 2) : int(img.shape[1] / 2 + standard / 2),
        :,
    ]
    return img


def rescale_cropped_image(df):
    df["Standard_Image"] = df["Cropped_Image"].apply(lambda x: rescale_image(x))
    return df


def get_train_df(create_cache=False):
    cpu_count = psutil.cpu_count(logical=False)
    train_df = None
    store = get_store()
    if not create_cache:
        train_df = store["train"]
    else:
        train_df = pd.read_csv(target_dir.joinpath("Train.csv"))
        train_df = parallelize_dataframe(train_df, update_path, cpu_count)
        train_df = parallelize_dataframe(train_df, read_image_into_numpy, cpu_count)
        train_df = parallelize_dataframe(train_df, crop_to_roi, cpu_count)
        train_df = parallelize_dataframe(train_df, get_cropped_dimensions, cpu_count)
        train_df = parallelize_dataframe(train_df, rescale_cropped_image, cpu_count)
        store["train"] = train_df
    store.close()
    return train_df


def get_test_df(create_cache=False):
    cpu_count = psutil.cpu_count(logical=False)
    test_df = None
    store = get_store()
    if not create_cache:
        test_df = store["test"]
    else:
        test_df = pd.read_csv(target_dir.joinpath("Test.csv"))
        test_df = parallelize_dataframe(test_df, update_path, cpu_count)
        test_df = parallelize_dataframe(test_df, read_image_into_numpy, cpu_count)
        test_df = parallelize_dataframe(test_df, crop_to_roi, cpu_count)
        test_df = parallelize_dataframe(test_df, get_cropped_dimensions, cpu_count)
        test_df = parallelize_dataframe(test_df, rescale_cropped_image, cpu_count)
        store["test"] = test_df
    store.close()
    return test_df


def get_meta_df(create_cache=False):
    meta_df = None
    store = get_store()
    if not create_cache:
        meta_df = store["meta"]
    else:
        meta_df = pd.read_csv(target_dir.joinpath("Meta.csv"))
        store["meta"] = meta_df
    store.close()
    return meta_df


def pad_cropped_image_to_original(original_image, cropped_image):
    target_shape = original_image.shape

    # Create a new array of zeros with the target shape
    padded_array = np.zeros(target_shape, dtype=cropped_image.dtype)

    # Copy the smaller array into the top-left corner of the padded array
    padded_array[: cropped_image.shape[0], : cropped_image.shape[1]] = cropped_image
    return_array = np.concatenate((original_image, padded_array), axis=1)
    return return_array


def main():
    cpu_count = psutil.cpu_count(logical=False)

    store_path = get_store_path()
    if store_path.exists():
        store_path.unlink()

    # read csv files
    print("(1/4) Processing csv files.", file=sys.stderr)
    start_time = datetime.now()
    meta_df = pd.read_csv(target_dir.joinpath("Meta.csv"))
    train_df = pd.read_csv(target_dir.joinpath("Train.csv"))
    test_df = pd.read_csv(target_dir.joinpath("Test.csv"))
    end_time = datetime.now()
    print(f"\tTime elapsed: {end_time - start_time}", file=sys.stderr)

    # update path and load images
    print("(2/4) Processing test data.", file=sys.stderr)
    start_time = datetime.now()
    test_df = parallelize_dataframe(test_df, update_path, cpu_count)
    test_df = parallelize_dataframe(test_df, read_image_into_numpy, cpu_count)
    test_df = parallelize_dataframe(test_df, crop_to_roi, cpu_count)
    test_df = parallelize_dataframe(test_df, get_cropped_dimensions, cpu_count)
    test_df = parallelize_dataframe(test_df, rescale_cropped_image, cpu_count)
    end_time = datetime.now()
    print(f"\tTime elapsed: {end_time - start_time}", file=sys.stderr)

    # update path and load images
    print("(3/4) Processing train data.", file=sys.stderr)
    start_time = datetime.now()
    train_df = parallelize_dataframe(train_df, update_path, cpu_count)
    train_df = parallelize_dataframe(train_df, read_image_into_numpy, cpu_count)
    train_df = parallelize_dataframe(train_df, crop_to_roi, cpu_count)
    train_df = parallelize_dataframe(train_df, get_cropped_dimensions, cpu_count)
    train_df = parallelize_dataframe(train_df, rescale_cropped_image, cpu_count)
    end_time = datetime.now()
    print(f"\tTime elapsed: {end_time - start_time}", file=sys.stderr)

    # write to disk as hdf5
    print("(4/4) Writing data to disk.", file=sys.stderr)
    start_time = datetime.now()
    store = pd.HDFStore(get_store_path())
    store["test"] = test_df
    store["train"] = train_df
    store["meta"] = meta_df
    store.close()
    end_time = datetime.now()
    print(f"\tTime elapsed: {end_time - start_time}", file=sys.stderr)


if __name__ == "__main__":
    main()
