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
    df["Numpy"] = df["Path"].map(lambda x: np.array(Image.open(x)))
    return df


def do_convert(x):
    mem_file = BytesIO()
    np.save(mem_file, x)
    return mem_file.getvalue()


def convert_to_bytes(df):
    mem_file = BytesIO()
    df["Bytesio"] = df["Numpy"].map(lambda x: do_convert(x))
    return df


def get_store():
    store = pd.HDFStore(get_store_path())
    return store


def get_store_path():
    return target_dir.parent.joinpath("sign_data.h5")


def store_exists():
    return get_store_path().exists()


def crop_to_roi(df):
    # df['Cropped']  = df['Numpy'].map(lambda x: x[x['RoiY1']:x['RoiY2'], x['RoiX1']:x['RoiX2'])
    df["Cropped"] = df.apply(
        lambda x: x["Numpy"][x["Roi.Y1"] : x["Roi.Y2"], x["Roi.X1"] : x["Roi.X2"]],
        axis=1,
    )
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
    end_time = datetime.now()
    print(f"\tTime elapsed: {end_time - start_time}", file=sys.stderr)

    # update path and load images
    print("(3/4) Processing train data.", file=sys.stderr)
    start_time = datetime.now()
    train_df = parallelize_dataframe(train_df, update_path, cpu_count)
    train_df = parallelize_dataframe(train_df, read_image_into_numpy, cpu_count)
    train_df = parallelize_dataframe(train_df, crop_to_roi, cpu_count)
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
