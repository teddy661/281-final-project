from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import psutil
from multiprocessing import Pool
from io import BytesIO

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
    store = pd.HDFStore(target_dir.parent.joinpath("sign_data.h5"))
    return store


def store_exists():
    return target_dir.parent.joinpath("sign_data.h5").exists()


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
        train_df = parallelize_dataframe(train_df, convert_to_bytes, cpu_count)
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
        test_df = parallelize_dataframe(test_df, convert_to_bytes, cpu_count)
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

    # read csv files
    meta_df = pd.read_csv(target_dir.joinpath("Meta.csv"))
    train_df = pd.read_csv(target_dir.joinpath("Train.csv"))
    test_df = pd.read_csv(target_dir.joinpath("Test.csv"))

    # update path and load images
    test_df = parallelize_dataframe(test_df, update_path, cpu_count)
    test_df = parallelize_dataframe(test_df, read_image_into_numpy, cpu_count)

    # update path and load images
    train_df = parallelize_dataframe(train_df, update_path, cpu_count)
    train_df = parallelize_dataframe(train_df, read_image_into_numpy, cpu_count)

    # write to disk as hdf5
    store = pd.HDFStore(target_dir.parent.joinpath("sign_data.h5"))
    store["test"] = test_df
    store["train"] = train_df
    store["meta"] = meta_df
    store.close()


if __name__ == "__main__":
    main()
