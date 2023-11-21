from io import BytesIO
from multiprocessing import Pool
from typing import Callable

import cv2
import numpy as np
import polars as pl
from skimage.feature import canny, hog, local_binary_pattern
from skimage.filters import farid, gaussian


def parallelize_dataframe(
    df: pl.DataFrame, func: Callable[[pl.DataFrame], pl.DataFrame], n_cores: int = 4
) -> pl.DataFrame:
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


def blur_image(image: np.array, sigma: int = 2) -> np.array:
    """
    Blur the image to remove noise
    """
    return gaussian(image, sigma=sigma, channel_axis=-1)


def edge_detection(image: np.array) -> np.array:
    """
    Look at Canny and Farid Edge Detection algorithms. Farid was better for our use
    case, but doesn't give us anythong above and beyond our LPB feature.
    """
    image = (image * 255.0).astype(np.uint8)
    l_channel, a_channel, b_channel = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
    # return farid(l_channel, axis=-1)
    return canny(l_channel, sigma=3)


def convert_to_lab(image: np.array) -> np.array:
    """
    Take an existing RGB image and convert it to LAB color space
    """
    image = (image * 255.0).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def create_local_binary_pattern(image: np.array) -> np.array:
    """
    Create the local binary pattern for the image
    """
    image = (image * 255.0).astype(np.uint8)
    l_channel, a_channel, b_channel = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
    return local_binary_pattern(l_channel, 8, 1, method="uniform")


def create_hog_features(image: np.array) -> np.array:
    """
    This function expects a 3 channel image
    The best parameters for the HOG features are:
            block_norm="L2-Hys",
            pixels_per_cell=(6, 6),
            cells_per_block=(2, 2),
    There appears to be no difference betwen using the RGB or LAB color space
    Therefore we will use the RGB color space
    The hog image doesn't need to be 0-255 it will work with a normalized image
    as input
    """
    features_rgb, hog_image_rgb = hog(
        image,
        orientations=9,
        block_norm="L2-Hys",
        pixels_per_cell=(6, 6),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=-1,
    )
    # HOG  Features is a 1-D 2916 element array
    return features_rgb, hog_image_rgb


def stretch_gray_histogram(image: np.array) -> np.array:
    """
    Input a single channel uint8 image 0-255
    Stretch the histogram of the image to the full range of 0-255
    """

    min_val = np.min(image)
    max_val = np.max(image)
    new_min = 0
    new_max = 255
    stretched_image = np.interp(image, (min_val, max_val), (new_min, new_max)).astype(
        image.dtype
    )
    return stretched_image


def compute_sift(image: np.array) -> np.array:
    """
    Imput 3 channel rgb image on the range 0-1 float32
    Convert to grayscale
    Perform the SIFT algorithm on the image.
    """
    uint8_image = (image * 255.0).astype(np.uint8)
    gray_image = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2GRAY)
    stretched_image = stretch_gray_histogram(gray_image)
    sift = cv2.SIFT_create()
    _, des = sift.detectAndCompute(stretched_image, None)
    return des


def compute_hsv_histograms(image: np.array) -> np.array:
    """
    Compute the HSV histograms for the image. Convert image to 255 scale
    Change Color Space to HSV, return histograms and edges
    """
    image = (image * 255.0).astype(np.uint8)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hue_hist, hue_edges = np.histogram(
        hsv_image[:, :, 0].astype(np.uint8).ravel(), bins=180, range=(0, 180)
    )

    sat_hist, sat_edges = np.histogram(
        hsv_image[:, :, 1].astype(np.uint8).ravel(), bins=256, range=(0, 256)
    )

    val_hist, val_edges = np.histogram(
        hsv_image[:, :, 2].astype(np.uint8).ravel(), bins=256, range=(0, 256)
    )
    return hue_hist, hue_edges, sat_hist, sat_edges, val_hist, val_edges


def convert_numpy_to_bytesio(image: np.array) -> bytes:
    """
    Save a numpy array to a BytesIO object
    """
    mem_file = BytesIO()
    np.save(mem_file, image)
    return mem_file.getvalue()


def compute_lbp_image_and_histogram(image: np.array) -> np.array:
    """
    Expect an input image on the range 0-1 float32. Convert to 255 scale
    for color space conversion. Convert to uint8 for LBP computation.
    After much trial and error the best parameters are:
        radius = 3
        n_points = 16
        method = "uniform"
    Well operate on the Lumanance channel of the LAB color space.
    There will be 18 types returned.
    """
    radius = 3
    n_points = 16
    image = (image * 255.0).astype(np.uint8)
    l_channel, a_channel, b_channel = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
    lbp_image = local_binary_pattern(l_channel, n_points, radius, method="uniform")
    n_bins = int(lbp_image.max() + 1)
    lbp_hist, lbp_edges = np.histogram(
        lbp_image.ravel().astype(np.uint8), bins=n_bins, range=(0, n_bins)
    )
    # We can cast this to  unit8 because we know the range is 0-17
    return lbp_image.astype(np.uint8), lbp_hist, lbp_edges


def normalize_histogram(hist: np.array) -> np.array:
    """
    Normalize the histogram or any array really
    """
    n_hist = hist.astype(np.float32)
    n_hist /= n_hist.sum() + 1e-6
    return n_hist
