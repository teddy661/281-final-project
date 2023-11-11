import cv2
import numpy as np
from skimage.feature import canny, hog, local_binary_pattern
from skimage.filters import farid, gaussian


def restore_image_from_list(
    width: int, height: int, image: list, num_channels: int = 3
) -> np.array:
    """
    Images are stored as lists in the parquet file. This function creates a numpy,
    array reshapes it to the correct size using the width and height of the image.
    Expects the original image to be a 3 channel image. We ensure the images
    are always written to disk as float64. This function reshapes it back to an image
    the dtype is not altered.

    There's something really odd in here. When I run this in ipython everything works
    when I run the script it always returns a float64. We'll just force it to float64
    """
    return np.asarray(image, dtype=np.float64).reshape((height, width, num_channels))


def image_for_display(image: np.array) -> np.array:
    """
    Convert image to 0-255 range with type np.uint8
    """
    return np.clip((image * 255.0), 0, 255).astype(np.uint8)


def blur_image(image: np.array, sigma=2) -> np.array:
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


def perform_sift(image: np.array) -> tuple:
    """
    Perform the SIFT algorithm on the image. We will use the L channel of the LAB color space
    We discarded this feature after exploration.
    """
    l_channel, a_channel, b_channel = cv2.split(convert_to_lab(image))
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(l_channel, None)
    return kp, des


def compute_hsv_histograms(image: np.array) -> np.array:
    """
    Compute the HSV histograms for the image
    """
    image = (image * 255.0).astype(np.uint8)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hue_hist, hue_edges = np.histogram(
        hsv_image[:, :, 0].ravel(), bins=180, range=(0, 180)
    )

    sat_hist, sat_edges = np.histogram(
        hsv_image[:, :, 1].ravel(), bins=256, range=(0, 256)
    )

    val_hist, val_edges = np.histogram(
        hsv_image[:, :, 2].ravel(), bins=256, range=(0, 256)
    )
    return hue_hist, hue_edges, sat_hist, sat_edges, val_hist, val_edges


def compute_lbp_image_and_histogram(image: np.array) -> np.array:
    """
    Compute the LBP image for the image. After much trial and error
    the best parameters are:
        radius = 3
        n_points = 16
    Well operate on the Lumanance channel of the LAB color space.
    There will be 18 types returned.
    """
    radius = 3
    n_points = 16
    image = (image * 255.0).astype(np.uint8)
    l_channel, a_channel, b_channel = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
    lbp_image = local_binary_pattern(l_channel, n_points, radius, method="uniform")
    lbp_hist, lbp_edges = np.histogram(
        lbp_image.ravel().astype(np.uint8), bins=18, range=(0, 18)
    )
    # We can cast this to  unit8 because we know the range is 0-17
    return lbp_image.astype(np.uint8), lbp_hist, lbp_edges


def normalize_histogram(hist: np.array) -> np.array:
    """
    Normalize the histogram or any array really
    """
    n_hist = hist.astype(np.float64)
    n_hist /= n_hist.sum() + 1e-6
    return n_hist
