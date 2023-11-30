import itertools
import multiprocessing as mp
import os
from io import BytesIO
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
# import tensorflow as tf
from matplotlib import ticker
from PIL import Image
from skimage.feature import canny, hog, local_binary_pattern
from skimage.filters import gaussian

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Class to match templates (Advanced Feature Extraction)
# class TemplateMatching:
#     """
#     TQDM and display are no no in batch processing. remove them
#     """

#     def __init__(self, img=None, template=None, thresholdVal=-1):
#         """
#         Class instanciation
#         Input parameter: img - Original Image
#                          template - Image template
#                          thresholdVal - Pixels in the template with value greater that threshold are used
#                                         -1 (default) to use all pixels or 0  to use edges with value > 0
#         """
#         self.image = img
#         self.template = template
#         self.thresholdVal = thresholdVal
#         self.maxima = 0
#         self.minima = 0
#         return

#     def get_imageMaxMin(self, img=None):
#         img = self.image if img is None else img
#         return np.amax(img), np.amin(img)

#     def shrink_image(self, img=None, factor=30):
#         img = self.image if img is None else img

#         width, height = img.shape[1], img.shape[0]
#         width = (width * factor) // 100
#         height = (height * factor) // 100

#         lchannel = False
#         if len(img.shape) <= 2:
#             img = tf.expand_dims(img, -1)
#             lchannel = True

#         img = tf.image.resize(img, size=(width, height), preserve_aspect_ratio=True)
#         if lchannel is True:
#             img = img[:, :, 0]

#         return img

#     def plot3DHistograma(self, img, zRange=[0, 0], axesView=[0, 0], zTicks=True):
#         def _major_formatter(x, pos):
#             return "{:.1f}".format(x + zRange[0])

#         img = self.image if img is None else img

#         width = img.shape[1]
#         height = img.shape[0]

#         # Z clipping fails to clip bar charts
#         if zRange[0] != 0:
#             for x in range(0, width):
#                 for y in range(0, height):
#                     img[y, x] -= zRange[0]
#                     if img[y, x] < 0:
#                         img[y, x] = 0

#         # Convert data to array
#         zData = np.array(img)

#         # Create an X-Y mesh of the same dimension as the 2D data
#         xData, yData = np.meshgrid(np.arange(width), np.arange(height))

#         # Flatten the arrays so that they may be passed to "axes.bar3d".
#         xData = xData.flatten()
#         yData = yData.flatten()
#         zData = zData.flatten()

#         # Create figure
#         fig = plt.figure()
#         axes = fig.add_subplot(projection="3d")
#         axes.bar3d(
#             xData,
#             yData,
#             np.zeros(len(zData)),
#             0.98,
#             0.98,
#             zData,
#             alpha=1.0,
#             zsort="min",
#         )

#         # Axes
#         axes.set_xlim3d(0, width)
#         axes.set_ylim3d(0, height)

#         # View
#         if axesView[0] != 0 or axesView[1] != 0:
#             axes.view_init(axesView[0], axesView[1])

#         # Shift the tick labels up by minimum, set_zlim3d does not work
#         if zRange[0] != 0 or zRange[1] != 0:
#             if zTicks:
#                 axes.zaxis.set_major_formatter(ticker.FuncFormatter(_major_formatter))
#             else:
#                 axes.zaxis.set_major_formatter(ticker.NullFormatter())

#         # Set the shifted range
#         if zRange[0] != 0 or zRange[1] != 0:
#             axes.set_zlim3d([0, zRange[1] - zRange[0]])

#         plt.title("Histogram of Pattern Matching", fontsize=10)
#         plt.show()
#         return

#     def match_template(self, img=None, template=None, thresholdVal=-1):
#         img = self.image if img is None else img
#         template = self.template if template is None else template

#         # Reshape images (eliminate channel dimension if at all)
#         if len(img.shape) > 2:
#             img = img[:, :, 0]
#         if len(template.shape) > 2:
#             template = template[:, :, 0]

#         # Get sizes
#         width, height = img.shape[1], img.shape[0]
#         widthTemplate, heightTemplate = template.shape[1], template.shape[0]

#         # Create a collector
#         collector = np.zeros((height, width), dtype="float")

#         # Template matching
#         tCenterX = (widthTemplate - 1) // 2
#         tCenterY = (heightTemplate - 1) // 2

#         for x in range(0, width):
#             for y in range(0, height):
#                 for wx, wy in itertools.product(
#                     range(0, widthTemplate), range(0, heightTemplate)
#                 ):
#                     posY = y + wy - tCenterY
#                     posX = x + wx - tCenterX

#                     # The threshold is used to accumulate only the edge pixels in an edge template
#                     # The difference of pixel values is inverted to show the best match as a peak
#                     if (
#                         posY > -1
#                         and posY < height
#                         and posX > -1
#                         and posX < width
#                         and template[wy, wx] > thresholdVal
#                     ):
#                         diff = (
#                             1.0
#                             - abs(float(img[posY, posX]) - float(template[wy, wx]))
#                             / 255.0
#                         )
#                         collector[y, x] += diff * diff

#         # Get collector (accumulator) within a maxima and mininma region
#         self.maxima, self.minima = self.get_imageMaxMin(collector)

#         return collector


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
    pool = mp.Pool(n_cores)
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
        radius = 1
        n_points = 16
        method = "default"
    Convert Image to grayscale and then stretch the histogram to the full range
    There will be 18 types returned. we know that radius 2 points 12  will not
    run out of memory on a 256GB machine. In an attempt to get better accuracy
    we would like to bump this up to 3 and 24 that will require segmenting the
    input of the training data
    """
    radius = 1
    n_points = 16
    uint8_image = (image * 255.0).astype(np.uint8)
    gray_image = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2GRAY)
    stretched_image = stretch_gray_histogram(gray_image)
    lbp_image = local_binary_pattern(
        stretched_image, n_points, radius, method="default"
    )
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
