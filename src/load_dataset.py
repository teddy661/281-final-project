####################################################################################################
# Description:                                                                                     #
# This file contains the functions to load the dataset.                                            #
#                                                                                                  #
# Expected File Structure:                                                                         #
# data/                                                                                            #
#   tfrecords-jpeg-512x512/                                                                        #
#     train/                                                                                       #
#       flowers00-230.tfrec                                                                        #
#       ...                                                                                        #
#     val/                                                                                         #
#       flowers00-30.tfrec                                                                         #
#       ...                                                                                        #
#     test/                                                                                        #
#       flowers00-30.tfrec                                                                         #
#       ...                                                                                        #
# Repeats for other Image Sizes                                                                    #
#                                                                                                  #
# Original Source:                                                                                 #
# https://www.kaggle.com/code/ryanholbrook/create-your-first-submission/notebook                   #
#                                                                                                  #
# Modifications:                                                                                   #
# - Remove GCS                                                                                     #
#                                                                                                  #
####################################################################################################

import math
import os
import re
from pathlib import Path

import numpy as np
import tensorflow as tf

script_path = Path(__file__).parent.parent.absolute()
data_dir = script_path.joinpath("data")

if not data_dir.exists():
    print("Data directory does not exist.")
    exit()

selected_size = 512

image_dirs = {
    512: "tfrecords-jpeg-512x512",
    331: "tfrecords-jpeg-331x331",
    224: "tfrecords-jpeg-224x224",
    192: "tfrecords-jpeg-192x192",
}

image_dir = image_dirs[selected_size]
IMAGE_SIZE = [selected_size, selected_size]

GCS_PATH = data_dir.joinpath(image_dir)
AUTO = tf.data.experimental.AUTOTUNE

TRAINING_FILENAMES = tf.io.gfile.glob(str(GCS_PATH) + "/train/*.tfrec")
VALIDATION_FILENAMES = tf.io.gfile.glob(str(GCS_PATH) + "/val/*.tfrec")
TEST_FILENAMES = tf.io.gfile.glob(str(GCS_PATH) + "/test/*.tfrec")

CLASSES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "wild geranium",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",  # 00 - 09
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",  # 10 - 19
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",  # 20 - 29
    "carnation",
    "garden phlox",
    "love in the mist",
    "cosmos",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",  # 30 - 39
    "barberton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "daisy",
    "common dandelion",  # 40 - 49
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "lilac hibiscus",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",  # 50 - 59
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "iris",
    "windflower",
    "tree poppy",  # 60 - 69
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",  # 70 - 79
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen ",
    "watercress",
    "canna lily",  # 80 - 89
    "hippeastrum ",
    "bee balm",
    "pink quill",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",  # 90 - 99
    "trumpet creeper",
    "blackberry lily",
    "common tulip",
    "wild rose",
]  # 100 - 102


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = (
        tf.cast(image, tf.float32) / 255.0
    )  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3])  # explicit size needed for TPU
    return image


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["class"], tf.int32)
    return image, label  # returns a dataset of (image, label) pairs


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    idnum = example["id"]
    return image, idnum  # returns a dataset of image(s)


def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable order, increase speed

    dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=AUTO
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
    )
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


def data_augment(image, label):
    # Thanks to the dataset.prefetch(AUTO)
    # statement in the next function (below), this happens essentially
    # for free on TPU. Data pipeline code is executed on the "CPU"
    # part of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_saturation(image, 0, 2)
    return image, label


def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()  # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(
        AUTO
    )  # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec
    # files, i.e. flowers00-230.tfrec = 230 data items
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
print(
    "Dataset: {} training images, {} validation images, {} unlabeled test images".format(
        NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES
    )
)
