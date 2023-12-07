import sys
from io import BytesIO
from pathlib import Path

import joblib
from sklearnex import patch_sklearn

patch_sklearn()

import multiprocessing as mp
from datetime import datetime
from itertools import combinations

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classification_data_loader import *

FEATURE_LIST = ["hue", "sat", "value", "hog", "lbp", "template", "vgg16", "resnet101"]


def build_model(features):
    """
    Fit model
    """
    svc_model = make_pipeline(
        SVC(
            kernel="linear",
            C=0.01,
            decision_function_shape="ovo",
            probability=False,
            random_state=42,
        )
    )
    raw_train_df, test_df, meta_df = load_raw_dataframes()
    sampled_train_df = get_sampled_data(raw_train_df)
    train_df, validation_df = split_data(sampled_train_df)
    del sampled_train_df, raw_train_df

    ### LBP
    LBP_scalar = StandardScaler()

    X_train_LBP, X_test_LBP, X_validation_LBP = get_lbp_features(
        train_df, test_df, validation_df
    )
    X_train_LBP = LBP_scalar.fit_transform(X_train_LBP)
    X_test_LBP = LBP_scalar.transform(X_test_LBP)
    X_validation_LBP = LBP_scalar.transform(X_validation_LBP)

    ### HSV Features
    Hue_scaler = StandardScaler()
    Sat_Scaler = StandardScaler()
    Val_Scaler = StandardScaler()

    X_train_Hue, X_test_Hue, X_validation_Hue = get_hue_features(
        train_df, test_df, validation_df
    )
    (
        X_train_Saturation,
        X_test_Saturation,
        X_validation_Saturation,
    ) = get_saturation_features(train_df, test_df, validation_df)
    X_train_Value, X_test_Value, X_validation_Value = get_value_features(
        train_df, test_df, validation_df
    )

    X_train_Hue = Hue_scaler.fit_transform(X_train_Hue)
    X_train_Saturation = Sat_Scaler.fit_transform(X_train_Saturation)
    X_train_Value = Val_Scaler.fit_transform(X_train_Value)

    X_test_Hue = Hue_scaler.transform(X_test_Hue)
    X_test_Saturation = Sat_Scaler.transform(X_test_Saturation)
    X_test_Value = Val_Scaler.transform(X_test_Value)

    X_validation_Hue = Hue_scaler.transform(X_validation_Hue)
    X_validation_Saturation = Sat_Scaler.transform(X_validation_Saturation)
    X_validation_Value = Val_Scaler.transform(X_validation_Value)

    ### Template Fratures
    Template_scalar = StandardScaler()

    X_train_Template, X_test_Template, X_validation_Template = get_template_features(
        train_df, test_df, validation_df
    )

    X_train_Template = Template_scalar.fit_transform(X_train_Template)
    X_test_Template = Template_scalar.transform(X_test_Template)
    X_validation_Template = Template_scalar.transform(X_validation_Template)

    ### HOG Features
    HOG_scaler = StandardScaler()

    X_train_HOG, X_test_HOG, X_validation_HOG = get_hog_features(
        train_df, test_df, validation_df
    )

    X_train_HOG = HOG_scaler.fit_transform(X_train_HOG)
    X_test_HOG = HOG_scaler.transform(X_test_HOG)
    X_validation_HOG = HOG_scaler.transform(X_validation_HOG)

    ### VGG16 Features
    VGG16_scaler = StandardScaler()

    X_train_VGG16, X_test_VGG16, X_validation_VGG16 = get_vgg16_features(
        train_df, test_df, validation_df
    )
    X_train_VGG16 = VGG16_scaler.fit_transform(X_train_VGG16)
    X_test_VGG = VGG16_scaler.transform(X_test_VGG16)
    X_validation_VGG = VGG16_scaler.transform(X_validation_VGG16)

    ### RESNET101 Features
    RESNET101_scaler = StandardScaler()

    (
        X_train_RESNET101,
        X_test_RESNET101,
        X_validation_RESNET101,
    ) = get_resnet101_features(train_df, test_df, validation_df)
    X_train_RESNET101 = RESNET101_scaler.fit_transform(X_train_RESNET101)
    X_test_RESNET101 = RESNET101_scaler.transform(X_test_RESNET101)
    X_validation_RESNET101 = RESNET101_scaler.transform(X_validation_RESNET101)
    X_train = np.empty((X_train_Template.shape[0], 0))
    X_test = np.empty((X_test_Template.shape[0], 0))
    X_validation = np.empty((X_validation_Template.shape[0], 0))

    if "lbp" in features:
        X_train = np.concatenate((X_train, X_train_LBP), axis=1)
        X_test = np.concatenate((X_test, X_test_LBP), axis=1)
        X_validation = np.concatenate((X_validation, X_validation_LBP), axis=1)
    if "hue" in features:
        X_train = np.concatenate((X_train, X_train_Hue), axis=1)
        X_test = np.concatenate((X_test, X_test_Hue), axis=1)
        X_validation = np.concatenate((X_validation, X_validation_Hue), axis=1)
    if "sat" in features:
        X_train = np.concatenate((X_train, X_train_Saturation), axis=1)
        X_test = np.concatenate((X_test, X_test_Saturation), axis=1)
        X_validation = np.concatenate((X_validation, X_validation_Saturation), axis=1)
    if "value" in features:
        X_train = np.concatenate((X_train, X_train_Value), axis=1)
        X_test = np.concatenate((X_test, X_test_Value), axis=1)
        X_validation = np.concatenate((X_validation, X_validation_Value), axis=1)
    if "template" in features:
        X_train = np.concatenate((X_train, X_train_Template), axis=1)
        X_test = np.concatenate((X_test, X_test_Template), axis=1)
        X_validation = np.concatenate((X_validation, X_vaidation_Template), axis=1)
    if "hog" in features:
        X_train = np.concatenate((X_train, X_train_HOG), axis=1)
        X_test = np.concatenate((X_test, X_test_HOG), axis=1)
        X_validation = np.concatenate((X_validation, X_validation_HOG), axis=1)
    if "vgg16" in features:
        X_train = np.concatenate((X_train, X_train_VGG16), axis=1)
        X_test = np.concatenate((X_test, X_test_VGG16), axis=1)
        X_validation = np.concatenate((X_validation, X_validation_VGG16), axis=1)
    if "resnet101" in features:
        X_train = np.concatenate((X_train, X_train_RESNET101), axis=1)
        X_test = np.concatenate((X_test, X_test_RESNET101), axis=1)
        X_validation = np.concatenate((X_validation, X_validation_RESNET101), axis=1)

    y_train = train_df["ClassId"].to_numpy()
    y_test = test_df["ClassId"].to_numpy()
    y_validation = validation_df["ClassId"].to_numpy()
    num_features = X_train.shape[1]
    svc_model.fit(X_train, y_train)
    joblib.dump(svc_model, f"models/linear-svc-{'-'.join(features)}.joblib")


def main():
    """
    Main function
    """

    all_combinations = []
    for r in range(1, len(FEATURE_LIST) + 1):
        all_combinations.extend(list(combinations(FEATURE_LIST, r)))

    for i, c in enumerate(all_combinations):
        start_time = datetime.now()
        process1 = mp.Process(target=build_model, args=(c,))
        process1.start()
        process1.join()
        end_time = datetime.now()
        print(
            f"{i+1}/{len(all_combinations)}:\t{'-'.join(c)}:\t{(end_time - start_time).total_seconds()}"
        )


if __name__ == "__main__":
    main()
