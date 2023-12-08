import sys
from io import BytesIO
from pathlib import Path

import joblib
from sklearnex import patch_sklearn

patch_sklearn()

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


def main():
    """
    Main function
    """

    all_feature_combinations = []
    for r in range(1, len(FEATURE_LIST) + 1):
        all_feature_combinations.extend(list(combinations(FEATURE_LIST, r)))

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

    y_train = train_df["ClassId"].to_numpy()
    y_test = test_df["ClassId"].to_numpy()
    y_validation = validation_df["ClassId"].to_numpy()
    results_df = pl.DataFrame()
    for i, current_features in enumerate(all_feature_combinations):
        X_train = np.empty((y_train.shape[0], 0))
        X_test = np.empty((y_test.shape[0], 0))
        X_validation = np.empty((y_validation.shape[0], 0))

        svc_model = make_pipeline(
            SVC(
                kernel="linear",
                C=0.01,
                decision_function_shape="ovo",
                probability=False,
                random_state=42,
            )
        )

        if "lbp" in current_features:
            X_train = np.concatenate((X_train, X_train_LBP), axis=1)
            X_test = np.concatenate((X_test, X_test_LBP), axis=1)
            X_validation = np.concatenate((X_validation, X_validation_LBP), axis=1)
        if "hue" in current_features:
            X_train = np.concatenate((X_train, X_train_Hue), axis=1)
            X_test = np.concatenate((X_test, X_test_Hue), axis=1)
            X_validation = np.concatenate((X_validation, X_validation_Hue), axis=1)
        if "sat" in current_features:
            X_train = np.concatenate((X_train, X_train_Saturation), axis=1)
            X_test = np.concatenate((X_test, X_test_Saturation), axis=1)
            X_validation = np.concatenate(
                (X_validation, X_validation_Saturation), axis=1
            )
        if "value" in current_features:
            X_train = np.concatenate((X_train, X_train_Value), axis=1)
            X_test = np.concatenate((X_test, X_test_Value), axis=1)
            X_validation = np.concatenate((X_validation, X_validation_Value), axis=1)
        if "template" in current_features:
            X_train = np.concatenate((X_train, X_train_Template), axis=1)
            X_test = np.concatenate((X_test, X_test_Template), axis=1)
            X_validation = np.concatenate((X_validation, X_validation_Template), axis=1)
        if "hog" in current_features:
            X_train = np.concatenate((X_train, X_train_HOG), axis=1)
            X_test = np.concatenate((X_test, X_test_HOG), axis=1)
            X_validation = np.concatenate((X_validation, X_validation_HOG), axis=1)
        if "vgg16" in current_features:
            X_train = np.concatenate((X_train, X_train_VGG16), axis=1)
            X_test = np.concatenate((X_test, X_test_VGG16), axis=1)
            X_validation = np.concatenate((X_validation, X_validation_VGG16), axis=1)
        if "resnet101" in current_features:
            X_train = np.concatenate((X_train, X_train_RESNET101), axis=1)
            X_test = np.concatenate((X_test, X_test_RESNET101), axis=1)
            X_validation = np.concatenate(
                (X_validation, X_validation_RESNET101), axis=1
            )

        num_features = X_train.shape[1]
        train_start_time = datetime.now()
        svc_model.fit(X_train, y_train)
        train_end_time = datetime.now()
        total_train_time = (train_end_time - train_start_time).total_seconds()
        model_path = Path(f"models/linear-svc-{'-'.join(current_features)}.joblib")
        # joblib.dump(svc_model, model_path) # This creates over 1TB of joblib files. Disabled for now.
        predict_start_time = datetime.now()
        y_pred = svc_model.predict(X_test)
        predict_end_time = datetime.now()
        total_predict_time = (predict_end_time - predict_start_time).total_seconds()
        current_results_df = pl.DataFrame(
            {
                "Features": [current_features],
                "NumFeatures": [num_features],
                "TrainTime": [total_train_time],
                "PredictTime": [total_predict_time],
                "Accuracy": [accuracy_score(y_test, y_pred)],
                "ConfusionMatrix_Shape": [confusion_matrix(y_test, y_pred).shape],
                "ConfusionMatrix": [confusion_matrix(y_test, y_pred).flatten()],
                "ClassificationReport": [classification_report(y_test, y_pred)],
                "ModelPath": [str(model_path)],
            }
        )
        results_df = pl.concat([results_df, current_results_df])
        print(
            results_df.select(
                ["NumFeatures", "Accuracy", "TrainTime", "PredictTime", "Features"]
            )
        )
    results_df.write_parquet(
        "linear-svc-results.parquet", compression="zstd", compression_level=6
    )


if __name__ == "__main__":
    main()
