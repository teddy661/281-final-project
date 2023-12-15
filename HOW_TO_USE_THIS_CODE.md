
Download the German Street Signs dataset from [here](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and extract it to a folder in the root of the repository called `sign_data`. The folder structure should look like this:

``` sign_data
├── Meta
├── Meta.csv
├── Meta-full.csv (To be copied from github repo to this location)
├── Test
|  └── 00000.png
|  └── ...
├── Test.csv
├── Train
│   └── 00000_00000_00000.png
|   └── ...
└── Train.csv
```

Manually copy the created_csv\Meta-full.csv to sign_data\Meta-full.csv 

Install the required packages using `pip install -r requirements.txt`. We used python 3.11.7. Alternatively install the docker image using `docker pull ebrown/comp-viz-jupyter:latest` and -v this/repo:/tf/notebooks.

Create the parquet files used for analysis by running `python process_raw_dataset.py`. This will create the files `data/train.parquet` `sign_data/test.parquet` and `sign_data/meta_full.parquet`.

Create the features used for analysis by running `python create_features.py`. This will create the files `data/train_features.parquet` and `data/test_features.parquet`.

Add the vgg16 feature to the features parquet files by running `python create_vgg16_features.py`. This will update the files `data/train_features.parquet` and `data/test_features.parquet`.

Add the resnet101 feature to the features parquet files by running `python create_resnet101_features.py`. This will update the files `data/train_features.parquet` and `data/test_features.parquet`.

From now on the train_features.parquet and test_features.parquet files are used for the analysis.

`check_process.ipynb` is used to check the result of the process_raw_dataset.py script.

`check_features.ipynb` is used to check the result of the create_feature(s) scripts.

`check_classes.ipynb` looks at the median distribution of features per class in the dataset and displays them in the notebook. 

`evaluate_all_feature_combinations.py` breaks our feature vectors into the 255 unique combinations of features and runs an SVC linear model on all of them to 1) find the most accurate and most efficient combination of features and 2) to train and predict using the models to get an accurate representation of performance of each of the feature vectors in the same model on the same computer. 

`efficient_vs_most_accurate_features.ipynb` is used to visualize the results of evaluate_all_feature_combinations.py.

`svm_classifier_linear.ipynb` is used to train and evaluate the linear SVM model on our dataset. This file will also run a GridSearchCV to find the best hyperparameters for the model if enabled.

`CNN_LW.ipynb` evaluates the data against a 1d convolutional network. 

`PCA_LW.ipynb` is the Principle component anaylsis and tSNE analysis. 

`logistic_regression_model.ipynb` is the logitistic regression model using tensorflow
