import os
import urllib
import zipfile

import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from aif360.datasets import BankDataset, StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from codecarbon import OfflineEmissionsTracker
from tensorboard.compat import tf

from sostenibilita.modelManager import ModelManager
output_dir = '.'
output_file = 'emissions.csv'

modelManager= ModelManager()


def load_and_preprocess_data(url):
    zip_path, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        csv_path = zip_ref.extract('bank.csv')

    df = pd.read_csv(csv_path, sep=';')
    df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
    return df


def encode_categorical_features(df, categorical_features):
    le = LabelEncoder()
    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])

    return df

def split_and_normalize_data(df):
     X = df.drop('y', axis=1)
     y = df['y']
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     scaler = StandardScaler()
     X_train = scaler.fit_transform(X_train)
     X_test = scaler.transform(X_test)
     return X_train, X_test, y_train, y_test


def create_and_train_model(X_train, y_train):
    # Start monitoring with CodeCarbon
    tracker = OfflineEmissionsTracker(
        country_iso_code="ITA",
        output_file=output_file,
        output_dir=output_dir
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")


    tracker.start()

    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    for _ in range(10):
        model.fit(X_train, y_train, epochs=10, batch_size=32)

    tracker.stop()
    return model


#Function for calculating the metrics needed for the default model
def calculate_metrics(y_test, y_pred):
    return precision_score(y_test, y_pred), recall_score(y_test, y_pred),np.mean(y_pred)

#Function for calculating social metrics for the default model
def calculate_fairness_metrics(dataset, dataset_test, y_pred, unprivileged_groups, privileged_groups):
    dataset_test_pred = dataset_test.copy()
    dataset_test_pred.labels = y_pred
    metric_test_bld = BinaryLabelDatasetMetric(dataset_test_pred, unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
    classified_metric_test = ClassificationMetric(dataset_test, dataset_test_pred,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
    return metric_test_bld.mean_difference(), classified_metric_test.equal_opportunity_difference(), classified_metric_test.average_odds_difference()

def loadModel():
    global modelManager
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    df = load_and_preprocess_data(url)
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    df = encode_categorical_features(df, categorical_features)
    X_train, X_test, y_train, y_test = split_and_normalize_data(df)
    model = create_and_train_model(X_train, y_train)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    calculate_metrics(y_test, y_pred)
    dataset = StandardDataset(df, label_name='y', favorable_classes=[1], protected_attribute_names=['age'],
                                                          privileged_classes=[[1]])
    dataset_train, dataset_test = dataset.split([0.8], shuffle=True)
    X_train = dataset_train.features
    y_train = dataset_train.labels.ravel()
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    X_test = dataset_test.features
    y_test = dataset_test.labels.ravel()
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    precision_base, recall_base, mean = calculate_metrics(y_test, y_pred)
    mean_base, equal_opportunity_difference, average_odds_difference = calculate_fairness_metrics(dataset, dataset_test,
                                                                                                  y_pred, [{'age': 0}], [{'age': 1}])
    modelManager.addModel(
        name='modelBank',
        accuracy=accuracy_score(y_test, y_pred),
        precision_base=precision_base,
        recall_base=recall_base,
        f1_score_base=f1_score(y_test, y_pred),
        mean_base=mean,
        mean_difference_base=mean_base,
        equal_opportunity_difference=equal_opportunity_difference,
        average_odds_difference=average_odds_difference,
    )