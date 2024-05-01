import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import zipfile
import urllib.request
import os

# Carica il dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv(url, names=col_names)

# Preprocessing
df.replace(' ?', np.nan, inplace=True)
df = df.dropna()
df['income'] = df['income'].apply(lambda x: 1 if x==' >50K' else 0)

# Codifica le variabili categoriche
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
le = LabelEncoder()
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Divisione del dataset
X = df.drop('income', axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizzazione
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crea il modello
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Addestra il modello
model.fit(X_train, y_train, epochs=10, batch_size=32)

from aif360.metrics import ClassificationMetric

# Calcolo delle metriche di equità con aif360
dataset = StandardDataset(df, label_name='income', favorable_classes=[1], protected_attribute_names=['sex'], privileged_classes=[[1]])
dataset_train, dataset_test = dataset.split([0.8], shuffle=True)

# Addestra il modello sui dati di addestramento
X_train = dataset_train.features
y_train = dataset_train.labels.ravel()
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Valuta il modello sui dati di test
X_test = dataset_test.features
y_test = dataset_test.labels.ravel()
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calcola le metriche di base
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Calcola le metriche di equità
dataset_test_pred = dataset_test.copy()
dataset_test_pred.labels = y_pred
metric_test_bld = BinaryLabelDatasetMetric(dataset_test_pred, unprivileged_groups=[{'sex': 0}], privileged_groups=[{'sex': 1}])
print("Mean Difference:", metric_test_bld.mean_difference())

classified_metric_test = ClassificationMetric(dataset_test, dataset_test_pred, unprivileged_groups=[{'sex': 0}], privileged_groups=[{'sex': 1}])
print("Equal Opportunity Difference:", classified_metric_test.equal_opportunity_difference())
print("Average Odds Difference:", classified_metric_test.average_odds_difference())


# Carica il dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
zip_path, _ = urllib.request.urlretrieve(url)

# Estrai il file csv
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    csv_path = zip_ref.extract('bank.csv')

# Ora puoi leggere il file csv
df = pd.read_csv(csv_path, sep=';')


# Preprocessing
df['y'] = df['y'].apply(lambda x: 1 if x=='yes' else 0)

# Codifica le variabili categoriche
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
le = LabelEncoder()
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Divisione del dataset
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizzazione
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crea il modello
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Addestra il modello
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Valutazione del modello
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Calcolo delle metriche di equità con aif360
dataset = StandardDataset(df, label_name='y', favorable_classes=[1], protected_attribute_names=['age'], privileged_classes=[[1]])
dataset_train, dataset_test = dataset.split([0.8], shuffle=True)

# Addestra il modello sui dati di addestramento
X_train = dataset_train.features
y_train = dataset_train.labels.ravel()
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Valuta il modello sui dati di test
X_test = dataset_test.features
y_test = dataset_test.labels.ravel()
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calcola le metriche di base
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Calcola le metriche di equità
dataset_test_pred = dataset_test.copy()
dataset_test_pred.labels = y_pred
metric_test_bld = BinaryLabelDatasetMetric(dataset_test_pred, unprivileged_groups=[{'age': 0}], privileged_groups=[{'age': 1}])
print("Mean Difference:", metric_test_bld.mean_difference())

classified_metric_test = ClassificationMetric(dataset_test, dataset_test_pred, unprivileged_groups=[{'age': 0}], privileged_groups=[{'age': 1}])
print("Equal Opportunity Difference:", classified_metric_test.equal_opportunity_difference())
print("Average Odds Difference:", classified_metric_test.average_odds_difference())