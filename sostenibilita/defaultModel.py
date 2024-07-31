# import csv
# import json
# import os
# import pickle
#
# import pandas as pd
# import numpy as np
# from codecarbon import OfflineEmissionsTracker
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from aif360.datasets import StandardDataset
# from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
# from tensorboard.compat import tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import plotly.express as px
#
# output_dir = '.'
# output_file = 'emissions.csv'
#
#
# def load_and_preprocess_data(url):
#     col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
#                  'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
#                  'income']
#     df = pd.read_csv(url, names=col_names)
#     df.replace(' ?', np.nan, inplace=True)
#     df = df.dropna()
#     df['income'] = df['income'].apply(lambda x: 1 if x == ' >50K' else 0)
#     return df
#
#
# def encode_categorical_features(df, categorical_features):
#     le = LabelEncoder()
#     for feature in categorical_features:
#         df[feature] = le.fit_transform(df[feature])
#     return df
#
#
# def split_and_normalize_data(df):
#     X = df.drop('income', axis=1)
#     y = df['income']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     return X_train, X_test, y_train, y_test
#
#
# def create_and_train_model(X_train, y_train):
#     # Start monitoring with CodeCarbon
#     tracker = OfflineEmissionsTracker(
#         country_iso_code="ITA",
#         output_file=output_file,
#         output_dir=output_dir
#     )
#
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#     if tf.test.gpu_device_name():
#         print('GPU found')
#     else:
#         print("No GPU found")
#
#     tracker.start()
#     # Define the model
#     model = Sequential()
#     model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     # Train the model 50 times
#     for _ in range(50):
#         model.fit(X_train, y_train, epochs=10, batch_size=32)
#     tracker.stop()
#
#     return model
#
#
# def calculate_metrics(y_test, y_pred):
#     print("Precision:", precision_score(y_test, y_pred))
#     print("Recall:", recall_score(y_test, y_pred))
#     print("F1 Score:", f1_score(y_test, y_pred))
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#
#     df_metrics_accuracy = pd.DataFrame({
#         'Accuracy': [accuracy_score(y_test, y_pred)],
#         'Precision': [precision_score(y_test, y_pred)],
#         'Recall': [recall_score(y_test, y_pred)],
#         'F1-score': [f1_score(y_test, y_pred)]
#     })
#
#     df_metrics_accuracy = df_metrics_accuracy.melt(var_name='Metric', value_name='Value')
#     fig_accuracy = px.box(df_metrics_accuracy, x='Metric', y='Value')
#     fig_accuracy.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
#     add_interactivity(fig_accuracy)
#     accuracy_graph = fig_accuracy.to_json()
#
#     return precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), accuracy_graph
#
#
# def calculate_fairness_metrics(dataset, dataset_test, y_pred, unprivileged_groups, privileged_groups):
#     dataset_test_pred = dataset_test.copy()
#     dataset_test_pred.labels = y_pred
#     metric_test_bld = BinaryLabelDatasetMetric(dataset_test_pred, unprivileged_groups=unprivileged_groups,
#                                                privileged_groups=privileged_groups)
#     print("Mean Difference:", metric_test_bld.mean_difference())
#     classified_metric_test = ClassificationMetric(dataset_test, dataset_test_pred,
#                                                   unprivileged_groups=unprivileged_groups,
#                                                   privileged_groups=privileged_groups)
#     print("Equal Opportunity Difference:", classified_metric_test.equal_opportunity_difference())
#     print("Average Odds Difference:", classified_metric_test.average_odds_difference())
#
#     return metric_test_bld.mean_difference(), classified_metric_test.equal_opportunity_difference(), classified_metric_test.average_odds_difference()
#
#
# def calculate_additional_metrics(y_true, y_pred):
#     mean = np.mean(y_pred)
#     median = np.median(y_pred)
#     variance = np.var(y_pred)
#     overall_accuracy = accuracy_score(y_true, y_pred)
#     return mean, median, variance, overall_accuracy
#
#
# # Utilizzo delle funzioni
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
# df = load_and_preprocess_data(url)
# categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
#                         'native-country']
# df = encode_categorical_features(df, categorical_features)
# X_train, X_test, y_train, y_test = split_and_normalize_data(df)
# model, energy_consumption_graph, combined_energy_graph = create_and_train_model(X_train, y_train)
# y_pred = (model.predict(X_test) > 0.5).astype("int32")
# calculate_metrics(y_test, y_pred)
# dataset = StandardDataset(df, label_name='income', favorable_classes=[1], protected_attribute_names=['sex'],
#                           privileged_classes=[[1]])
# dataset_train, dataset_test = dataset.split([0.8], shuffle=True)
# X_train = dataset_train.features
# y_train = dataset_train.labels.ravel()
# model.fit(X_train, y_train, epochs=10, batch_size=32)
# X_test = dataset_test.features
# y_test = dataset_test.labels.ravel()
# y_pred = (model.predict(X_test) > 0.5).astype("int32")
# precision_base, recall_base, f1_score_base, accuracy_graph = calculate_metrics(y_test, y_pred)
# mean_base, equal_opportunity_difference, average_odds_difference = calculate_fairness_metrics(dataset, dataset_test,
#                                                                                               y_pred, [{'sex': 0}],
#                                                                                               [{'sex': 1}])
# mean, median, variance, overall_accuracy = calculate_additional_metrics(y_test, y_pred)
#
# # Creating a DataFrame with your metrics.
# df_metrics = pd.DataFrame({
#     'Mean difference': [mean_base],
#     'Equal opportunity difference': [equal_opportunity_difference],
#     'Average odds difference': [average_odds_difference],
#     'Mean of mean differences': [mean],
#     'Median of mean differences': [median],
#     'Variance of mean differences': [variance],
#     'Overall 0,1': [overall_accuracy]
# })
#
# df_metrics = df_metrics.melt(var_name='Metric', value_name='Value')
# fig_metrics = px.box(df_metrics, x='Metric', y='Value')
# fig_metrics.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
# metrics_graph = fig_metrics.to_json()
#
# metrics = {
#     'name': 'modelAdult',
#     'accuracy':accuracy_score,
#     'precision_base': precision_base,
#     'recall_base': recall_base,
#     'f1_score_base': f1_score_base,
#     'mean_base': mean_base,
#     'mean_difference_base': mean,
#     'equal_opportunity_difference_base': equal_opportunity_difference,
#     'average_odds_difference_base': average_odds_difference
#
#
# }
#



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from aif360.datasets import StandardDataset
# from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import zipfile
# import urllib.request
# import os
#
#
# def load_and_preprocess_data(url):
#     zip_path, _ = urllib.request.urlretrieve(url)
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         csv_path = zip_ref.extract('bank.csv')
#     df = pd.read_csv(csv_path, sep=';')
#     df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
#     return df
#
#
# def encode_categorical_features(df, categorical_features):
#     le = LabelEncoder()
#     for feature in categorical_features:
#         df[feature] = le.fit_transform(df[feature])
#     return df
#
#
# def split_and_normalize_data(df):
#     X = df.drop('y', axis=1)
#     y = df['y']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     return X_train, X_test, y_train, y_test
#
#
# def create_and_train_model(X_train, y_train):
#     # Start monitoring with CodeCarbon
#     tracker = OfflineEmissionsTracker(
#         country_iso_code="ITA",
#         output_file=output_file,
#         output_dir=output_dir
#     )
#
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#     if tf.test.gpu_device_name():
#         print('GPU found')
#     else:
#         print("No GPU found")
#
#     tracker.start()
#
#     model = Sequential()
#     model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     for _ in range(50):
#         model.fit(X_train, y_train, epochs=10, batch_size=32)
#
#     tracker.stop()
#
#     # Check if the file has been created
#     csv_file_path = os.path.join(output_dir, output_file)
#     if os.path.isfile(csv_file_path):
#         print(f"CSV file created: {csv_file_path}")
#
#         with open(csv_file_path, 'r') as csvfile:
#             csv_reader = csv.DictReader(csvfile, delimiter=',')
#             for row in csv_reader:
#                 print(row)
#
#     else:
#         print(f"CSV file not found: {csv_file_path}")
#
#     dataResults = []
#     # Check if the file has been created
#     csv_file_path = os.path.join(output_dir, output_file)
#     if os.path.isfile(csv_file_path):
#         print(f"CSV file created: {csv_file_path}")
#
#         with open(csv_file_path, 'r') as csvfile:
#             csv_reader = csv.DictReader(csvfile, delimiter=',')
#             for row in csv_reader:
#                 # Convert kWh to Joules
#                 energy_in_joules = float(
#                     row['energy_consumed']) * 3600000  # Conversion factor (1 kWh = 3600000 J)
#                 ram_energy_in_joules = float(row['ram_energy']) * 3600000
#                 cpu_energy_in_joules = float(row['cpu_energy']) * 3600000
#                 gpu_energy_in_joules = float(row['gpu_energy']) * 3600000
#
#                 dataResults.append({
#                     'timestamp': row['timestamp'],
#                     'run_id': row['run_id'],
#                     'energy_consumed': energy_in_joules,
#                     'duration': row['duration'],
#                     'ram_energy': ram_energy_in_joules,
#                     'cpu_energy': cpu_energy_in_joules,
#                     'gpu_energy': gpu_energy_in_joules
#                 })
#     else:
#         print(f"CSV file not found: {csv_file_path}")
#
#     energy_consumption_data = [row['energy_consumed'] for row in dataResults]
#     fig = px.violin(energy_consumption_data, title="Energy Consumption Distribution")
#     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
#     fig.update_layout(
#         xaxis_title="Energy Consumed (Joules)",
#         yaxis_title="Count",
#         violingroupgap=0,
#     )
#     add_interactivity(fig)
#
#     combined_energy_data = []
#     for row in dataResults:
#         combined_energy_data.append([row['ram_energy'], row['cpu_energy'], row['gpu_energy']])
#
#     df_combined = pd.DataFrame(combined_energy_data, columns=["RAM", "CPU", "GPU"])
#
#     figEnergy = px.violin(df_combined.melt(var_name='Type', value_name='Energy'), y="Energy", x="Type",
#                           box=True, title="Energy Consumption Distribution (RAM,CPU,GPU)")
#
#     figEnergy.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
#     add_interactivity(figEnergy)
#
#     energy_consumption_graph = fig.to_json()
#     combined_energy_graph = figEnergy.to_json()
#
#     return model, energy_consumption_graph, combined_energy_graph
#
#
# def calculate_metrics(y_test, y_pred):
#     print("Precision:", precision_score(y_test, y_pred))
#     print("Recall:", recall_score(y_test, y_pred))
#     print("F1 Score:", f1_score(y_test, y_pred))
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#
#     df_metrics_accuracy = pd.DataFrame({
#         'Accuracy': [accuracy_score(y_test, y_pred)],
#         'Precision': [precision_score(y_test, y_pred)],
#         'Recall': [recall_score(y_test, y_pred)],
#         'F1-score': [f1_score(y_test, y_pred)]
#     })
#
#     df_metrics_accuracy = df_metrics_accuracy.melt(var_name='Metric', value_name='Value')
#     fig_accuracy = px.box(df_metrics_accuracy, x='Metric', y='Value')
#     fig_accuracy.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
#     add_interactivity(fig_accuracy)
#     accuracy_graph = fig_accuracy.to_json()
#
#     return precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), accuracy_graph
#
#
# def calculate_fairness_metrics(df, y_pred, unprivileged_groups, privileged_groups):
#     dataset = StandardDataset(df, label_name='y', favorable_classes=[1], protected_attribute_names=['age'],
#                               privileged_classes=[[1]])
#     dataset_train, dataset_test = dataset.split([0.8], shuffle=True)
#     X_train = dataset_train.features
#     y_train = dataset_train.labels.ravel()
#     model.fit(X_train, y_train, epochs=10, batch_size=32)
#     X_test = dataset_test.features
#     y_test = dataset_test.labels.ravel()
#     y_pred = (model.predict(X_test) > 0.5).astype("int32")
#     calculate_metrics(y_test, y_pred)
#     dataset_test_pred = dataset_test.copy()
#     dataset_test_pred.labels = y_pred
#     metric_test_bld = BinaryLabelDatasetMetric(dataset_test_pred, unprivileged_groups=unprivileged_groups,
#                                                privileged_groups=privileged_groups)
#     print("Mean Difference:", metric_test_bld.mean_difference())
#     classified_metric_test = ClassificationMetric(dataset_test, dataset_test_pred,
#                                                   unprivileged_groups=unprivileged_groups,
#                                                   privileged_groups=privileged_groups)
#     print("Equal Opportunity Difference:", classified_metric_test.equal_opportunity_difference())
#     print("Average Odds Difference:", classified_metric_test.average_odds_difference())
#     return metric_test_bld.mean_difference(), classified_metric_test.equal_opportunity_difference(), classified_metric_test.average_odds_difference()
#
#
# def calculate_additional_metrics(y_true, y_pred):
#     mean = np.mean(y_pred)
#     median = np.median(y_pred)
#     variance = np.var(y_pred)
#     overall_accuracy = accuracy_score(y_true, y_pred)
#     return mean, median, variance, overall_accuracy
#
#
# # Utilizzo delle funzioni
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
# df = load_and_preprocess_data(url)
# categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
# df = encode_categorical_features(df, categorical_features)
# model, energy_consumption_graph, combined_energy_graph = create_and_train_model(X_train, y_train)
# y_pred = (model.predict(X_test) > 0.5).astype("int32")
# precision_base, recall_base, f1_score_base, accuracy_graph = calculate_metrics(y_test, y_pred)
# # mean_base,equal_opportunity_difference,average_odds_difference=calculate_fairness_metrics(df, y_pred, [{'age': 0}], [{'age': 1}])
# mean_base, equal_opportunity_difference, average_odds_difference = 0, 0, 0
# mean, median, variance, overall_accuracy = calculate_additional_metrics(y_test, y_pred)
#
# # Creating a DataFrame with your metrics.
# df_metrics = pd.DataFrame({
#     'Mean difference': [mean_base],
#     'Equal opportunity difference': [equal_opportunity_difference],
#     'Average odds difference': [average_odds_difference],
#     'Mean of mean differences': [mean],
# })
#
# df_metrics = df_metrics.melt(var_name='Metric', value_name='Value')
# fig_metrics = px.box(df_metrics, x='Metric', y='Value')
# fig_metrics.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
# add_interactivity(fig_metrics)
# metrics_graph = fig_metrics.to_json()
#
# metrics = {
#     'name': 'modelBank',
#     'precision_base': precision_base,
#     'recall_base': recall_base,
#     'f1_score_base': f1_score_base,
#     'mean_base': mean_base,
#     'energy_consumption_graph': energy_consumption_graph,
#     'combined_energy_graph': combined_energy_graph,
#     'metrics_graph': metrics_graph,
#     'accuracy_graph': accuracy_graph
# }
#
# all_metrics = {}
#
# all_metrics['modelBank'] = metrics
#
# with open('../metrics.json', 'a') as f:
#     json.dump(all_metrics, f)
#
