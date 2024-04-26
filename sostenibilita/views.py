import json
import numpy as np
import joblib
import pandas as pd
import yaml
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from django.http import HttpResponse, FileResponse
from django.shortcuts import render, HttpResponseRedirect, redirect
from codecarbon import EmissionsTracker
from pprint import pprint
from django.urls import reverse
from django.contrib import messages
from codecarbon import OfflineEmissionsTracker
from sklearn.metrics._classification import precision_score, recall_score, f1_score
from tensorboard.compat import tf
from torch import device
from sostenibilita.forms import FileTraniningForm, ModelTrainedForm, FileSocialForm, \
    UploadDatasetForm, SelectProtectedAttributesForm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, \
    glue_compute_metrics, training_args, DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
import sklearn
import tensorflow
import torch
import os
import tempfile
import onnx
import io
import csv
import onnxruntime
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import plotly.express as px
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import uuid
from sostenibilita.modelManager import ModelManager

output_dir = '.'
output_file = 'emissions.csv'

#Class to manage added models, getting model cards with results to manage
model_manager = ModelManager()

# Homepage loading function
def index(request):
    return render(request, 'index.html')


# Function for managing the download of the emissions.csv file processed by CodeCarbon
def downloadFileEmission(request):
    response = FileResponse(open('emissions.csv', 'rb'), content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="emission.csv"'

    return response


# Function to manage dictionary keys
def clean_dict_keys(data):
    cleaned_data = []
    for item in data:
        cleaned_item = {key.strip().lower().replace('-', ''): value for key, value in item.items()}
        cleaned_data.append(cleaned_item)
    return cleaned_data


def split_string(data, separator):
    split_data = []
    for item in data:
        split_item = {key: value.split(separator) if isinstance(value, str) else value for key, value in item.items()}
        split_data.append(split_item)
    return split_data


# Function for redirecting to the pre-trained models section of social sustainability
def redirectModelsPreaddestratedSocial(request):
    df1 = pd.read_csv('isBiased.csv', )
    data = df1.to_dict('records')
    df2 = pd.read_csv('final_bias_categories.csv')
    data2 = df2.to_dict('records')

    merged_data = []
    merged_model_names = []

    for item1 in data:
        for item2 in data2:
            if item1['Model_Name'] == item2['Model_Name']:
                merged_item = {**item1, **item2}
                merged_data.append(merged_item)
                merged_model_names.append(item1['Model_Name'])

    data = [item for item in data if item['Model_Name'] not in merged_model_names]

    merged_data = clean_dict_keys(merged_data)
    merged_data = split_string(merged_data, ', ')
    data=clean_dict_keys(data)

    return render(request, 'modelsPreaddestratedSocial.html', {'data': merged_data,'model_not_available':data,})


# Redirect function for loading the dataset
def redirectUploadDataset(request):
    form = UploadDatasetForm()
    return render(request, 'uploadDataset.html', {'form': form})


# Function that reads the contents of the dataset and identifies protected attributes
def uploadDataset(request):
    if request.method == 'POST':
        form = UploadDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            use_default_model = form.cleaned_data['useDefaultModel']
            request.session['useDefaultModel'] = use_default_model
            print(use_default_model)
            if use_default_model:
                return trainingFile(request)
            else:
                # Save dataset in session and columns
                dataFile = form.cleaned_data['datasetFile']

                # Verify data format
                if not dataFile.name.endswith(('.csv', '.xlsx', '.xls', '.json', '.yaml')):
                    errore = "Data file type not supported"
                    messages.error(request, errore)
                    print(errore)
                    context = {
                        'errore': errore,
                    }
                    return render(request, '404.html', context)

                    # Read data from the dataset
                try:
                    if dataFile.name.endswith('.csv'):
                        data = pd.read_csv(io.StringIO(dataFile.read().decode('utf-8')))
                        # Setting dataset columns in checkboxes to define which protected attributes to consider
                        columns = list(data.columns)
                        print(columns)
                        request.session['columns'] = columns
                        uploadForm = FileSocialForm()
                        select_form = SelectProtectedAttributesForm(columns=columns)
                        return render(request, 'socialUpdateFile.html',
                                      {'uploadForm': uploadForm, 'selectForm': select_form})


                    elif dataFile.name.endswith('.xlsx') or dataFile.name.endswith('.xls'):
                        data = pd.read_excel(io.BytesIO(dataFile.read()))
                        # Setting dataset columns in checkboxes to define which protected attributes to consider
                        columns = list(data.columns)
                        print(columns)
                        request.session['columns'] = columns
                        uploadForm = FileSocialForm()

                        select_form = SelectProtectedAttributesForm(columns=columns)
                        return render(request, 'socialUpdateFile.html',
                                      {'uploadForm': uploadForm, 'selectForm': select_form})

                    elif dataFile.name.endswith('.json'):
                        data = pd.read_json(io.StringIO(dataFile.read().decode('utf-8')))
                        # Setting dataset columns in checkboxes to define which protected attributes to consider
                        columns = list(data.columns)
                        print(columns)
                        uploadForm = FileSocialForm()
                        request.session['columns'] = columns
                        select_form = SelectProtectedAttributesForm(columns=columns)
                        return render(request, 'socialUpdateFile.html',
                                      {'uploadForm': uploadForm, 'selectForm': select_form})

                    elif dataFile.name.endswith('.yaml'):
                        data = pd.json_normalize(yaml.safe_load(io.StringIO(dataFile.read().decode('utf-8'))))
                        # Setting dataset columns in checkboxes to define which protected attributes to consider
                        columns = list(data.columns)
                        print(columns)
                        request.session['columns'] = columns
                        uploadForm = FileSocialForm()
                        select_form = SelectProtectedAttributesForm(columns=columns)
                        return render(request, 'socialUpdateFile.html',
                                      {'uploadForm': uploadForm, 'selectForm': select_form})

                    else:
                        raise ValueError("Data file type not supported")

                    print("The data file was read correctly.")
                except Exception as e:
                    errore = "Error while reading the data file: " + str(e)
                    messages.error(request, errore)
                    print(errore)
                    context = {
                        'errore': errore,
                    }
                    return render(request, '404.html', context)
        else:
            print(form.errors)
            return render(request,'404.html',{'errore':str(form.errors)})

    return render(request, '404.html')


# Redirection function for loading model and dataset files
def trainingFile(request):
    # Load the list of countries from the session, if available.
    countries = request.session.get('countries', None)

    dataLoad = False

    if countries is None:
        try:
            # Read the JSON file
            if os.path.exists('global_energy_mix.json'):
                with open('global_energy_mix.json', 'r') as f:
                    data = json.load(f)
                    dataLoad = True
        except Exception as e:
            data = None
            errore = f"Error opening Json file: {str(e)}"
            messages.error(request, errore)
            print(errore)
            context = {
                'errore': errore,
            }
            return render(request, '404.html', context)

    if dataLoad:
        # Create a list of tuples for the select field.
        countries = [(key, data[key]['country_name']) for key in data]
        print(countries)

    # Save the list of countries in the session.
    request.session['countries'] = countries
    print(f"Value of session: "+str(request.session.get('useDefaultModel')))
    # Create an instance of form.
    form = FileTraniningForm(countries=countries, initial={'useDefaultModel': request.session.get('useDefaultModel', False)})
    # Create an instance of form.
    return render(request, 'trainingFile.html', {'form': form})


# Redirection function to the section for monitoring pre-trained models
def modelView(request):
    # Load the list of countries from the session, if available.
    countries = request.session.get('countries', None)

    dataLoad = False

    if countries is None:
        try:
            # Read the JSON file
            if os.path.exists('global_energy_mix.json'):
                with open('global_energy_mix.json', 'r') as f:
                    data = json.load(f)
                    dataLoad = True
        except Exception as e:
            data = None
            errore = f"Error opening Json file: {str(e)}"
            messages.error(request, errore)
            print(errore)
            context = {
                'errore': errore,
            }
            return render(request, '404.html', context)

    if dataLoad:
        # Create a list of tuples for the select field.
        countries = [(key, data[key]['country_name']) for key in data]
        print(countries)

    # Save the list of countries in the session.
    request.session['countries'] = countries

    # Create an instance of form.
    form = ModelTrainedForm(countries=countries)
    # Create an instance of form.
    return render(request, 'trainingModel.html', {'form': form})


# Function for defining the metrics of models pre-trained by Hugging Face
def compute_metrics(preds, labels):
    predictions = preds.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Wrapper function
def evaluate(eval_pred):
    pred = eval_pred.predictions
    labels = eval_pred.label_ids
    return compute_metrics(pred, labels)


# Function for trainingModel.html form processing for tracking pre-trained models
def machineLearningTraining(request):
    countries = request.session.get('countries', None)

    if request.method == 'POST':
        form = ModelTrainedForm(request.POST, countries=countries)

        if form.is_valid():
            print(form.cleaned_data)
            # Selection from available pre-trained models.
            modelTypeTrained = form.cleaned_data['modelTypeTrained']
            # Selecting the preloaded ISO code of the downloaded JSON file from CodeCarbon's GitHub repo
            countryIsoCode = form.cleaned_data['countryIsoCode']
            # Select where the model should be run whether CPU or GPU
            inferenceDevice = form.cleaned_data['inferenceDevice']
            # Selection of the number of inferences to be made
            numInferences = form.cleaned_data['numInferences']

            try:
                if numInferences <= 0:
                    raise ValueError("The number of inferences can either be nonnegative or zero")
            except Exception as e:
                errore = "Error during data verification and validity: " + str(e)
                messages.error(request, errore)
                print(errore)

                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)


            try:
                    # Get tokenizer and the pre-trained model
                    tokenizer = AutoTokenizer.from_pretrained(modelTypeTrained)
                    model = AutoModelForSequenceClassification.from_pretrained(modelTypeTrained)
                    model_device = device('cpu') if inferenceDevice == 'CPU' else device('cuda')
                    model.to(model_device)
            except Exception as e:
                    errore = f"Error while loading model or tokenizer: {str(e)}"
                    messages.error(request, errore)
                    print(errore)
                    context = {
                        'errore': errore,
                    }
                    return render(request, '404.html', context)

            try:
                # Load dataset GLUE MRPC
                dataset = load_dataset('glue', 'mrpc')
                encoded_dataset = dataset.map(
                    lambda examples: tokenizer(examples['sentence1'], examples['sentence2'], truncation=True,
                                               padding='max_length'), batched=True)
            except Exception as e:
                errore = f"Error while loading dataset: {str(e)}"
                messages.error(request, errore)
                print(errore)
                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)

                # Start monitoring with CodeCarbon
            tracker = OfflineEmissionsTracker(
                country_iso_code=countryIsoCode,
                output_file=output_file,
                output_dir=output_dir
            )

            tracker.start()

            try:
                # Train the model on the dataset
                training_args = TrainingArguments("test_trainer", per_device_train_batch_size=16,
                                                  per_device_eval_batch_size=64,
                                                  num_train_epochs=numInferences, weight_decay=0.01,
                                                  evaluation_strategy="epoch")
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=encoded_dataset["train"],
                    eval_dataset=encoded_dataset["validation"],
                    compute_metrics=evaluate  # Use the wrapper function
                )

                trainer.train()

            except Exception as e:
                errore = f"Error during model execution: {str(e)}"
                messages.error(request, errore)
                print(errore)
                tracker.stop()
                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)

            finally:
                # Stop emissioni del tracker
                tracker.stop()

                dataResults = []
                # Check if the file has been created
                csv_file_path = os.path.join(output_dir, output_file)
                if os.path.isfile(csv_file_path):
                    print(f"CSV file created: {csv_file_path}")

                    with open(csv_file_path, 'r') as csvfile:
                        csv_reader = csv.DictReader(csvfile, delimiter=',')
                        for row in csv_reader:
                            print(row)

                            # Convert kWh to Joules
                            energy_in_joules = float(
                                row['energy_consumed']) * 3600000  # Conversion factor (1 kWh = 3600000 J)
                            ram_energy_in_joules = float(row['ram_energy']) * 3600000
                            cpu_energy_in_joules = float(row['cpu_energy']) * 3600000
                            gpu_energy_in_joules = float(row['gpu_energy']) * 3600000

                            dataResults.append({
                                'timestamp': row['timestamp'],
                                'run_id': row['run_id'],
                                'energy_consumed': energy_in_joules,
                                'duration': row['duration'],
                                'ram_energy': ram_energy_in_joules,
                                'cpu_energy': cpu_energy_in_joules,
                                'gpu_energy': gpu_energy_in_joules
                            })
                else:
                    print(f"CSV file not found: {csv_file_path}")

                energy_consumption_data = [row['energy_consumed'] for row in dataResults]
                fig = px.violin(energy_consumption_data, title="Energy Consumption Distribution")

                combined_energy_data = []
                for row in dataResults:
                    combined_energy_data.append([row['ram_energy'], row['cpu_energy'], row['gpu_energy']])

                df_combined = pd.DataFrame(combined_energy_data, columns=["RAM", "CPU", "GPU"])

                figEnergy = px.violin(df_combined.melt(var_name='Type', value_name='Energy'), y="Energy", x="Type",
                                      box=True, title="Energy Consumption Distribution (RAM,CPU,GPU)")

                # Optional customizations for the plot
                fig.update_layout(
                    xaxis_title="Energy Consumed (Joules)",
                    yaxis_title="Count",
                    violingroupgap=0
                )

                messages.success(request, "Processing successfully completed!")
                return render(request, 'results.html',
                              {'data': dataResults, 'fig': fig.to_json(), 'combined_fig_json': figEnergy.to_json()})
        else:
            print(form.errors)


    return render(request, '404.html')


#Wrapper function for creating the default template
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=10, activation='relu'))  # Adjust input_dim to 10
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def loadDefaultModel(request):
    if request.method == "POST":
        # Load the dataset
        data = load_diabetes()

        df = pd.DataFrame(data.data, columns=data.feature_names)

        df['target'] = data.target

        # Salva il DataFrame in un file CSV
        df.to_csv('diabetes_dataset.csv', index=False)

        # Standardize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(data.data)

        # Binarize the target
        y = np.where(data.target > data.target.mean(), 1, 0)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # create the model
        model = create_model()

        model.fit(X_train, y_train, epochs=50, batch_size=10)

        _, accuracy = model.evaluate(X_test, y_test)
        print('Accuracy: %.2f' % (accuracy * 100))
        predictions = model.predict(X_test)
        predictions = (predictions > 0.5).astype(int)  # Converti le probabilità in etichette di classe

        # Calculate accuracy
        precision = precision_score(y_test, predictions)
        print(f'Precision: {precision}')

        # Calculate the recall
        recall = recall_score(y_test, predictions)
        print(f'Recall: {recall}')

        # Calculate the F1 score
        f1 = f1_score(y_test, predictions)
        print(f'F1 Score: {f1}')

        # Calculate the average
        mean = np.mean(predictions)
        print(f'Mean: {mean}')

        # Calculate the median
        median = np.median(predictions)
        print(f'Median: {median}')

        # Calculate the variance
        variance = np.var(predictions)
        print(f'Variance: {variance}')

        # Calculates overall accuracy
        overall_accuracy = np.mean(predictions == y_test)
        print(f'Overall Accuracy: {overall_accuracy}')


        # Crea un DataFrame con i tuoi dati e le etichette
        df = pd.DataFrame(X_test, columns=data.feature_names)
        df['label'] = y_test

        # Calcola la mediana di 'sex' e 'age'
        sex_median = df['sex'].median()
        age_median = df['age'].median()

        # Crea nuove variabili binarie
        df['sex_above_median'] = np.where(df['sex'] > sex_median, 1, 0)
        df['age_above_median'] = np.where(df['age'] > age_median, 1, 0)

        # Converti il DataFrame in un BinaryLabelDataset
        dataset = BinaryLabelDataset(df=df, label_names=['label'], protected_attribute_names=['sex_above_median', 'age_above_median'])

        # Crea un altro BinaryLabelDataset per le tue previsioni
        dataset_pred = dataset.copy()
        dataset_pred.labels = model.predict(X_test).reshape(-1, 1)

        # Definisci i tuoi gruppi privilegiati e non privilegiati
        privileged_groups = [{'sex_above_median': 1,'age_above_median':1}]
        unprivileged_groups = [{'sex_above_median': 0,'age_above_median':0}]

        # Calcola le metriche
        metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)

        # Ora puoi accedere alle metriche di equità
        mean_difference = metric.mean_difference()
        equal_opportunity_difference = metric.equal_opportunity_difference()
        average_odds_difference = metric.average_odds_difference()

        print('Mean Difference:', mean_difference)
        print('Equal Opportunity Difference:', equal_opportunity_difference)
        print('Average Odds Difference:', average_odds_difference)

        # Get the parameters from the form
        country_iso_code = request.POST.get('countryIsoCode')
        num_inferences = int(request.POST.get('numInferences'))
        inference_device = request.POST.get('inferenceDevice')

        # Start monitoring with CodeCarbon
        tracker = OfflineEmissionsTracker(country_iso_code=country_iso_code,output_file=output_file,
                output_dir=output_dir)
        tracker.start()

        # Set the device for the model
        device = 'cuda' if inference_device == 'gpu' and tf.test.is_gpu_available() else 'cpu'

        # Run the model for num_inferences times
        with tf.device(device):
            for _ in range(num_inferences):
                predictions = model.predict(X_test)  # Replace X_test with your actual test data

        # Stop monitoring
        tracker.stop()

        # Check if the file has been created
        csv_file_path = os.path.join(output_dir, output_file)
        if os.path.isfile(csv_file_path):
            print(f"CSV file created: {csv_file_path}")

            with open(csv_file_path, 'r') as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter=',')
                for row in csv_reader:
                    print(row)

        else:
            print(f"CSV file not found: {csv_file_path}")

        dataResults = []
        # Check if the file has been created
        csv_file_path = os.path.join(output_dir, output_file)
        if os.path.isfile(csv_file_path):
            print(f"CSV file created: {csv_file_path}")

            with open(csv_file_path, 'r') as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter=',')
                for row in csv_reader:
                    print(row)

                    # Convert kWh to Joules
                    energy_in_joules = float(
                        row['energy_consumed']) * 3600000  # Conversion factor (1 kWh = 3600000 J)
                    ram_energy_in_joules = float(row['ram_energy']) * 3600000
                    cpu_energy_in_joules = float(row['cpu_energy']) * 3600000
                    gpu_energy_in_joules = float(row['gpu_energy']) * 3600000

                    dataResults.append({
                        'timestamp': row['timestamp'],
                        'run_id': row['run_id'],
                        'energy_consumed': energy_in_joules,
                        'duration': row['duration'],
                        'ram_energy': ram_energy_in_joules,
                        'cpu_energy': cpu_energy_in_joules,
                        'gpu_energy': gpu_energy_in_joules
                    })
        else:
            print(f"CSV file not found: {csv_file_path}")

        energy_consumption_data = [row['energy_consumed'] for row in dataResults]
        fig = px.violin(energy_consumption_data, title="Energy Consumption Distribution")

        combined_energy_data = []
        for row in dataResults:
            combined_energy_data.append([row['ram_energy'], row['cpu_energy'], row['gpu_energy']])

        df_combined = pd.DataFrame(combined_energy_data, columns=["RAM", "CPU", "GPU"])

        figEnergy = px.violin(df_combined.melt(var_name='Type', value_name='Energy'), y="Energy", x="Type",
                              box=True, title="Energy Consumption Distribution (RAM,CPU,GPU)")

        # Optional customizations for the plot
        fig.update_layout(
            xaxis_title="Energy Consumed (Joules)",
            yaxis_title="Count",
            violingroupgap=0
        )

        # Creating a DataFrame with your metrics.
        df_metrics = pd.DataFrame({
            'Mean difference': [mean_difference],
            'Equal opportunity difference': [equal_opportunity_difference],
            'Average odds difference': [average_odds_difference],
            'Mean of mean differences': [mean],
            'Median of mean differences': [median],
            'Variance of mean differences': [variance],
            'Overall 0,1': [overall_accuracy]
        })

        df_metrics = df_metrics.melt(var_name='Metric', value_name='Value')
        # Create the box plot with Plotly
        fig_metrics = px.box(df_metrics, x='Metric', y='Value')


        df_metrics_accuracy = pd.DataFrame({
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1-score': [f1]
        })

        df_metrics_accuracy = df_metrics_accuracy.melt(var_name='Metric', value_name='Value')
        fig_accuracy = px.box(df_metrics_accuracy, x='Metric', y='Value')


        energy_consumption_graph=fig.to_json()
        combined_energy_graph=figEnergy.to_json()
        metrics_graph=fig_metrics.to_json()
        accuracy_graph=fig_accuracy.to_json()

        model_manager.addModel(
            name='base_model',
            precision_base=precision,
            recall_base=recall,
            f1_score_base=f1,
            mean_base=mean,
            median_base=median,
            variance_base=variance,
            overall_accuracy_base=accuracy ,
            energy_consumption_graph=energy_consumption_graph,
            combined_energy_graph=combined_energy_graph,
            metrics_graph=metrics_graph,
            accuracy_graph=accuracy_graph,
        )

        return render(request, 'cardModels.html',{'manager_models':model_manager.models})

    return HttpResponse(request, '404.html',{'errore':'Error loading the default model template'})

# Function for processing the trainingFile.html form for tracking with CodeCarbon the models uploaded by the user
def uploadFile(request):
    countries = request.session.get('countries', None)
    sess = None

    if request.method == 'POST':
        form = FileTraniningForm(request.POST, request.FILES, countries=countries)
        if form.is_valid():
            use_default_model = form.cleaned_data['useDefaultModel']
            countryIsoCode = form.cleaned_data['countryIsoCode']
            numInferences = form.cleaned_data['numInferences']
            inferenceDevice = form.cleaned_data['inferenceDevice']

            # Number of inferences is not negative and is an integer
            if numInferences < 0 or not isinstance(numInferences, int):
                errore = "The number of inferences must be an integer and positive"
                messages.error(request, errore)
                print(errore)
                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)


            if use_default_model:
                return loadDefaultModel(request)
            else:
                file = form.cleaned_data['fileTraining']
                dataFile = form.cleaned_data['dataFile']


                # Verify data format
                if not dataFile.name.endswith(('.csv', '.xlsx', '.xls', '.json', '.yaml')):
                    errore = "Data file type not supported"
                    messages.error(request, errore)
                    print(errore)
                    context = {
                        'errore': errore,
                    }
                    return render(request, '404.html', context)

                # Save the file temporarily
                _, temp_file_path = tempfile.mkstemp()
                with open(temp_file_path, 'wb+') as temp_file:
                    for chunk in file.chunks():
                        temp_file.write(chunk)

                # Upload the template
                try:
                    model = joblib.load(temp_file_path)  # Load the model with joblib
                    modelType = None

                    # Determine the library of the model
                    if isinstance(model, sklearn.base.BaseEstimator):
                        modelType = 'sklearn'
                    elif isinstance(model, tensorflow.python.keras.engine.training.Model):
                        modelType = 'tensorflow'
                    elif isinstance(model, torch.nn.modules.module.Module):
                        modelType = 'pytorch'
                    elif isinstance(model, onnx.onnx_ml_pb2.ModelProto):
                        modelType = 'onnx'

                    else:
                        raise ValueError("Model type not supported")

                    print(f"The model was loaded correctly. Model type: {modelType}")
                except Exception as e:
                    errore = f"Error while loading the model: {str(e)}"
                    messages.error(request, errore)
                    print(errore)
                    context = {
                        'errore': errore,
                    }
                    return render(request, '404.html', context)

                # Read data from the file
                try:
                    if dataFile.name.endswith('.csv'):
                        data = pd.read_csv(io.StringIO(dataFile.read().decode('utf-8')))
                    elif dataFile.name.endswith('.xlsx') or dataFile.name.endswith('.xls'):
                        data = pd.read_excel(io.BytesIO(dataFile.read()))
                    elif dataFile.name.endswith('.json'):
                        data = pd.read_json(io.StringIO(dataFile.read().decode('utf-8')))
                    elif dataFile.name.endswith('.yaml'):
                        data = pd.json_normalize(yaml.safe_load(io.StringIO(dataFile.read().decode('utf-8'))))
                    else:
                        raise ValueError("Data file type not supported")
                    print("The data file was read correctly.")
                except Exception as e:
                    errore = "Error while reading the data file: " + str(e)
                    messages.error(request, errore)
                    print(errore)
                    context = {
                        'errore': errore,
                    }
                    return render(request, '404.html', context)

                    # Start monitoring with CodeCarbon
                tracker = OfflineEmissionsTracker(
                    country_iso_code=countryIsoCode,
                    output_file=output_file,
                    output_dir=output_dir
                )

                tracker.start()

                # Set the device for the model
                device = 'cuda' if inferenceDevice == 'gpu' and torch.cuda.is_available() else 'cpu'

                # Run the model for numInferences times
                for _ in range(numInferences):
                    try:
                        if modelType == 'sklearn' and isinstance(model, sklearn.base.BaseEstimator):
                            predictions = model.predict(data)
                        elif modelType == 'tensorflow' and isinstance(model, tensorflow.python.keras.engine.training.Model):
                            with tf.device(device):
                                predictions = model.predict(data)
                        elif modelType == 'pytorch' and isinstance(model, torch.nn.modules.module.Module):
                            data_tensor = torch.from_numpy(data.values.astype('float32')).to(device)
                            model = model.to(device)
                            predictions = model(data_tensor)
                        elif modelType == 'onnx':
                            input_name = sess.get_inputs()[0].name
                            predictions = sess.run(None, {input_name: data.values.astype('float32')})
                        else:
                            raise ValueError("Model type not supported")
                    except Exception as e:
                        errore = "Error during model execution: " + str(e)
                        messages.error(request, errore)
                        print(errore)
                        tracker.stop()
                        context = {
                            'errore': errore,
                        }
                        return render(request, '404.html', context)
                    finally:
                        tracker.stop()

                dataResults = []
                # Check if the file has been created
                csv_file_path = os.path.join(output_dir, output_file)
                if os.path.isfile(csv_file_path):
                    print(f"CSV file created: {csv_file_path}")

                    with open(csv_file_path, 'r') as csvfile:
                        csv_reader = csv.DictReader(csvfile, delimiter=',')
                        for row in csv_reader:
                            print(row)

                            # Convert kWh to Joules
                            energy_in_joules = float(
                                row['energy_consumed']) * 3600000  # Conversion factor (1 kWh = 3600000 J)
                            ram_energy_in_joules = float(row['ram_energy']) * 3600000
                            cpu_energy_in_joules = float(row['cpu_energy']) * 3600000
                            gpu_energy_in_joules = float(row['gpu_energy']) * 3600000

                            dataResults.append({
                                'timestamp': row['timestamp'],
                                'run_id': row['run_id'],
                                'energy_consumed': energy_in_joules,
                                'duration': row['duration'],
                                'ram_energy': ram_energy_in_joules,
                                'cpu_energy': cpu_energy_in_joules,
                                'gpu_energy': gpu_energy_in_joules
                            })
                else:
                    print(f"CSV file not found: {csv_file_path}")

                energy_consumption_data = [row['energy_consumed'] for row in dataResults]
                fig = px.violin(energy_consumption_data, title="Energy Consumption Distribution")

                combined_energy_data = []
                for row in dataResults:
                    combined_energy_data.append([row['ram_energy'], row['cpu_energy'], row['gpu_energy']])

                df_combined = pd.DataFrame(combined_energy_data, columns=["RAM", "CPU", "GPU"])

                figEnergy = px.violin(df_combined.melt(var_name='Type', value_name='Energy'), y="Energy", x="Type",
                                      box=True, title="Energy Consumption Distribution (RAM,CPU,GPU)")

                # Optional customizations for the plot
                fig.update_layout(
                    xaxis_title="Energy Consumed (Joules)",
                    yaxis_title="Count",
                    violingroupgap=0
                )

                energy_consumption_graph = fig.to_json()
                combined_energy_graph = figEnergy.to_json()

                messages.success(request, "Processing successfully completed!")
                unique_id = uuid.uuid4()

                model_manager.addModel(
                    name=unique_id,
                    precision_base=None,
                    recall_base=None,
                    f1_score_base=None,
                    mean_base=None,
                    median_base=None,
                    variance_base=None,
                    overall_accuracy_base=None,
                    energy_consumption_graph=energy_consumption_graph,
                    combined_energy_graph=combined_energy_graph,
                    metrics_graph=None,
                    accuracy_graph=None,
                )

                os.remove(temp_file_path)  # Remove the temporary file
                return render(request, 'cardModels.html', {'manager_models': model_manager.models})

        else:
            print(form.errors)
            errore = str(form.errors)
            messages.error(request, errore)
            print(errore)
            context = {
                'errore': errore,
            }
            return render(request, '404.html', context)

    else:
        return render(request, 'trainingFile.html')


# Function for monitoring social sustainability using aif360 with identified metrics
def uploadFileSocial(request):


    sess = None

    if request.method == 'POST':
        print(request.POST)
        form = FileSocialForm(request.POST, request.FILES)
        formAttributes = SelectProtectedAttributesForm(request.POST, columns=request.session.get('columns'))

        if form.is_valid() and formAttributes.is_valid():

            file = form.cleaned_data['fileModel']
            dataFile = form.cleaned_data['datasetFile']
            labelAttribute = formAttributes.cleaned_data['attribute']
            print(f"Selected label attribute: {labelAttribute}")
            protectedAttributes = formAttributes.cleaned_data['protected_attributes']
            print(f"Selected protected attribute: {protectedAttributes}")

            # verification of protected attribute selections and labels
            if not (protectedAttributes and labelAttribute):
                errore = "The fields of the protected attribute column and label column cannot be empty"
                messages.error(request, errore)
                print(errore)
                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)

            # Verify data format
            if not dataFile.name.endswith(('.csv', '.xlsx', '.xls', '.json', '.yaml')):
                errore = "Data file type not supported"
                messages.error(request, errore)
                print(errore)
                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)

            # Save the file temporarily
            _, temp_file_path = tempfile.mkstemp()
            with open(temp_file_path, 'wb+') as temp_file:
                for chunk in file.chunks():
                    temp_file.write(chunk)

            # Upload the template
            try:
                model = joblib.load(temp_file_path)  # Load the model with joblib
                modelType = None

                # Determine the library of the model
                if isinstance(model, sklearn.base.BaseEstimator):
                    modelType = 'sklearn'
                elif isinstance(model, tensorflow.python.keras.engine.training.Model):
                    modelType = 'tensorflow'
                elif isinstance(model, torch.nn.modules.module.Module):
                    modelType = 'pytorch'
                elif isinstance(model, onnx.onnx_ml_pb2.ModelProto):
                    modelType = 'onnx'

                else:
                    raise ValueError("Model type not supported")

                print(f"The model was loaded correctly. Model type: {modelType}")
            except Exception as e:
                errore = f"Error while loading the model: {str(e)}"
                messages.error(request, errore)
                print(errore)
                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)

            # Read data from the file
            try:
                if dataFile.name.endswith('.csv'):
                    data = pd.read_csv(io.StringIO(dataFile.read().decode('utf-8')))
                elif dataFile.name.endswith('.xlsx') or dataFile.name.endswith('.xls'):
                    data = pd.read_excel(io.BytesIO(dataFile.read()))
                elif dataFile.name.endswith('.json'):
                    data = pd.read_json(io.StringIO(dataFile.read().decode('utf-8')))
                elif dataFile.name.endswith('.yaml'):
                    data = pd.json_normalize(yaml.safe_load(io.StringIO(dataFile.read().decode('utf-8'))))
                else:
                    raise ValueError("Data file type not supported")
                print("The data file was read correctly.")
            except Exception as e:
                errore = "Error while reading the data file: " + str(e)
                messages.error(request, errore)
                print(errore)
                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)

            # Get the number of model features.
            try:
                num_features_model = model.n_features_
            except AttributeError:
                print("Il modello non ha l'attributo 'n_features_'")
                num_features_model = None

            # Get the number of data features.
            num_features_data = data.shape[1]

            try:
                # Check if the number of features matches
                if num_features_model is not None and num_features_model != num_features_data:
                    raise ValueError(
                        f"Error: the number of model features ({num_features_model}) does not match the number of data features ({num_features_data})")
                else:
                    raise ValueError("The number of model features corresponds to the number of data features")
            except Exception as e:
                errore = "Error during data verification and validity: " + str(e)
                messages.error(request, errore)
                print(errore)

                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)

                # Run the model
                try:
                    if modelType == 'sklearn' and isinstance(model, sklearn.base.BaseEstimator):
                        predictions = model.predict(data)
                    elif modelType == 'tensorflow' and isinstance(model, tensorflow.python.keras.engine.training.Model):
                        predictions = model.predict(data)
                    elif modelType == 'pytorch' and isinstance(model, torch.nn.modules.module.Module):
                        data = torch.from_numpy(data.values.astype('float32'))  # Converti i dati in un tensore PyTorch
                        predictions = model(data)
                    elif modelType == 'onnx':
                        input_name = sess.get_inputs()[0].name
                        result = sess.run(None, {input_name: data.values.astype('float32')})
                        predictions = result[0]
                    else:
                        raise ValueError("Model type not supported")

                except Exception as e:
                    errore = "Error during model execution: " + str(e)
                    messages.error(request, errore)
                    print(errore)

                    context = {
                        'errore': errore,
                    }
                    return render(request, '404.html', context)

            try:
                # Verify that the DataFrame is not empty
                if data.empty:
                    raise ValueError("The data file is empty")

                # Verify that the DataFrame contains the protected attribute and the label
                if protectedAttributes not in data.columns or labelAttribute not in data.columns:
                    raise ValueError("The data file does not contain the necessary columns")

                # Checks that the protected attribute and label contain only valid values
                if not all(data[protectedAttributes].notna()) or not all(data[labelAttribute].notna()):
                    raise ValueError("The protected attribute or label contains invalid values")

                # Verify that the predictions are the same length as the true labels
                if len(predictions) != len(data[labelAttribute]):
                    raise ValueError("The predictions are not the same length as the true labels")

                # Verify that predictions contain only valid values
                if not all(np.isfinite(predictions)):
                    raise ValueError("The predictions contain invalid values")
            except Exception as e:
                errore = "Error during data verification and validity: " + str(e)
                messages.error(request, errore)
                print(errore)

                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)

            try:
                # Create an AIF360 Dataset object
                dataset = BinaryLabelDataset(df=data, label_names=[labelAttribute],
                                             protected_attribute_names=protectedAttributes)

                # Calculate equity metrics.
                metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{protectedAttributes: 0}],
                                                  privileged_groups=[{protectedAttributes: 1}])
                mean_difference = metric.mean_difference()

                # Calculate accuracy metrics.
                y_true = dataset.labels
                y_pred = model.predict(dataset.features)

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                dataset_pred = dataset.copy()
                dataset_pred.labels = y_pred
                classification_metric = ClassificationMetric(dataset, dataset_pred,
                                                             unprivileged_groups=[{protectedAttributes: 0}],
                                                             privileged_groups=[{protectedAttributes: 1}])
                equal_opportunity_difference = classification_metric.equal_opportunity_difference()
                average_odds_difference = classification_metric.average_odds_difference()

                # Calculate the mean, median, and variance for each group
                grouped = data.groupby(protectedAttributes)
                mean_differences = grouped.apply(lambda x: BinaryLabelDatasetMetric(
                    BinaryLabelDataset(df=x, label_names=[labelAttribute],
                                       protected_attribute_names=[protectedAttributes])).mean_difference())
                mean = mean_differences.mean()
                median = mean_differences.median()
                variance = mean_differences.var()

                # Calculate the overall 0,1 for each group
                overall_01 = grouped.apply(lambda x: BinaryLabelDatasetMetric(
                    BinaryLabelDataset(df=x, label_names=[labelAttribute],
                                       protected_attribute_names=[protectedAttributes])).num_positives() / len(x))

                # Calculate accuracy metrics.
                accuracy = accuracy_score(data[labelAttribute], predictions)
                precision = precision_score(data[labelAttribute], predictions)
                recall = recall_score(data[labelAttribute], predictions)
                f1 = f1_score(data[labelAttribute], predictions)


                # Print metrics
                print("Mean difference =", mean_difference)
                print("Equal opportunity difference =", equal_opportunity_difference)
                print("Average odds difference =", average_odds_difference)
                print("Accuracy =", accuracy)
                print("Precision =", precision)
                print("Recall =", recall)
                print("F1-score =", f1)
                print("Mean of mean differences:\n", mean)
                print("Median of mean differences:\n", median)
                print("Variance of mean differences:\n", variance)
                print("Overall 0,1:\n", overall_01)

                # Creating a DataFrame with your metrics.
                df_metrics = pd.DataFrame({
                    'Mean difference': [mean_difference],
                    'Equal opportunity difference': [equal_opportunity_difference],
                    'Average odds difference': [average_odds_difference],
                    'Mean of mean differences': [mean],
                    'Median of mean differences': [median],
                    'Variance of mean differences': [variance],
                    'Overall 0,1': [overall_01]
                })

                df_metrics = df_metrics.melt(var_name='Metric', value_name='Value')
                # Create the box plot with Plotly
                fig = px.box(df_metrics, x='Metric', y='Value')

                df_metrics_accuracy = pd.DataFrame({
                    'Accuracy': [accuracy],
                    'Precision': [precision],
                    'Recall': [recall],
                    'F1-score': [f1]
                })

                df_metrics_accuracy = df_metrics_accuracy.melt(var_name='Metric', value_name='Value')
                fig_accuracy = px.box(df_metrics_accuracy, x='Metric', y='Value')

                metrics_graph = fig.to_json()
                accuracy_graph = fig_accuracy.to_json()


                unique_id = uuid.uuid4()

                model_manager.addModel(
                    name=unique_id,
                    precision_base=precision,
                    recall_base=recall,
                    f1_score_base=f1,
                    mean_base=mean,
                    median_base=median,
                    variance_base=variance,
                    overall_accuracy_base=overall_01,
                    energy_consumption_graph=None,
                    combined_energy_graph=None,
                    metrics_graph=metrics_graph,
                    accuracy_graph=accuracy_graph,
                )

                return render(request, 'cardModels.html', {'manager_models': model_manager.models})

            except Exception as e:
                errore = "Error when calculating AIF360 metrics: " + str(e)
                messages.error(request, errore)
                print(errore)

                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)


        else:
            print(form.errors)
            errore = str(form.errors)
            context = {
                'errore': errore,
            }
            return render(request, '404.html', context)
    else:

        # Redirects to the page for loading the dataset
        form = UploadDatasetForm()
        return render(request, 'uploadDataset.html', {'form': form})
