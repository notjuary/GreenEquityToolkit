import json

import joblib
import pandas as pd
import plotly
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
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json

output_dir = '.'
output_file = 'emissions.csv'


# Homepage loading function
def index(request):
    return render(request, 'index.html')


# Function for managing the download of the emissions.csv file processed by CodeCarbon
def downloadFileEmission(request):
    response = FileResponse(open('emissions.csv', 'rb'), content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="emission.csv"'

    return response



# Redirect function for loading the dataset
def redirectUploadDataset(request):
    form = UploadDatasetForm()
    return render(request, 'uploadDataset.html', {'form': form})


# Function that reads the contents of the dataset and identifies protected attributes
def uploadDataset(request):
    if request.method == 'POST':
        form = UploadDatasetForm(request.POST, request.FILES)
        if form.is_valid():
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

    # Create an instance of form.
    form = FileTraniningForm(countries=countries)
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
    return render(request, 'modelView.html', {'form': form})


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


# Function for modelView.html form processing for tracking pre-trained models
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
            print(str(countryIsoCode))
            print(form.cleaned_data)

            if modelTypeTrained == 'bert-base-uncased':
                try:
                    # Get tokenizer and the pre-trained model.
                    tokenizer = AutoTokenizer.from_pretrained(modelTypeTrained)
                    model = AutoModelForSequenceClassification.from_pretrained(modelTypeTrained)
                except Exception as e:
                    errore = f"Error when loading model or tokenizer: {str(e)}"
                    messages.error(request, errore)
                    print(errore)
                    context = {
                        'errore': errore,
                    }
                    return render(request, '404.html', context)
            elif modelTypeTrained == 'distilbert-base-uncased':
                try:
                    # Get tokenizer and the pre-trained model
                    tokenizer = DistilBertTokenizer.from_pretrained(modelTypeTrained)
                    model = DistilBertForSequenceClassification.from_pretrained(modelTypeTrained)
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
                                                  num_train_epochs=1, weight_decay=0.01, evaluation_strategy="epoch")
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
                            dataResults.append({
                                'timestamp': row['timestamp'],
                                'run_id': row['run_id'],
                                'energy_consumed': row['energy_consumed'],
                                'duration': row['duration'],
                                'ram_energy': row['ram_energy'],
                                'cpu_energy': row['cpu_energy'],
                                'gpu_energy': row['gpu_energy']

                            })
                else:
                    print(f"CSV file not found: {csv_file_path}")

                df = pd.DataFrame(dataResults)

                for column in ['energy_consumed', 'duration', 'ram_energy', 'cpu_energy', 'gpu_energy']:
                    df[column] = df[column].astype(float)

                # Create violin plot
                plt.figure(figsize=(10, 6))
                sns.violinplot(data=df[['energy_consumed', 'ram_energy', 'cpu_energy', 'gpu_energy']])
                plt.title('Violin Plot of Energy Metrics')
                plt.ylabel('Energy consumed')

                new_folder_path = 'static/img'

                # Create the new folder if it does not already exist.
                os.makedirs(new_folder_path, exist_ok=True)

                # Save the image in the new folder.
                plt.savefig(os.path.join(new_folder_path, 'violin_plot.png'))

                messages.success(request, "Processing successfully completed!")
                return render(request, 'results.html', {'image_path': '/img/violin_plot.png'})
        else:
            print(form.errors)

    return render(request, '404.html')


# Function for processing the trainingFile.html form for tracking with CodeCarbon the models uploaded by the user
def uploadFile(request):
    countries = request.session.get('countries', None)
    sess = None

    if request.method == 'POST':
        form = FileTraniningForm(request.POST, request.FILES, countries=countries)
        if form.is_valid():
                file = form.cleaned_data['fileTraining']
                dataFile = form.cleaned_data['dataFile']
                countryIsoCode = form.cleaned_data['countryIsoCode']
                numInferences = form.cleaned_data['numInferences']
                inferenceDevice = form.cleaned_data['inferenceDevice']

                #Number of inferences is not negative and is an integer
                if numInferences <0 or not isinstance(numInferences, int):
                    errore = "The number of inferences must be an integer and positive"
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
                            dataResults.append({
                                'timestamp': row['timestamp'],
                                'run_id': row['run_id'],
                                'energy_consumed': row['energy_consumed'],
                                'duration': row['duration'],
                                'ram_energy': row['ram_energy'],
                                'cpu_energy':row['cpu_energy'],
                                'gpu_energy':row['gpu_energy']
                            })
                else:
                    print(f"CSV file not found: {csv_file_path}")

                energy_consumption_data = [pd.to_numeric(row['energy_consumed']) for row in dataResults]
                fig = px.violin(energy_consumption_data, title="Energy Consumption Distribution")

                combined_energy_data=[]
                for row in dataResults:
                    combined_energy_data.append([row['ram_energy'], row['cpu_energy'], row['gpu_energy']])

                df_combined = pd.DataFrame(combined_energy_data, columns=["RAM", "CPU", "GPU"])

                figEnergy = px.violin(df_combined.melt(var_name='Type', value_name='Energy'), y="Energy", x="Type", box=True, title="Energy Consumption Distribution (RAM,CPU,GPU)")

                # Optional customizations for the plot
                fig.update_layout(
                    xaxis_title="Energy Consumed (Joules)",
                    yaxis_title="Count",
                    violingroupgap=0
                )


                messages.success(request, "Processing successfully completed!")
                return render(request, 'results.html', {'data': dataResults, 'fig': fig.to_json(),'combined_fig_json': figEnergy.to_json()})



                #os.remove(temp_file_path)  # Remove the temporary file

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

                return render(request, 'resultsSocial.html')

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
