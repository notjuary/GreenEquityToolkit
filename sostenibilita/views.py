import json

import joblib
import pandas as pd
import yaml
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from django.http import HttpResponse
from django.shortcuts import render, HttpResponseRedirect
from codecarbon import EmissionsTracker

from django.urls import reverse
from django.contrib import messages
from codecarbon import OfflineEmissionsTracker
from sklearn.metrics._classification import precision_score, recall_score, f1_score

from sostenibilita.forms import FileTraniningForm, ModelTrainedForm, ModelTrainedSocialForm, FileSocialForm
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

output_dir = '.'
output_file = 'emissions.csv'

#Homepage loading function
def index(request):
    return render(request, 'index.html')

# Redirection function to the section for social monitoring pre-trained models
def modelsPreaddestratedSocial(request):
    # Create an instance of form.
    form=ModelTrainedSocialForm()
    return render(request, 'sustainabilitysocialemodel.html',{'form': form})

# Redirect function to the template for uploading files required for social sustainability monitoring
def modelsUploadFileSocial(request):
    # Create an instance of form.
    form=FileSocialForm()
    return render(request, 'socialUpdateFile.html',{'form': form})


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

    dataLoad=False

    if countries is None:
        try:
             # Read the JSON file
            if os.path.exists('global_energy_mix.json'):
                with open('global_energy_mix.json','r') as f:
                     data = json.load(f)
                     dataLoad=True
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

    #Create an instance of form.
    form=ModelTrainedForm(countries=countries)
    #Create an instance of form.
    return render(request, 'modelView.html', {'form': form})


# Function for defining the metrics of models pre-trained by Hugging Face
def compute_metrics(preds,labels):
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
                            })
                else:
                    print(f"CSV file not found: {csv_file_path}")

                for item in dataResults:
                    item['energy_consumed'] = pd.to_numeric(item['energy_consumed'], errors='coerce')

                # Creates bubble sort with metrics taken from CodeCarbon
                fig = px.scatter(
                    dataResults,
                    x="timestamp",
                    y="run_id",
                    size="energy_consumed",  # Size of bubbles represents energy consumed
                    color="duration",  # Color of bubbles represents duration
                    hover_data=["ram_energy"]  # Show RAM energy on hover
                )
                # Graph creation and forwarding to the template results in JSON format
                fig.update_layout(title_text="Emissions result",
                                  xaxis_title="Timestamp",
                                  yaxis_title="Run ID")

                messages.success(request, "Processing successfully completed!")
                return render(request, 'results.html', {'data': dataResults, 'fig': fig.to_json()})
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
                        })
            else:
                print(f"CSV file not found: {csv_file_path}")

            for item in dataResults:
                item['energy_consumed'] = pd.to_numeric(item['energy_consumed'], errors='coerce')

                # Creates bubble sort with metrics taken from CodeCarbon
            fig = px.scatter(
                dataResults,
                x="timestamp",
                y="run_id",
                size="energy_consumed",  # Size of bubbles represents energy consumed
                color="duration",  # Color of bubbles represents duration
                hover_data=["ram_energy"]  # Show RAM energy on hover
            )

            #Graph creation and forwarding to the template results in JSON format
            fig.update_layout(title_text="Emissions result",
                              xaxis_title="Timestamp",
                              yaxis_title="Run ID")

            messages.success(request, "Processing successfully completed!")
            return render(request, 'results.html', {'data': dataResults, 'fig': fig.to_json()})
            os.remove(temp_file_path)  # Remove the temporary file

        else:
            print(form.errors)
            errore = "Error loading file"
            messages.error(request, errore)
            print(errore)
            context = {
                'errore': errore,
            }
            return render(request, '404.html', context)
    else:
        return render(request, 'trainingFile.html')

# def modelTrainedSustainabilitySocial(request):
#     if request.method == 'POST':
#         form=ModelTrainedSocialForm(request.POST)
#         if form.is_valid():
#             modelTypeSocial = form.cleaned_data['modelTypeSocial']
#
#         if modelTypeSocial=='distilbert-base-uncased':
#             try:
#                 # Get tokenizer and the pre-trained model
#                 tokenizer = DistilBertTokenizer.from_pretrained(modelTypeSocial)
#                 model = DistilBertForSequenceClassification.from_pretrained(modelTypeSocial)
#             except Exception as e:
#                 errore = f"Error while loading model or tokenizer: {str(e)}"
#                 messages.error(request, errore)
#                 print(errore)
#                 context = {
#                     'errore': errore,
#                 }
#                 return render(request, '404.html', context)
#
#             try:
#                 # Load dataset reuters21578
#                 dataset = load_dataset('reuters21578')
#
#                 # Preprocess the data
#                 def preprocess_data(examples):
#                     # Tokenize the text
#                     encodings = tokenizer(examples['text'], truncation=True, padding=True)
#                     # Convert the labels to integers
#                     labels = [int(label) for label in examples['label']]
#                     # Add the labels to the encodings
#                     encodings['labels'] = labels
#                     return encodings
#
#                 # Preprocess the training and test dataset
#                 train_dataset = dataset['train'].map(preprocess_data, batched=True)
#                 test_dataset = dataset['test'].map(preprocess_data, batched=True)
#
#                 # Define the training arguments
#                 training_args = TrainingArguments(
#                     output_dir='./results',  # output directory for the training results
#                     num_train_epochs=3,  # total number of training epochs
#                     per_device_train_batch_size=16,  # batch size for training
#                     per_device_eval_batch_size=64,  # batch size for evaluation
#                     warmup_steps=500,  # number of warmup steps
#                     weight_decay=0.01,  # weight decay
#                     logging_dir='./logs',  # output directory for the logs
#                 )
#
#                 # Create the Trainer
#                 trainer = Trainer(
#                     model=model,  # the pre-trained model
#                     args=training_args,  # training arguments
#                     train_dataset=train_dataset,  # training dataset
#                     eval_dataset=test_dataset  # evaluation dataset
#                 )
#
#                 # Train the model
#                 trainer.train()
#
#                 # Get the predictions
#                 predictions = trainer.predict(test_dataset)
#
#                 # Convert the predictions and the test data into BinaryLabelDataset
#                 test_bld = BinaryLabelDataset(df=test_dataset, label_names=['your_label'],
#                                               protected_attribute_names=['your_protected_attribute'])
#                 predictions_bld = BinaryLabelDataset(df=predictions, label_names=['your_label'],
#                                                      protected_attribute_names=['your_protected_attribute'])
#
#                 # Create a ClassificationMetric
#                 metric = ClassificationMetric(test_bld, predictions_bld,
#                                               unprivileged_groups=[{'your_protected_attribute': 0}],
#                                               privileged_groups=[{'your_protected_attribute': 1}])
#
#                 # Calculate the metrics
#                 mean_difference = metric.mean_difference()
#                 equal_opportunity_difference = metric.equal_opportunity_difference()
#                 average_odds_difference = metric.average_odds_difference()
#
#                 # Calculate the accuracy metrics
#                 accuracy = accuracy_score(test_dataset.labels, predictions.labels)
#                 precision = precision_score(test_dataset.labels, predictions.labels)
#                 recall = recall_score(test_dataset.labels, predictions.labels)
#                 f1 = f1_score(test_dataset.labels, predictions.labels)
#
#                 # Print the metrics
#                 print("Mean difference =", mean_difference)
#                 print("Equal opportunity difference =", equal_opportunity_difference)
#                 print("Average odds difference =", average_odds_difference)
#                 print("Accuracy =", accuracy)
#                 print("Precision =", precision)
#                 print("Recall =", recall)
#                 print("F1-score =", f1)
#             except Exception as e:
#                 errore = f"Error while loading dataset: {str(e)}"
#                 messages.error(request, errore)
#                 print(errore)
#                 context = {
#                     'errore': errore,
#                 }
#                 return render(request, '404.html', context)


# Function for monitoring social sustainability using aif360 with identified metrics
def uploadFileSocial(request):
    sess = None
    
    if request.method == 'POST':
        form = FileSocialForm(request.POST)
        if form.is_valid():
            file = form.cleaned_data['fileModel']
            dataFile = form.cleaned_data['datasetFile']
            protectedAttribute = form.cleaned_data['protectedAttribute']
            labelAttribute = form.cleaned_data['labelAttribute']

            if not (protectedAttribute and labelAttribute):
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
                    predictions =result[0]
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
                if protectedAttribute not in data.columns or labelAttribute not in data.columns:
                    raise ValueError("The data file does not contain the necessary columns")

                # Checks that the protected attribute and label contain only valid values
                if not all(data[protectedAttribute].notna()) or not all(data[labelAttribute].notna()):
                    raise ValueError("The protected attribute or label contains invalid values")

                #Verify that the predictions are the same length as the true labels
                if len(predictions) != len(data[labelAttribute]):
                    raise ValueError("The predictions are not the same length as the true labels")

                # Verify that predictions contain only valid values
                if not all(np.isfinite(predictions)):
                    raise ValueError("The predictions contain invalid values")
            except Exception as e:
                errore="Error during data verification and validity: "+str(e)
                messages.error(request, errore)
                print(errore)

                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)

            try:
                # Create an AIF360 Dataset object
                dataset = BinaryLabelDataset(df=data, label_names=[labelAttribute], protected_attribute_names=[protectedAttribute])

                # Calculate equity metrics.
                metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{protectedAttribute: 0}],
                                                  privileged_groups=[{protectedAttribute: 1}])
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
                                                             unprivileged_groups=[{protectedAttribute: 0}],
                                                             privileged_groups=[{protectedAttribute: 1}])
                equal_opportunity_difference = classification_metric.equal_opportunity_difference()
                average_odds_difference = classification_metric.average_odds_difference()

                # Calculate accuracy metrics.
                accuracy = accuracy_score(data[labelAttribute], predictions)
                precision = precision_score(data[labelAttribute], predictions)
                recall = recall_score(data[labelAttribute], predictions)
                f1 = f1_score(data[labelAttribute], predictions)
            except Exception as e:
                errore = "Error when calculating AIF360 metrics: " + str(e)
                messages.error(request, errore)
                print(errore)

                context = {
                    'errore': errore,
                }
                return render(request, '404.html', context)

            # Print metrics
            print("Mean difference =", mean_difference)
            print("Equal opportunity difference =", equal_opportunity_difference)
            print("Average odds difference =", average_odds_difference)
            print("Accuracy =", accuracy)
            print("Precision =", precision)
            print("Recall =", recall)
            print("F1-score =", f1)
