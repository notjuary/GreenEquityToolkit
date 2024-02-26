import json

import pandas as pd
import yaml
from django.http import HttpResponse
from django.shortcuts import render, HttpResponseRedirect
from codecarbon import EmissionsTracker

from django.urls import reverse
from django.contrib import messages
from codecarbon import OfflineEmissionsTracker
from sostenibilita.forms import FileTraniningForm, ModelTrainedForm
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
import onnxruntime as ort
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

output_dir = '.'
output_file = 'emissions.csv'

#Homepage loading function
def index(request):
    return render(request, 'index.html')

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


#Function for modelView.html form processing for tracking pre-trained models
def machineLearningTraining(request):
    countries = request.session.get('countries', None)

    if request.method == 'POST':
        form = ModelTrainedForm(request.POST, countries=countries)

        if form.is_valid():
            print(form.cleaned_data)
            # Selection from available pre-trained models.
            modelTypeTrained=form.cleaned_data['modelTypeTrained']
            #Selecting the preloaded ISO code of the downloaded JSON file from CodeCarbon's GitHub repo
            countryIsoCode=form.cleaned_data['countryIsoCode']
            print(str(countryIsoCode))
            print(form.cleaned_data)

            if modelTypeTrained=='bert-base-uncased':
                try:
                    #Get tokenizer and the pre-trained model.
                    tokenizer=AutoTokenizer.from_pretrained(modelTypeTrained)
                    model=AutoModelForSequenceClassification.from_pretrained(modelTypeTrained)
                except Exception as e:
                    errore=f"Error when loading model or tokenizer: {str(e)}"
                    messages.error(request,errore)
                    print(errore)
                    context = {
                        'errore': errore,
                    }
                    return render(request, '404.html',context)
            elif modelTypeTrained=='distilbert-base-uncased':
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
                #Load dataset GLUE MRPC
                dataset=load_dataset('glue','mrpc')
                encoded_dataset=dataset.map(lambda examples: tokenizer(examples['sentence1'],examples['sentence2'],truncation=True,padding='max_length'),batched=True)
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
                training_args = TrainingArguments("test_trainer", per_device_train_batch_size=16, per_device_eval_batch_size=64,
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
                                'timestamp':row['timestamp'],
                                'run_id':row['run_id'],
                                'energy_consumed': row['energy_consumed'],
                                'duration': row['duration'],
                                'ram_energy': row['ram_energy'],
                            })
                else:
                    print(f"CSV file not found: {csv_file_path}")

                messages.success(request, "Processing successfully completed!")
                return render(request, 'results.html', {'data': dataResults})

        else:
            print(form.errors)

    return render(request, '404.html')

#Function for processing the trainingFile.html form for tracking with CodeCarbon the models uploaded by the user
def uploadFile(request):
    countries = request.session.get('countries', None)

    if request.method == 'POST':
        form = FileTraniningForm(request.POST, request.FILES,countries=countries)
        if form.is_valid():
            file = form.cleaned_data['fileTraining']
            dataFile = form.cleaned_data['dataFile']
            modelType = form.cleaned_data['modelType']
            countryIsoCode = form.cleaned_data['countryIsoCode']
            print(str(countryIsoCode))
            print(form.cleaned_data)

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
                if modelType == 'tensorflow':
                    model = tensorflow.keras.models.load_model(temp_file_path)
                elif modelType == 'pytorch':
                    model = torch.load(temp_file_path)
                elif modelType == 'onnx':
                    model = onnx.load(temp_file_path)
                    sess = ort.InferenceSession(model.SerializeToString())
                else:
                    raise ValueError("Model type not supported")
                print("The model was loaded correctly.")
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
                errore = "Error while reading the data file: "+ str(e)
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
                if modelType=='sklearn' and isinstance(model, sklearn.base.BaseEstimator):
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
                return render(request, '404.html',context)
            finally:
                #Stop emissioni del tracker
                tracker.stop()

            dataResults=[]
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

            messages.success(request, "Processing successfully completed!")
            os.remove(temp_file_path) # Remove the temporary file
            return render(request, 'results.html',{'data':dataResults})
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


