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

def index(request):
    return render(request, 'index.html')

def trainingFile(request):
    return render(request, 'trainingFile.html')

def modelView(request):
    return render(request, 'modelView.html')

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

def evaluate(eval_pred):
    pred = eval_pred.predictions
    labels = eval_pred.label_ids
    return compute_metrics(pred, labels)


def machineLearningTraining(request):
    if request.method == 'POST':
        form=ModelTrainedForm(request.POST)
        if form.is_valid():
            # Selezione tra i modelli pre-addestrati disponibili
            modelTypeTrained=form.cleaned_data['modelTypeTrained']

            if modelTypeTrained=='bert-base-uncased':
                try:
                    #Preleva tokenizer ed il modello pre-addestrato
                    tokenizer=AutoTokenizer.from_pretrained(modelTypeTrained)
                    model=AutoModelForSequenceClassification.from_pretrained(modelTypeTrained)
                except Exception as e:
                    errore=f"Errore durante il caricamento del modello o del tokenizer: {str(e)}"
                    messages.error(request,errore)
                    print(errore)
                    return render(request, '404.html')
            elif modelTypeTrained=='distilbert-base-uncased':
                try:
                    # Preleva tokenizer ed il modello pre-addestrato
                    tokenizer = DistilBertTokenizer.from_pretrained(modelTypeTrained)
                    model = DistilBertForSequenceClassification.from_pretrained(modelTypeTrained)
                except Exception as e:
                    errore = f"Errore durante il caricamento del modello o del tokenizer: {str(e)}"
                    messages.error(request, errore)
                    print(errore)
                    return render(request, '404.html')
            try:
                #Carica il dataset -- test prova
                dataset=load_dataset('glue','mrpc')
                encoded_dataset=dataset.map(lambda examples: tokenizer(examples['sentence1'],examples['sentence2'],truncation=True,padding='max_length'),batched=True)
            except Exception as e:
                errore = f"Errore durante il caricamento del dataset: {str(e)}"
                messages.error(request, errore)
                print(errore)
                return render(request, '404.html')

                #Inizia monitoraggio con CodeCarbon
            tracker = OfflineEmissionsTracker(
                country_iso_code="ITA",
                output_file=output_file,
                output_dir=output_dir
            )

            tracker.start()

            try:
                # Addestra il modello sul dataset
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
                #trainer.save_model("tesiSostenibilita/modelliAddestrati")
            except Exception as e:
                errore = f"Errore durante l'esecuzione del modello:{str(e)}"
                messages.error(request, errore)
                tracker.stop()
                print(errore)

                return render(request, '404.html')


            #Stop del monitoraggio delle emissioni
            tracker.stop()

            # Verifica se il file è stato creato
            print(f"File CSV creato: {os.path.join(output_dir, output_file)}")

            with open(output_dir+'/'+output_file,'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                for row in csv_reader:
                    print(row)

            messages.success(request, "Elaborazione completata con successo!")
            return render(request, 'results.html')

    return(render(request, '404.html'))


def uploadFile(request):
    if request.method == 'POST':
        form = FileTraniningForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data['fileTraining']
            dataFile = form.cleaned_data['dataFile']
            modelType = form.cleaned_data['modelType']

            # Verifica il formato dei dati
            if not dataFile.name.endswith(('.csv', '.xlsx', '.xls', '.json', '.yaml')):
                errore = "Tipo di file dei dati non supportato"
                messages.error(request, errore)
                print(errore)
                return render(request, '404.html')

            # Salva il file temporaneamente
            _, temp_file_path = tempfile.mkstemp()
            with open(temp_file_path, 'wb+') as temp_file:
                for chunk in file.chunks():
                    temp_file.write(chunk)

            # Carica il modello
            try:
                if modelType == 'tensorflow':
                    model = tensorflow.keras.models.load_model(temp_file_path)
                elif modelType == 'pytorch':
                    model = torch.load(temp_file_path)
                elif modelType == 'onnx':
                    model = onnx.load(temp_file_path)
                    sess = ort.InferenceSession(model.SerializeToString())
                else:
                    raise ValueError("Tipo di modello non supportato")
                print("Il modello è stato caricato correttamente")
            except Exception as e:
                errore = f"Errore durante il caricamento del modello: {str(e)}"
                messages.error(request, errore)
                print(errore)
                return render(request, '404.html')

            # Leggi i dati dal file
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
                    raise ValueError("Tipo di file dei dati non supportato")
                print("Il file dei dati è stato letto correttamente")
            except Exception as e:
                errore = "Errore durante la lettura del file dei dati: " + str(e)
                messages.error(request, errore)
                print(errore)
                return render(request, '404.html')

                # Inizia monitoraggio con CodeCarbon
            tracker = OfflineEmissionsTracker(
                    country_iso_code="ITA",
                    output_file=output_file,
                    output_dir=output_dir
            )

            tracker.start()

            # Esegui il modello
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
                    raise ValueError("Tipo di modello non supportato")
            except Exception as e:
                errore = "Errore durante l'esecuzione del modello: " + str(e)
                messages.error(request, errore)
                print(errore)
                tracker.stop()
                return render(request, '404.html')

            tracker.stop()

            # Verifica se il file è stato creato
            print(f"File CSV creato: {os.path.join(output_dir, output_file)}")

            with open(output_dir + '/' + output_file, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                for row in csv_reader:
                    print(row)

            messages.success(request, "Elaborazione completata con successo!")
            os.remove(temp_file_path)  # Rimuovi il file temporaneo
            return render(request, 'results.html')
        else:
            print(form.errors)
            errore = "Errore nel caricamento del file"
            messages.error(request, errore)
            print(errore)
            return render(request, 'index.html')
    else:
        return render(request, 'trainingFile.html')


