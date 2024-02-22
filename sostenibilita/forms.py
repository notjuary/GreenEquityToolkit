from django import forms
class FileTraniningForm(forms.Form):
    MODEL_CHOICES = [
        ('sklearn', 'Scikit-learn'),
        ('tensorflow', 'TensorFlow'),
        ('pytorch', 'PyTorch'),
        ('onnx','ONNX')
    ]
    fileTraining=forms.FileField(label="fileTrainining")
    dataFile=forms.FileField(label="dataFile")
    modelType = forms.ChoiceField(choices=MODEL_CHOICES, label="modelType")

class ModelTrainedForm(forms.Form):
    MODEL_CHOICES = [
        ('bert-base-uncased', 'BERT'),
        ('distilbert-base-uncased','DistilBERT'),
    ]

    modelTypeTrained = forms.ChoiceField(choices=MODEL_CHOICES, label="modelTypeTrained")
