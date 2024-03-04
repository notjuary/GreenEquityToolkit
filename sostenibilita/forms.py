from django import forms
from django.core.exceptions import ValidationError

#Function to check the types of files accepted by the HTML form to upload templates
def validate_model_file_extension(value):
    import os
    ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
    valid_extensions = ['.tensorflow','.pytorch', '.onnx']
    if not ext.lower() in valid_extensions:
        raise ValidationError(u'Unsupported file extension.')

#Function to check the file types accepted by the HTML form to upload dataset files
def validate_data_file_extension(value):
    import os
    ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
    valid_extensions = ['.csv', '.xlsx', '.xls', '.json', '.yaml']
    if not ext.lower() in valid_extensions:
        raise ValidationError(u'Unsupported file extension for the data file.')

class FileTraniningForm(forms.Form):
    fileTraining=forms.FileField(label="Select the ML model file to be uploaded ",validators=[validate_model_file_extension])
    dataFile=forms.FileField(label="Selects the data file on which the model is trained ",validators=[validate_data_file_extension])
    countryIsoCode = forms.ChoiceField(label="Select ISO Code of the country where the experiment is being run ",required=True)
    def __init__(self, *args, **kwargs):
        country_choices = kwargs.pop('countries', [])
        super(FileTraniningForm, self).__init__(*args, **kwargs)
        self.fields['countryIsoCode'].choices = country_choices

    def clean_countryIsoCode(self):
        countryIsoCode = self.cleaned_data.get('countryIsoCode')

        available_codes = [code for code, _ in self.fields['countryIsoCode'].choices]
        if countryIsoCode not in available_codes:
            raise forms.ValidationError(f"Invalid country code. Choose from: {', '.join(available_codes)}")

        return countryIsoCode


class FileSocialForm(forms.Form):
    fileModel=forms.FileField(label="Select the ML model file to be uploaded ",validators=[validate_model_file_extension])
    datasetFile=forms.FileField(label="Selects the data file on which the model is trained ",validators=[validate_data_file_extension])
    protectedAttribute = forms.CharField(label="Enter the name of the protected attribute column")
    labelAttribute = forms.CharField(label="Enter the name of the label column")


class ModelTrainedForm(forms.Form):
    MODEL_CHOICES = [
        ('bert-base-uncased', 'BERT'),
        ('distilbert-base-uncased','DistilBERT'),
    ]

    modelTypeTrained = forms.ChoiceField(choices=MODEL_CHOICES, label="Select the type of model pre-trained ")
    countryIsoCode = forms.ChoiceField(label="Select ISO Code of the country where the experiment is being run ",required= True)

    def __init__(self, *args, **kwargs):
        country_choices = kwargs.pop('countries', [])
        super(ModelTrainedForm, self).__init__(*args, **kwargs)
        self.fields['countryIsoCode'].choices = country_choices

    def clean_countryIsoCode(self):
        countryIsoCode = self.cleaned_data.get('countryIsoCode')

        available_codes = [code for code, _ in self.fields['countryIsoCode'].choices]
        if countryIsoCode not in available_codes:
            raise forms.ValidationError(f"Invalid country code. Choose from: {', '.join(available_codes)}")

        return countryIsoCode


class ModelTrainedSocialForm(forms.Form):
    MODEL_CHOICES = [
        ('bert-base-uncased', 'BERT'),
        ('distilbert-base-uncased','DistilBERT'),
    ]

    modelTypeSocial = forms.ChoiceField(choices=MODEL_CHOICES, label="Select the type of model pre-trained ")