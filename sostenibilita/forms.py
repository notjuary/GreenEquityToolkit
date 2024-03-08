from django import forms
from django.core.exceptions import ValidationError
import re

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


#Function that checks that the label for the dataset to be analyzed in social sustainability is not empty
def validate_label_attribute(value):
    if not value:
        raise forms.ValidationError("This field is required.")
    if not re.match(r"^[\w\-]+$", value):  # Allow only alphanumeric characters and hyphens
        raise forms.ValidationError("This field can only contain letters, numbers, and hyphens.")

#Class for managing the form for evaluating the environmental sustainability of files to be uploaded
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

#Class for managing the form for evaluating the social sustainability of files to be uploaded
class FileSocialForm(forms.Form):
    fileModel=forms.FileField(label="Select the ML model file to be uploaded ",validators=[validate_model_file_extension])
    datasetFile = forms.FileField(label="Selects the data file on which the model is trained ",
                                  validators=[validate_data_file_extension])
    labelAttribute = forms.CharField(label="Enter the name of the label column",validators=[validate_label_attribute])

#Class for managing the form for loading the dataset and defining the protected attributes
class UploadDatasetForm(forms.Form):
    datasetFile = forms.FileField(label="Selects the data file on which the model is trained ",
                                  validators=[validate_data_file_extension])

#Class for form management to define pre-trained models for environmental sustainability monitoring
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

#Class for managing the form for selecting protected social sustainability attributes within the dataset
class SelectProtectedAttributesForm(forms.Form):
    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns', [])
        super().__init__(*args, **kwargs)
        CHOICES = [(column, column) for column in columns if column != 'Unnamed: 0']
        self.fields['protected_attributes'] = forms.MultipleChoiceField(choices=CHOICES, widget=forms.CheckboxSelectMultiple)


#Class for form management for selection of pre-trained models for social sustainability assessment
class ModelTrainedSocialForm(forms.Form):
    MODEL_CHOICES = [
        ('bert-base-uncased', 'BERT'),
        ('distilbert-base-uncased','DistilBERT'),
    ]

    modelTypeSocial = forms.ChoiceField(choices=MODEL_CHOICES, label="Select the type of model pre-trained ")
