from django import forms
from django.core.exceptions import ValidationError
import re

#Function to check the types of files accepted by the HTML form to upload templates
def validate_model_file_extension(value):
    import os
    ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
    valid_extensions = ['.tensorflow','.pytorch', '.onnx','.pkl']
    if not ext.lower() in valid_extensions:
        raise ValidationError(u'Unsupported file extension.')

#Function to check the file types accepted by the HTML form to upload dataset files
def validate_data_file_extension(value):
    import os
    ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
    valid_extensions = ['.csv', '.xlsx', '.xls', '.json', '.yaml']
    if not ext.lower() in valid_extensions:
        raise ValidationError(u'Unsupported file extension for the data file.')

def validate_numbers_inferences(value):
    if not value:
        raise forms.ValidationError("This field is required.")
    if value <= 0: # Allow only numeric characters
        raise forms.ValidationError("This field can only contain numbers")


#Class for managing the form for evaluating the environmental sustainability of files to be uploaded
class FileTraniningForm(forms.Form):
    DEVICE_CHOICES = [
        ('cpu', 'CPU'),
        ('gpu', 'GPU'),
    ]

    fileTraining=forms.FileField(label="Select the ML model file to be uploaded ",validators=[validate_model_file_extension])
    dataFile=forms.FileField(label="Selects the data file on which the model is trained ",validators=[validate_data_file_extension])
    countryIsoCode = forms.ChoiceField(label="Select ISO Code of the country where the experiment is being run ",required=True)
    inferenceDevice=forms.ChoiceField(choices=DEVICE_CHOICES,label="Selects where to perform inference from the machine learning model ",required= True)
    numInferences=forms.IntegerField(label="Enter the inference number to be performed for the ML model ",required= True,validators=[validate_numbers_inferences])

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


#Class for managing the form for loading the dataset and defining the protected attributes
class UploadDatasetForm(forms.Form):
    datasetFile = forms.FileField(label="Selects the data file on which the model is trained ",
                                  validators=[validate_data_file_extension])

#Class for form management to define pre-trained models for environmental sustainability monitoring
class ModelTrainedForm(forms.Form):
    DEVICE_CHOICES = [
        ('CPU', 'CPU'),
        ('GPU', 'GPU'),
    ]

    MODEL_CHOICES = [
        ('distilbert-base-uncased','DistilBERT'),
    ]

    modelTypeTrained = forms.ChoiceField(choices=MODEL_CHOICES, label="Select the type of model pre-trained ",required=True)
    countryIsoCode = forms.ChoiceField(label="Select ISO Code of the country where the experiment is being run ",
                                       required=True)
    inferenceDevice = forms.ChoiceField(choices=DEVICE_CHOICES,
                                        label="Selects where to perform inference from the machine learning model ",
                                        required=True)
    numInferences = forms.IntegerField(label="Enter the inference number to be performed for the ML model ",
                                       required=True, validators=[validate_numbers_inferences])


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
        self.fields['protected_attributes'] = forms.MultipleChoiceField(choices=CHOICES, widget=forms.CheckboxSelectMultiple,error_messages={
                'required': 'Please select at least one protected attribute.',
            })
        self.fields['attribute']=forms.MultipleChoiceField(choices=CHOICES, widget=forms.CheckboxSelectMultiple,error_messages={
            'required': 'Please select at least one protected attribute.',
        })

        def clean(self):
            cleaned_data = super().clean()
            labelAttribute = cleaned_data.get('attribute')
            protected_attributes = cleaned_data.get('protected_attributes')

            if not labelAttribute:
                self.add_error('labelAttribute', 'The label field cannot be empty')

            if not protected_attributes:
                self.add_error('protected_attributes', 'You must select at least one protected attribute')

            return cleaned_data
