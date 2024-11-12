from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from datetime import datetime


class SignUpForm(UserCreationForm):
    email = forms.EmailField(max_length=254, required=True, help_text='Required. Enter a valid email address.')

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

class ImageUploadForm(forms.Form):
    image = forms.ImageField()
    patient_name = forms.CharField(max_length=100)
    patient_age = forms.IntegerField( min_value=0, max_value=100)
    patient_gender = forms.ChoiceField(choices=[('M', 'Male'), ('F', 'Female')])  
    location = forms.CharField(max_length=100)

class MonthlyReportForm(forms.Form):  
    graph_choices = [
        ('daily', 'Cases by Day'),
        ('quarterly', 'Cases by Quarter'),
        ('location', 'Cases by Location'),
    ]
    graph_type = forms.ChoiceField(label="Graph Type", choices=graph_choices)

    year = forms.IntegerField(label="Year", min_value=2024, max_value=2100)
    month = forms.IntegerField(label="Month", min_value=1, max_value=12,required=False)  

class StatisticsFilterForm(forms.Form):
    start_date = forms.DateField(
        label='Start Date', 
        required=False, 
        input_formats=['%d/%m/%Y'],  # Allows format as DD/MM/YYYY
        widget=forms.DateInput(format='%d/%m/%Y')
    )
    end_date = forms.DateField(
        label='End Date', 
        required=False, 
        input_formats=['%d/%m/%Y'],  # Allows format as DD/MM/YYYY
        widget=forms.DateInput(format='%d/%m/%Y')
    )