
from django.db import models
from django.contrib.auth.models import User
import uuid

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    patient_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)  # Unique patient ID
    patient_name = models.CharField(max_length=100, default="John Doe")
    patient_age = models.IntegerField(default=15)
    patient_gender = models.CharField(max_length=1, choices=[('M', 'Male'), ('F', 'Female')], default='M')
    image = models.ImageField(upload_to='images/')
    result = models.BooleanField()
    confidence = models.FloatField()
    location = models.CharField(max_length=100, default='Unknown Location')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.patient_name} - Result: {'Positive' if self.result else 'Negative'}"

class Patient(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    patient_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)  # Unique patient ID
    patient_name = models.CharField(max_length=100, default="John Doe")
    patient_age = models.IntegerField(default=15)
    patient_gender = models.CharField(max_length=1, choices=[('M', 'Male'), ('F', 'Female')], default='M')
    address = models.CharField(max_length=255)
    image = models.ImageField(upload_to='images/')
    result = models.BooleanField()
    treatment_status = models.CharField(max_length=20, default='Unknown')
    confidence = models.FloatField()
    location = models.CharField(max_length=100, default='Unknown Location')
    latitude = models.FloatField(default=0.0)
    longitude = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.patient_name