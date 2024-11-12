from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from .forms import SignUpForm, ImageUploadForm,  MonthlyReportForm,  StatisticsFilterForm
from .models import Prediction, Patient
from django.db.models import Avg
import tensorflow as tf
from PIL import Image
import io 
from io import BytesIO
import tempfile
import numpy as np
import os
from django.conf import settings
from django.http import HttpResponse
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter, A4
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from django.db.models import Count
import base64
import matplotlib.ticker as mticker
from collections import defaultdict


# Global variable to store the model after loading
model = None

def get_model():
    """Loads the TB detection model or defines a new one if loading fails."""
    global model
    if model is None:
        model_path = settings.MODEL_FILE_PATH  # Ensure this is set in your settings
        try:
            # Attempt to load an existing model
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully from:", model_path)
        except (IOError, ValueError) as e:
            # If loading fails, define a new model
            print("Loading model failed, defining a new model.")
            input_dim = 100  # Replace with your actual input dimension
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            # Compile the new model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print("New model defined and compiled.")
    return model

def generate_pdf_report(prediction):
    """Generates a styled PDF report for the TB prediction."""
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # Define colors
    BLUE = (0.149, 0.388, 0.922)  # #2563eb
    GRAY = (0.294, 0.333, 0.392)  # #4b5563
    
    # Header
    p.setFillColorRGB(*BLUE)
    p.setFont("Helvetica-Bold", 24)
    p.drawString(1 * inch, 10 * inch, "TB Detection Report")
    
    # Patient Information
    p.setFillColorRGB(*GRAY)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(1 * inch, 9 * inch, "Patient Information")
    
    p.setFont("Helvetica", 12)
    y_position = 8.5 * inch
    
    info_items = [
        f"Name: {prediction.patient_name}",
        f"Age: {prediction.patient_age}",
        f"Gender: {prediction.patient_gender}",
        f"Location: {prediction.location}",
        f"Date: {prediction.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    
    for item in info_items:
        p.drawString(1 * inch, y_position, item)
        y_position -= 0.4 * inch
    
    # Results Section
    p.setFillColorRGB(*BLUE)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(1 * inch, y_position - 0.2 * inch, "Analysis Results")
    
    p.setFillColorRGB(*GRAY)
    p.setFont("Helvetica", 12)
    y_position -= 0.8 * inch
    
    result_text = "Positive" if prediction.result else "Negative"
    p.drawString(1 * inch, y_position, f"TB Status: {result_text}")
    p.drawString(1 * inch, y_position - 0.4 * inch, f"Confidence: {prediction.confidence:.2f}")
    
    # Image
    img_path = os.path.join(settings.MEDIA_ROOT, str(prediction.image))
    img = Image.open(img_path)
    img = img.convert('RGB')
    
    # Calculate image dimensions while maintaining aspect ratio
    img_width, img_height = img.size
    aspect = img_height / float(img_width)
    display_width = 6 * inch  # Larger image size
    display_height = display_width * aspect
    
    # Center the image horizontally
    x_position = (letter[0] - display_width) / 2
    y_position -= display_height + inch  # Position image below text with some spacing
    
    p.drawImage(img_path, x_position, y_position, width=display_width, height=display_height)
    
    # Footer
    p.setFont("Helvetica-Oblique", 8)
    p.drawString(1 * inch, 0.5 * inch, "This report was generated automatically. Please consult with a medical professional for official diagnosis.")
    
    p.showPage()
    p.save()
    
    buffer.seek(0)
    return buffer

@login_required
def download_pdf_report(request, prediction_id):
    """Handles PDF report downloads."""
    try:
        prediction = Prediction.objects.get(id=prediction_id)
        pdf_buffer = generate_pdf_report(prediction)
        response = HttpResponse(pdf_buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="TB_Report_{prediction_id}.pdf"'
        return response
    except Prediction.DoesNotExist:
        return HttpResponse("Prediction not found.", status=404)

def predict_tb(image):
    """Predicts TB status using a loaded model."""
    try:
        model = get_model()
        if model is None:
            raise ValueError("Model could not be loaded.")

        # Define image dimensions
        img_height = 180
        img_width = 180

        # Load and preprocess the image
        img = tf.keras.utils.load_img(image.file, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        
        # Make predictions
        predictions = model.predict(img_array)
        print(predictions)
        score = predictions[0][0]
        print(score)

        if score > 0.5:
            print("This image most likely shows signs of TB with {:.2f}% confidence.".format(score))
        else:
            print("This image most likely does not show signs of TB with {:.2f}% confidence.".format(1 - score))

        return score > 0.5, score  # Return the result and confidence score
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, None

def signup(request):
    """Handles user signup."""
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('upload_image')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})

@login_required
def upload_image(request):
    """Handles image uploads and TB prediction."""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                image = form.cleaned_data['image']
                patient_name = form.cleaned_data['patient_name']
                patient_age = form.cleaned_data['patient_age']
                patient_gender = form.cleaned_data['patient_gender']
                location = form.cleaned_data['location']

                prediction_result, confidence = predict_tb(image)

                if prediction_result is None or confidence is None:
                    raise ValueError("Prediction failed")

                prediction = Prediction.objects.create(
                    user=request.user,
                    patient_name=patient_name,
                    patient_age=patient_age,
                    patient_gender=patient_gender,
                    image=image,
                    result=prediction_result,
                    confidence=confidence,
                    location=location
                )
                return redirect('prediction_results', prediction_id=prediction.id)

            except ValueError as ve:
                print(f"Prediction error: {str(ve)}")
                return render(request, 'upload_image.html', {
                    'form': form,
                    'error_message': 'There was an issue with the prediction. Please try again.'
                })

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                return render(request, 'upload_image.html', {
                    'form': form,
                    'error_message': 'There was an error processing the image. Please try again.'
                })

    else:
        form = ImageUploadForm()

    return render(request, 'upload_image.html', {'form': form})

@login_required
def prediction_results(request, prediction_id):
    """Displays the prediction results."""
    try:
        prediction = Prediction.objects.get(id=prediction_id)
        return render(request, 'prediction_results.html', {'prediction': prediction})
    except Prediction.DoesNotExist:
        return HttpResponse("Prediction not found.", status=404)


@login_required
def statistics(request):
    """Displays statistical information about TB predictions."""
    total_predictions = Prediction.objects.count()
    positive_predictions = Prediction.objects.filter(result=True).count()
    negative_predictions = total_predictions - positive_predictions
    avg_confidence = Prediction.objects.aggregate(Avg('confidence'))['confidence__avg']

    if avg_confidence is None:
        avg_confidence = 0.0  # Handle the case where no predictions have been made.

    user_predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')[:5]

    context = {
        'total_predictions': total_predictions,
        'positive_predictions': positive_predictions,
        'negative_predictions': negative_predictions,
        'avg_confidence': avg_confidence,
        'user_predictions': user_predictions,
    }
    return render(request, 'statistics.html', context)

def get_base64_image(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return graph_data

def generate_quarterly_cases_chart(predictions, year):
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Define month ranges for each quarter
    quarter_months = {
        'Q1': [1, 2, 3],
        'Q2': [4, 5, 6],
        'Q3': [7, 8, 9],
        'Q4': [10, 11, 12],
    }
    
    # Count positive and negative cases by quarter
    quarter_counts = {'positive': [], 'negative': []}
    for months in quarter_months.values():
        quarter_predictions = predictions.filter(created_at__month__in=months)
        quarter_counts['positive'].append(quarter_predictions.filter(result=True).count())
        quarter_counts['negative'].append(quarter_predictions.filter(result=False).count())

    # Plot positive and negative cases
    ax.bar(quarters, quarter_counts['positive'], color='red', label="Positive")
    ax.bar(quarters, quarter_counts['negative'], color='green', label="Negative", bottom=quarter_counts['positive'])
    ax.set_title(f'TB Cases by Quarter - {year}')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Number of Cases')
    ax.legend()
    return get_base64_image(fig)

def generate_location_cases_chart(predictions, year):
    fig, ax = plt.subplots(figsize=(8, 5))
    locations = predictions.values_list('location', flat=True).distinct()
    location_counts = defaultdict(lambda: {'positive': 0, 'negative': 0})
    
    # Count cases by location
    for location in locations:
        location_counts[location]['positive'] = predictions.filter(location=location, result=True).count()
        location_counts[location]['negative'] = predictions.filter(location=location, result=False).count()
    
    location_labels = list(location_counts.keys())
    positive_counts = [location_counts[loc]['positive'] for loc in location_labels]
    negative_counts = [location_counts[loc]['negative'] for loc in location_labels]

    ax.bar(location_labels, positive_counts, color='red', label="Positive")
    ax.bar(location_labels, negative_counts, color='green', label="Negative", bottom=positive_counts)
    ax.set_title(f'TB Cases by Location - {year}')
    ax.set_xlabel('Location')
    ax.set_ylabel('Number of Cases')
    ax.legend()
    return get_base64_image(fig)

@login_required
def monthly_report(request):
    form = MonthlyReportForm(request.GET or None)
    graph_data_list = []  # Create an empty list to hold the graphs
    graph_data= None

    if request.method == 'GET' and form.is_valid():
        year = form.cleaned_data['year']
        month = form.cleaned_data.get('month')
        quarter = form.cleaned_data.get('quarter')
        location = form.cleaned_data.get('location')
        graph_type = form.cleaned_data['graph_type']

        predictions = Prediction.objects.filter(created_at__year=year)

        if month:
            predictions = predictions.filter(created_at__month=month)
        if quarter:
            predictions = predictions.filter(created_at__quarter=quarter)
        if location:
            predictions = predictions.filter(location=location)

        # Generate the chosen graph
        if graph_type == 'daily':
            # Daily graph generation logic
            days_in_month = predictions.dates('created_at', 'day')
            fig, ax = plt.subplots(figsize=(8, 6))
            positive_counts = [predictions.filter(created_at__day=day.day, result=True).count() for day in days_in_month]
            negative_counts = [predictions.filter(created_at__day=day.day, result=False).count() for day in days_in_month]
            day_labels = [day.day for day in days_in_month]
            ax.bar(day_labels, positive_counts, color='red', label="Positive")
            ax.bar(day_labels, negative_counts, color='green', label="Negative", bottom=positive_counts)
            ax.set_title(f'TB Cases per Day - {year}-{month}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Cases')
            ax.legend()
            graph_data_list.append(get_base64_image(fig))  # Append graph to list

        elif graph_type == 'quarterly':
            graph_data_list.append(generate_quarterly_cases_chart(predictions, year))

        elif graph_type == 'location':
            graph_data_list.append(generate_location_cases_chart(predictions, year))

        # Store the generated graph data in the session
        request.session['graph_data_list'] = graph_data_list  # Save the graphs to the session

    return render(request, 'monthly_report.html', {
        'form': form,
        'graph_data': graph_data_list[0] if graph_data_list else None,
    })

@login_required
def download_graphs_pdf(request):
    graph_data_list = request.session.get('graph_data_list')
    
    if not graph_data_list:
        return HttpResponse("No graphs available to download.", status=404)

    # Create a PDF file in memory
    buffer = BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=A4)
    pdf_canvas.setTitle("TB Detection Graph Report")

    # Loop through each base64 image, decode it, save it temporarily, and add it to a new page in the PDF
    for graph_data in graph_data_list:
        image_data = base64.b64decode(graph_data)
        
        # Save the decoded image data as a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image_file:
            temp_image_file.write(image_data)
            temp_image_file_path = temp_image_file.name

        try:
            # Draw the image from the temporary file path
            pdf_canvas.drawImage(temp_image_file_path, 50, 250, width=500, height=400)
            pdf_canvas.showPage()  
        finally:
            # Remove the temporary file after usage
            if os.path.exists(temp_image_file_path):
                os.remove(temp_image_file_path)

    pdf_canvas.save()
    buffer.seek(0)

    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="tb_detection_graph_report.pdf"'
    return response