from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='upload_image'),
    path('signup/', views.signup, name='signup'),
    path('prediction_results/<int:prediction_id>/', views.prediction_results, name='prediction_results'),
    path('statistics/', views.statistics, name='statistics'),
    path('download_report/<int:prediction_id>/', views.download_pdf_report, name='download_pdf_report'),
    path('monthly-report/', views.monthly_report, name='monthly_report'),
    path('download-graphs-pdf/', views.download_graphs_pdf, name='download_graphs_pdf'),
]
