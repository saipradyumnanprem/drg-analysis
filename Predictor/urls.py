from django.urls import path
from . import views

urlpatterns = [
    path("", views.predictor, name="predict"),
    path("results/", views.results, name="results"),
]
