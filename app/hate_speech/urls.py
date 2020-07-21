from django.shortcuts import redirect
from django.urls import path
from .views import getMainPage, improveHurtlex
from app.backend.helper import ENGLISH

urlpatterns = [
    path('<language>/', getMainPage, name='main-page'),
    path('<language>/improve-hurtlex', improveHurtlex, name='improve-hurtlex'),
]