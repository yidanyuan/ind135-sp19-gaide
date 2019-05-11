from django.urls import path
from . import views

urlpatterns = [
    path('',views.index, name='homepage_guest'),
    path('login/',views.login, name='login'),
    path('homepage/',views.homepage, name='homepage'),
    path('speaker/',views.speaker, name='speaker')
]