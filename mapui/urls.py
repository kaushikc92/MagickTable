from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='mapui_index'),
    path('leaflet', views.leaflet, name='leaflet'),
    path('tilecount', views.tilecount, name='tilecount'),
    path('delete', views.delete, name='delete'),
    path('profile', views.table_profile, name='pandas_profile')
]
