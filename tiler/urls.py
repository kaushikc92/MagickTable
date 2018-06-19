from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='tiler_index'),
    path('v4/<id>/<z>/<x>/<y>', views.tile_request, name='tilerequest'),
]
# + include('mapui.urls', namespace="mapui", app_name='mapui')
