import os

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render


def index(request):
    return HttpResponse("Hello, world. You're at the map ui index.")


def leaflet(request):
    file_name = request.GET.get("file")
    output_name = file_name[:-4] + ".html"
    context = {'file': file_name, 'profile': output_name}
    return render(request, 'leaflet_map.html', context)


def table_profile(request):
    file_name = request.GET.get("file")
    file_name = file_name[:-4] + "html"
    output_name = os.path.join(settings.MEDIA_ROOT, "documents", file_name)
    str = open(output_name).read()
    return HttpResponse(str)
