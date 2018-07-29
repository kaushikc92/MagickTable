import os, shutil

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render

from tiler.models.Document import Document
from tiler.views import convert_html
import pdb

max_usage = 25000

def index(request):
    return HttpResponse("Hello, world. You're at the map ui index.")

def leaflet(request):
    file_name = request.GET.get("file")

    check_csv(file_name)

    output_name = file_name[:-4] + ".html"
    context = {'file': file_name, 'profile': output_name}
    return render(request, 'leaflet_map.html', context)

def check_csv(file_name):
    if os.path.isdir(os.path.join(settings.MEDIA_ROOT, 'tiles', file_name[:-4])):
        return
    check_disk_usage()
    doc = Document.objects.get(file_name=file_name)
    doc.save()

    convert_html(doc, file_name)

def check_disk_usage():
    csv_sizes = {}
    total_size = 0
    for dir_name in os.listdir(os.path.join(settings.MEDIA_ROOT, 'tiles')):
        size = get_directory_size(os.path.join(settings.MEDIA_ROOT, 'tiles', dir_name))
        total_size = size + total_size
        csv_sizes[dir_name] = size

    if total_size < max_usage:
        return
    accesses = {}

    for csv_name in csv_sizes:
        doc = Document.objects.get(file_name=csv_name+'.csv')
        accesses[doc.last_access] = csv_name

    while total_size > max_usage:
        oldest = accesses.pop(min(accesses.keys()))
        shutil.rmtree(os.path.join(settings.MEDIA_ROOT, 'tiles', oldest))
        total_size = total_size - csv_sizes.pop(oldest)

def get_directory_size(dir_path):
    total_size = 0
    for path, dirs, files in os.walk(dir_path):
        for f in files:
            fp = os.path.join(path, f)
            total_size = total_size + os.path.getsize(fp)

    return total_size / (1024 * 1024)

def table_profile(request):
    file_name = request.GET.get("file")
    file_name = file_name[:-4] + "html"
    output_name = os.path.join(settings.MEDIA_ROOT, "documents", file_name)
    str = open(output_name).read()
    return HttpResponse(str)
