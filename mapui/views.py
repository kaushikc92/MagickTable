import os, shutil

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.db.models import Sum
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from tiler.models.Document import Document, TiledDocument
from tiler.views import convert_html
import json
import pdb

max_usage = 25000

def index(request):
    return HttpResponse("Hello, world. You're at the map ui index.")

def leaflet(request):
    file_name = request.GET.get("file")

    rows, columns = check_csv(file_name)

    output_name = file_name[:-4] + ".html"
    context = {'file': file_name, 'profile': output_name, 'rows': str(rows), 'columns': str(columns)}
    return render(request, 'leaflet_map.html', context)

def tilecount(request):
    file_name = request.GET.get("file")
    tilecount = TiledDocument.objects.filter(document__file_name=file_name).aggregate(Sum('tile_count_on_y'))['tile_count_on_y__sum']
    return JsonResponse({'tilecount': tilecount})

@method_decorator(csrf_exempt)
def delete(request):
    file_name = request.POST.get("file_name")
    Document.objects.get(file_name=file_name).delete()

    dir_name = file_name[0:-4]
    shutil.rmtree(os.path.join(settings.MEDIA_ROOT, 'tiles', dir_name))
    return HttpResponse("Success")

def check_csv(file_name):
    doc = Document.objects.get(file_name=file_name)
    rows, columns = 0, 0
    if not os.path.isdir(os.path.join(settings.MEDIA_ROOT, 'tiles', file_name[:-4])):
        #check_disk_usage()
        rows, columns = convert_html(doc, file_name)
        doc.rows = rows
        doc.columns = columns
    else:
        rows = doc.rows
        columns = doc.columns
    doc.save()
    return rows, columns

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
