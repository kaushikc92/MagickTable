import math
import queue
import os
import threading
import time
import io

import imgkit
import pandas as pd
import pandas_profiling as pf
from PIL import Image
from django.conf import settings
from django.db.models import F, Sum, Max
from django.http import HttpResponse, JsonResponse
from django.utils.cache import add_never_cache_headers
from django.core.exceptions import ObjectDoesNotExist

from tiler.models.Document import TiledDocument
import pdb
import numpy as np
import random

max_chars_per_column = 2000
st_images = {}
st_images_max = 25
cv = threading.Lock()
write_q = queue.Queue()
progressStatus = "Uploading File"
progressValue = 10

def index(request):
    return HttpResponse("Index page of tiler")

def bb_check(file_name, x, y):
    if x < 0 or y < 0:
        return empty_response()

    if y > TiledDocument.objects.filter(document__file_name=file_name).aggregate(Sum('tile_count_on_y'))['tile_count_on_y__sum']:
        return empty_response()

def get_subtable_number(file_name, x, y, z):
    tile_details = TiledDocument.objects.filter(document__file_name=file_name).order_by('subtable_number')

    subtable_number = 0
    agg_rows = 0
    for i in range(0, len(tile_details)):
        agg_rows = agg_rows + tile_details[i].tile_count_on_y
        if y < agg_rows:
            subtable_number = i
            y = y + tile_details[i].tile_count_on_y - agg_rows
            break
        if i == len(tile_details) - 1:
            return empty_response()
    
    tile_count_on_x = tile_details[subtable_number].tile_count_on_x
    if x >= tile_count_on_x:
        return empty_response()

    return subtable_number
    

def tile_request(request, id, z, x, y):
    file_name = request.GET.get("file")
    z = int(z) - 3
    
    x = int(x) * (2 ** (10 - z))
    y = int(y) * (2 ** (10 - z))

    bb_check(file_name, x, y)
    subtable_number = get_subtable_number(file_name, x, y, z)

    subtable_name = file_name[:-4] + str(subtable_number) + '.jpg'

    img = None
    if subtable_name in st_images:
        img = st_images[subtable_name]
    else:
        if cv.acquire(blocking=False):
            path = os.path.join(settings.MEDIA_ROOT, 'tiles', file_name[:-4], subtable_name)
            img = Image.open(path)
            img.load()
            keys = st_images.keys()
            if len(keys) >= st_images_max:
                st_images.popitem()
            st_images[subtable_name] = img
            cv.release()
        else:
            while subtable_name not in st_images:
                pass
            img = st_images[subtable_name]
        
    tile_size = 2 ** (18 - z)
    tile_img = img.crop((x*256, y*256, x*256 + tile_size, y*256 + tile_size))
    tile_img = tile_img.resize((256,256))
        
    try:
        response = HttpResponse(content_type="image/jpg")
        tile_img.save(response, 'jpeg')
        return response

    except IOError:
        red = Image.new('RGB', (256, 256), (255, 0, 0))
        response = HttpResponse(content_type="image/jpg")
        add_never_cache_headers(response)
        red.save(response, "jpeg")
        return response

def progress(request):
    return JsonResponse({'progressValue': progressValue, 'progressStatus': progressStatus})

def get_subtable_dimensions(csv):
    chars_per_row = 0
    max_lines_per_row = 1
    for col in csv.columns:
        max_len = csv[col].map(len).max()
        if max_len > 50:
            lines_per_row = math.ceil(max_len / 200)
            max_lines_per_row = max(max_lines_per_row, lines_per_row)
            max_len = 50
        chars_per_row = chars_per_row + max_len

    max_width = chars_per_row * 5
    rows_per_image = math.ceil(400 / max_lines_per_row)
    return rows_per_image, max_width

def convert_html(document, csv_name):
    global progressValue
    global progressStatus
    progressValue = "25"
    progressStatus = "Processing CSV"
    add_entries = not TiledDocument.objects.filter(document=document).exists()
    csv = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "documents", csv_name))
    csv = csv.astype(str).apply(lambda x: x.str[:max_chars_per_column])
    
    rows_per_image, max_width = get_subtable_dimensions(csv)

    os.mkdir(os.path.join(settings.MEDIA_ROOT, 'tiles', csv_name[0:-4]))

    df = csv[0 : rows_per_image]

    progressValue = "50"
    progressStatus = "Creating image for first " + str(rows_per_image) + "rows" 
    img1 = convert_subtable_html(df, csv_name, 0, max_width)
    
    img2 = None
    if csv.shape[0] > rows_per_image:
        df = csv[rows_per_image : 2 * rows_per_image]
        img2 = convert_subtable_html(df, csv_name, 1, max_width)
    
    progressValue = "80"
    progressStatus = "Loading image into memory" 
    img, start_row = create_subtable_image(img1, img2, 0)
    pil_img = Image.fromarray(img)
    keys = st_images.keys()
    subtable_name = csv_name[:-4] + '0.jpg'
    if len(keys) >= st_images_max:
        st_images.popitem()
    st_images[subtable_name] = pil_img
    
    progressValue = "90"
    progressStatus = "Writing image to disk" 
    subtable_path = os.path.join(settings.MEDIA_ROOT, 'tiles', csv_name[:-4], subtable_name)
    pil_img.save(subtable_path, 'jpeg', quality=60)
    
    if add_entries:
        add_subtable_entries(document, csv_name, 0, [img])

    if img2 is not None:
        t = threading.Thread(target=convert_remaining_html, args=(document, csv_name, csv, rows_per_image, max_width, 
            img2, start_row, add_entries))
        t.start()

    progressStatus = "Uploading File"
    progressValue = 10
    return csv.shape[0], csv.shape[1]

def convert_remaining_html(document, csv_name, csv, rows_per_image, max_width, img1, start_row, add_entries):
    number_of_subtables = math.ceil(csv.shape[0] / rows_per_image)
    batch_size = 10
    no_of_batches = math.ceil(number_of_subtables / batch_size)
    
    num_write_threads = 10
    write_threads = []
    
    for i in range(num_write_threads):
        w = threading.Thread(target=worker)
        w.start()
        write_threads.append(w)

    for i in range(0, no_of_batches):
        converted_images = [None] * batch_size
        threads = []
        for j in range(0, batch_size):
            subtable_number = batch_size * i + j + 2
            df = csv[subtable_number * rows_per_image: (subtable_number+1) * rows_per_image]
            t = threading.Thread(target=convert_subtable_html, args=(df, csv_name, j, max_width, converted_images))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        subtable_images = []
        for j, img2 in enumerate(converted_images):
            subtable_number = batch_size * i + j + 1
            if start_row == -1:
                if img2 is not None:
                    subtable_images.append(img2)
                break
            img, start_row = create_subtable_image(img1, img2, start_row)
            img1 = img2
            subtable_images.append(img)
            pil_img = Image.fromarray(img)
            subtable_name = csv_name[:-4] + str(subtable_number) + '.jpg'
            subtable_path = os.path.join(settings.MEDIA_ROOT, 'tiles', csv_name[:-4], subtable_name)
            keys = st_images.keys()
            if len(keys) < st_images_max:
                st_images[subtable_name] = pil_img
            write_q.put((pil_img, subtable_path))

        if add_entries:
            add_subtable_entries(document, csv_name, batch_size*i, subtable_images)

    write_q.join()

    for i in range(num_write_threads):
        write_q.put(None)
    for w in write_threads:
        w.join()

def convert_subtable_html(df, csv_name, subtable_number, max_width, results=None):
    if df.shape[0] == 0:
        return None
    pd.set_option('display.max_colwidth', -1)
    html = df.to_html(index=False, border=1).replace('<td>', '<td style = "word-wrap: break-word;' + 
        ' text-align:center; font-family: Times New Roman; font-size: 18;">')
    html = html.replace('<th>', '<th style = "background-color: #9cd4e2; text-align: center;' + 
        ' font-family: Times New Roman; font-size: 18;">')
    html = html.replace('<table', '<div style="width:' + str(max_width) + 'px;">\n<table ' + 
        'style="border-collapse: collapse;"')
    html = html.replace('</table>', '</table>\n</div>')

    options = {
        'quality' : '60'
    }
    img = Image.open(io.BytesIO(imgkit.from_string(html, False, options=options)))
    img_arr = np.array(img)
    if results is not None:
        results[subtable_number] = img_arr
    else:
        return img_arr

def create_subtable_image(img1, img2, start_row):
    h1 = img1.shape[0]
    w1 = img1.shape[1]
    max_tile_size = 2 ** 12
    number_of_rows = int(math.ceil((h1 - start_row) / (max_tile_size * 1.0)))
    number_of_cols = int(math.ceil(w1 / (max_tile_size * 1.0)))

    diff = max_tile_size * number_of_rows - h1 + start_row
    
    if img2 is None:
        img = img1[start_row : h1, :]
        img = pad_img(img, max_tile_size * number_of_rows, max_tile_size * number_of_cols)
        return img, -1
    else:
        h2 = img2.shape[0]
        w2 = img2.shape[1]
        if h2 < diff:
            img = np.full((h1 - start_row + h2, max(w1, w2), 3), 255, dtype=np.uint8)
            img[:h1-start_row, :w1] = img1[start_row:, :]
            img[h1-start_row:,:w2] = img2[:, :]
            img = pad_img(img, max_tile_size * number_of_rows, max_tile_size * number_of_cols)
            return img, -1
        else:
            img = np.full((h1 - start_row + diff, max(w1, w2), 3), 255, dtype=np.uint8)
            img[:h1-start_row, :w1] = img1[start_row:, :]
            img[h1-start_row:,:w2] = img2[:diff, :]
            img = pad_img(img, max_tile_size * number_of_rows, max_tile_size * number_of_cols)
            return img, diff

def write_subtable_image(pil_img, subtable_path):
    pil_img.save(subtable_path, 'jpeg', quality=60)
    print("{0} written".format(subtable_path))

def worker():
    while True:
        item = write_q.get()
        if item is None:
            break
        write_subtable_image(item[0], item[1])
        write_q.task_done()

def add_subtable_entries(document, csv_name, start_st_no, images):
    entries = []
    for i, img in enumerate(images):
        tile_size = 2 ** 12
        nrows = int(math.ceil(img.shape[0]/tile_size)) * (2 ** 4) 
        ncols = int(math.ceil(img.shape[1]/tile_size)) * (2 ** 4)
        entries.append(TiledDocument(document=document, subtable_number=start_st_no+i,
            tile_count_on_x=ncols, tile_count_on_y=nrows, total_tile_count=ncols*nrows))

    TiledDocument.objects.bulk_create(entries)

def get_tile(img, subtable_number, j, i, zoom_level):
    tile_size = 2 ** (18 - zoom_level)

    number_of_rows = int(math.ceil(img.shape[0] / (tile_size * 1.0)))
    number_of_cols = int(math.ceil(img.shape[1] / (tile_size * 1.0)))

    if img.shape[1] % tile_size != 0 or img.shape[0] % tile_size != 0:
        img = pad_img(img, tile_size * number_of_rows, tile_size * number_of_cols)

    cropped_img = img[i * tile_size : (i+1) * tile_size, j * tile_size : (j+1) * tile_size]
    cropped_img = Image.fromarray(cropped_img)
    cropped_img = cropped_img.resize((256, 256))

    return cropped_img

def pad_img(img, h, w):
    height, width, channels = img.shape
    new_img = np.full((h, w, channels), 221, dtype=np.uint8)
    new_img[0:height, 0:width] = img
    return new_img

def empty_response():
    red = Image.new('RGBA', (1, 1), (255, 0, 0, 0))
    response = HttpResponse(content_type="image/png")
    red.save(response, "png")
    return response

def error_response():
    red = Image.new('RGB', (256, 256), (255, 0, 0))
    response = HttpResponse(content_type="image/jpg")
    red.save(response, "jpeg")
    return response
