import math
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
from django.http import HttpResponse
from django.utils.cache import add_never_cache_headers
from django.core.exceptions import ObjectDoesNotExist

from tiler.models.Document import TiledDocument
import pdb
import numpy as np

max_chars_per_column = 2000

def index(request):
    return HttpResponse("Index page of tiler")

def tile_request(request, id, z, x, y):
    file_name = request.GET.get("file")
    z = int(z) - 3
    
    print("{0},{1} zoom {2}".format(x, y, z))
    
    x = int(x)
    y = int(y)
    if x < 0 or y < 0:
        return empty_response()

    if y > TiledDocument.objects.filter(document__file_name=file_name, zoom_level=z).aggregate(Sum('tile_count_on_y'))['tile_count_on_y__sum']:
        return empty_response()

    tile_details = TiledDocument.objects.filter(document__file_name=file_name, zoom_level=z).order_by('subtable_number')

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

    tile_count = x + tile_count_on_x * y

    path = os.path.join(settings.MEDIA_ROOT, 'tiles', file_name[0:-4], str(subtable_number), str(z),
        file_name[0:-4] + str(tile_count).zfill(5).replace("-", "0") + ".jpg")

    print(path)

    try:
        with open(path, "rb") as f:
            return HttpResponse(f.read(), content_type="image/jpg")
    except IOError:
        red = Image.new('RGB', (256, 256), (255, 0, 0))
        response = HttpResponse(content_type="image/jpg")
        add_never_cache_headers(response)
        red.save(response, "jpeg")
        return response

def convert_html(document, csv_name):
    csv = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "documents", csv_name))
    csv = csv.astype(str).apply(lambda x: x.str[:max_chars_per_column])

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

    os.mkdir(os.path.join(settings.MEDIA_ROOT, 'tiles', csv_name[0:-4]))

    #max_width = 2000
    #rows_per_image = 100
    df = csv[0 : rows_per_image]
    img1 = convert_subtable_html(df, csv_name, 0, max_width)
    
    img2 = None
    if csv.shape[0] > rows_per_image:
        df = csv[rows_per_image : 2 * rows_per_image]
        img2 = convert_subtable_html(df, csv_name, 1, max_width)
    
    img, start_row = create_subtable_image(0, img1, img2, 0)
    add_subtable_entries(document, csv_name, 0, [img])
    threads = []

    os.mkdir(os.path.join(settings.MEDIA_ROOT, 'tiles', csv_name[0:-4], '0'))

    for zoom_level in range(6, 11):
        os.mkdir(os.path.join(settings.MEDIA_ROOT, 'tiles', csv_name[0:-4], '0', str(zoom_level)))

        t = threading.Thread(target=create_tiles, args=(img, csv_name, 0, zoom_level))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    if img2 is not None:
        t = threading.Thread(target=convert_remaining_html, args=(document, csv_name, csv, rows_per_image, max_width, 
            img2, start_row))
        t.start()

def convert_remaining_html(document, csv_name, csv, rows_per_image, max_width, img2, start_row):
    pdb.set_trace()
    number_of_subtables = math.ceil(csv.shape[0] / rows_per_image)
    batch_size = 5
    no_of_batches = math.ceil(number_of_subtables / batch_size)
    for i in range(0, no_of_batches):
        subtable_images = []
        for j in range(0, batch_size):
            subtable_number = batch_size * i + j + 1
            if start_row == -1:
                if img2 is not None:
                    subtable_images[i] = img2
                break
            df = csv[(subtable_number+1) * rows_per_image: (subtable_number+2) * rows_per_image]
            img1 = img2
            img2 = convert_subtable_html(df, csv_name, subtable_number, max_width)

            img, start_row = create_subtable_image(subtable_number, img1, img2, start_row)
            subtable_images.append(img)

        add_subtable_entries(document, csv_name, batch_size*i, subtable_images)
        
        for j,img in enumerate(subtable_images):
            subtable_number = batch_size * i + j + 1
            os.mkdir(os.path.join(settings.MEDIA_ROOT, 'tiles', csv_name[0:-4], str(subtable_number)))
            for zoom_level in range(6, 11):
                os.mkdir(os.path.join(settings.MEDIA_ROOT, 'tiles', csv_name[0:-4], str(subtable_number), str(zoom_level)))
                t = threading.Thread(target=create_tiles, args=(img, csv_name, subtable_number, zoom_level))
                t.start()

def convert_subtable_html(df, csv_name, subtable_number, max_width):
    if df.shape[0] == 0:
        return None
    pd.set_option('display.max_colwidth', -1)
    html = df.to_html(index=False, border=0).replace('<td>', '<td style = "word-wrap: break-word;' + 
        ' text-align:center; font-family: Times New Roman; font-size: 18;">')
    html = html.replace('<th>', '<th style = "background-color: #9cd4e2; text-align: center;' + 
        ' font-family: Times New Roman; font-size: 18;">')
    html = html.replace('<table', '<div style="width:' + str(max_width) + 'px;">\n<table ' + 
        'style="border-collapse: collapse;"')
    html = html.replace('</table>', '</table>\n</div>')

    #f = open(os.path.join(settings.MEDIA_ROOT, "documents", csv_name[0:-4] + '.html'), 'w')
    #f.write(html)
    #f.close()

    options = {
        'quality' : '60'
    }
    img = Image.open(io.BytesIO(imgkit.from_string(html, False, options=options)))
    img_arr = np.array(img)
    return img_arr

def create_subtable_image(subtable_number, img1, img2, start_row):
    h1 = img1.shape[0]
    w1 = img1.shape[1]
    max_tile_size = 2 ** 12
    number_of_rows = int(math.ceil((h1 - start_row) / (max_tile_size * 1.0)))
    number_of_cols = int(math.ceil(w1 / (max_tile_size * 1.0)))

    diff = max_tile_size * number_of_rows - h1 + start_row
    
    if img2 is None:
        img = img1[start_row : h1, :]
        return img, -1
    else:
        h2 = img2.shape[0]
        w2 = img2.shape[1]
        if h2 < diff:
            img = np.full((h1 - start_row + h2, max(w1, w2), 3), 255, dtype=np.uint8)
            img[:h1-start_row, :w1] = img1[start_row:, :]
            img[h1-start_row:,:w2] = img2[:, :]
            return img, -1
        else:
            img = np.full((h1 - start_row + diff, max(w1, w2), 3), 255, dtype=np.uint8)
            img[:h1-start_row, :w1] = img1[start_row:, :]
            img[h1-start_row:,:w2] = img2[:diff, :]
            return img, diff

def add_subtable_entries(document, csv_name, start_st_no, images):
    entries = []
    for i, img in enumerate(images):
        for zoom_level in range(6, 11):
            tile_size = 2 ** (18 - zoom_level)
            nrows = int(math.ceil(img.shape[0]/tile_size))  
            ncols = int(math.ceil(img.shape[1]/tile_size))
            entries.append(TiledDocument(document=document, zoom_level=zoom_level, subtable_number=start_st_no+i,
                tile_count_on_x=ncols, tile_count_on_y=nrows, total_tile_count=ncols*nrows))
    TiledDocument.objects.bulk_create(entries)

def create_tiles(img, csv_name, subtable_number, zoom_level):
    tile_size = 2 ** (18 - zoom_level)
    number_of_rows = int(math.ceil(img.shape[0] / (tile_size * 1.0)))
    number_of_cols = int(math.ceil(img.shape[1] / (tile_size * 1.0)))

    if img.shape[1] % tile_size != 0 or img.shape[0] % tile_size != 0:
        img = pad_img(img, tile_size * number_of_rows, tile_size * number_of_cols)

    tile_count = 0

    tile_dir = os.path.join(settings.MEDIA_ROOT, 'tiles', csv_name[0:-4], str(subtable_number), str(zoom_level))

    for i in range(0, number_of_rows):
        for j in range(0, number_of_cols):
            cropped_img = img[i * tile_size : (i+1) * tile_size, j * tile_size : (j+1) * tile_size]
            tile_path = os.path.join(tile_dir, csv_name[0:-4] + str(tile_count).zfill(5).replace("-", "0") + ".jpg")
            cropped_img = Image.fromarray(cropped_img)
            cropped_img = cropped_img.resize((256, 256))
            cropped_img.save(tile_path, 'jpeg', quality=60)
            tile_count = tile_count + 1
    
    print("Subtable {0}, Level {1} done".format(subtable_number, zoom_level))

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

