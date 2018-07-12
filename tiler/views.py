import math
import os
import threading

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

def index(request):
    return HttpResponse("Index page of tiler")

rows_per_image = 300
#max_chars_per_column = 100

# this is the function that will return a tile based on x, y, z
# TODO try different image formats
# TODO: add an inmemory caching layer ex: memcached
# TODO: see if we can bunch a number of requests together rather than 1 per tile
# TODO mapbox uses 256 by 256 squares: so we need to pad our generated image to fit that
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

    path = os.path.join(settings.MEDIA_ROOT, 'tiles', str(z), str(subtable_number), file_name + str(tile_count).zfill(5).replace("-", "0") + ".jpg")

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


# TODO: calculate this rather than hard code?
font_sizes = {
    0: 4,
    1: 5.6,
    2: 7.2,
    3: 8.8,
    4: 10,
    5: 11.6,
    6: 13.2,
    7: 14.8,
    8: 16,
    9: 17.2,
    10: 18.8}


def get_style_for_zoom_level(zoom_level):
    font_size = font_sizes.get(zoom_level)
    return [{'selector': 'thead th',
             'props': [('background-color', '#9cd4e2'), ('text-align', 'center'), ('font-family', 'Times New Roman'),
                       ('font-size', str(font_size))]},
            {'selector': 'tbody td',
             'props': [('text-align', 'center'), ('font-family', 'Times New Roman'), ('font-size', str(font_size))]}]

def convert_html(document, csv_name):
    csv = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "documents", csv_name))
    total_row_count = csv.shape[0]
    number_of_subtables = math.ceil(total_row_count / rows_per_image)

    df = csv[0 : rows_per_image]
    convert_subtable_html(df, csv_name, 0)
    
    if number_of_subtables > 1:
        df = csv[rows_per_image : 2 * rows_per_image]
        convert_subtable_html(df, csv_name, 1)
    
    add_subtable_entry(document, csv_name, 0)
    create_tiles(csv_name, 0, 10)

    t = threading.Thread(target=convert_remaining_html, args=(document, csv_name, csv, number_of_subtables))
    t.start()

def convert_remaining_html(document, csv_name, csv, number_of_subtables):
    threads = []
    if number_of_subtables > 2:
        for subtable_number in range(2, number_of_subtables):
            df = csv[subtable_number * rows_per_image: (subtable_number * rows_per_image) + rows_per_image]
            t = threading.Thread(target=convert_subtable_html, args=(df, csv_name, subtable_number))
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    for subtable_number in range(1, number_of_subtables):
        add_subtable_entry(document, csv_name, subtable_number)

    for subtable_number in range(0, number_of_subtables):
        for zoom_level in range(6, 11):
            t = threading.Thread(target=create_tiles, args=(csv_name, subtable_number, zoom_level))
            t.start()

def convert_subtable_html(df, csv_name, subtable_number):
    pd.set_option('max_colwidth', 5000)
    df = df.astype(str)
    #df = df.astype(str).apply(lambda x: x.str[:max_chars_per_column])
    html = df.style.set_table_styles(get_style_for_zoom_level(10)).hide_index().render()
    options = {
        'quality' : '60'
    }
    imgkit.from_string(html, os.path.join(settings.MEDIA_ROOT, "documents", csv_name + str(subtable_number) + '.jpg'),
                        options=options)
    #img = imgkit.from_string(html, False)

def add_subtable_entry(document, csv_name, subtable_number):
    if subtable_number == 0:
        img_no = 0
        end_row_pix = 0
    else:
        td1 = TiledDocument.objects.get(document__file_name=csv_name, subtable_number=subtable_number-1, zoom_level=6)
        img_no = td1.end_image
        end_row_pix = td1.end_row

    img1_path = os.path.join(settings.MEDIA_ROOT, 'documents', csv_name + str(img_no) + '.jpg')
    img1 = Image.open(img1_path)
    w1, h1 = img1.size

    if end_row_pix == h1:
        return

    max_tile_size = 2 ** 12
    number_of_rows = int(math.ceil((h1 - end_row_pix) / (max_tile_size * 1.0)))
    number_of_cols = int(math.ceil(w1 / (max_tile_size * 1.0)))

    diff = max_tile_size * number_of_rows - h1 + end_row_pix
    img2_path = os.path.join(settings.MEDIA_ROOT, "documents", csv_name + str(img_no + 1) + '.jpg')
    
    if os.path.isfile(img2_path):
        img2 = Image.open(img2_path)
        w2, h2 = img2.size
        if h2 < diff:
            for zoom_level in range(6, 11):
                tile_size = 2 ** (18 - zoom_level)
                nr = int(math.ceil((h1 - end_row_pix + h2) / (tile_size * 1.0)))
                nc = int(math.ceil(max(w1, w2) / (tile_size * 1.0)))
                td2 = TiledDocument(document=document, zoom_level=zoom_level, subtable_number=subtable_number,
                                    end_image=img_no+1, end_row=h2, tile_count_on_x=nc, tile_count_on_y=nr,
                                    total_tile_count=nc*nr, profile_file_name=csv_name[:-4] + ".html")
                td2.save()
        else:
            for zoom_level in range(6, 11):
                tile_size = 2 ** (18 - zoom_level)
                nr = int(math.ceil((h1 - end_row_pix + diff) / (tile_size * 1.0)))
                nc = int(math.ceil(max(w1, w2) / (tile_size * 1.0)))
                td2 = TiledDocument(document=document, zoom_level=zoom_level, subtable_number=subtable_number,
                                    end_image=img_no+1, end_row=diff, tile_count_on_x=nc, tile_count_on_y=nr,
                                    total_tile_count=nc*nr, profile_file_name=csv_name[:-4] + ".html")
                td2.save()

    else:
        for zoom_level in range(6, 11):
            tile_size = 2 ** (18 - zoom_level)
            nr = int(math.ceil(h1 / (tile_size * 1.0)))
            nc = int(math.ceil(w1 / (tile_size * 1.0)))
            td2 = TiledDocument(document=document, zoom_level=zoom_level, subtable_number=subtable_number,
                                end_image=img_no, end_row=h1, tile_count_on_x=nc, tile_count_on_y=nr,
                                total_tile_count=nc*nr, profile_file_name=csv_name[:-4] + ".html")
            td2.save()

def create_tiles(csv_name, subtable_number, zoom_level):
    if subtable_number == 0:
        img1_no = 0
        img1_end_px = 0
    else:
        td1 = TiledDocument.objects.get(document__file_name=csv_name, subtable_number=subtable_number-1,
                                        zoom_level=zoom_level)
        img1_end_px = td1.end_row
        img1_no = td1.end_image

    try:
        td2 = TiledDocument.objects.get(document__file_name=csv_name, subtable_number=subtable_number, zoom_level=zoom_level)
    except ObjectDoesNotExist:
        return
    img2_no = td2.end_image
    img2_end_px = td2.end_row

    img1_path = os.path.join(settings.MEDIA_ROOT, "documents", csv_name + str(img1_no) + '.jpg')
    img1 = np.array(Image.open(img1_path))

    if img1_no == img2_no:
        img = img1[img1_end_px : img1.shape[0], :]
    else:
        img2_path = os.path.join(settings.MEDIA_ROOT, "documents", csv_name + str(img2_no) + '.jpg')
        img2 = np.array(Image.open(img2_path))

        img = np.full((img1.shape[0] - img1_end_px + img2_end_px, max(img1.shape[1], img2.shape[1]), 3), 
                        255, dtype=np.uint8)

        img[0:img1.shape[0]-img1_end_px, 0:img1.shape[1]] = img1[img1_end_px:img1.shape[0], 0:img1.shape[1]]
        img[img1.shape[0]-img1_end_px:img.shape[0],0:img2.shape[1]] = img2[0:img2_end_px, 0:img2.shape[1]]

    tile_size = 2 ** (18 - zoom_level)
    number_of_rows = int(math.ceil(img.shape[0] / (tile_size * 1.0)))
    number_of_cols = int(math.ceil(img.shape[1] / (tile_size * 1.0)))

    if img.shape[1] % tile_size != 0 or img.shape[0] % tile_size != 0:
        img = pad_img(img, tile_size * number_of_rows, tile_size * number_of_cols)

    tile_count = 0

    tile_dir = os.path.join(settings.MEDIA_ROOT, 'tiles', str(zoom_level), str(subtable_number))
    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir)

    for i in range(0, number_of_rows):
        for j in range(0, number_of_cols):
            cropped_img = img[i * tile_size : (i+1) * tile_size, j * tile_size : (j+1) * tile_size]
            tile_path = os.path.join(tile_dir, csv_name + str(tile_count).zfill(5).replace("-", "0") + ".jpg")
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


def coordinate(x, y, tile_count_on_x):
    # i + nx * (j + ny * k)
    # print("{0}\n".format(tile_count_on_x))
    tile_number = x + tile_count_on_x * y
    return str(tile_number).zfill(5).replace("-", "0")
