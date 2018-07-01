import math
import multiprocessing
import os

import imgkit
import pandas as pd
import pandas_profiling as pf
from PIL import Image
from django.conf import settings
from django.db.models import F
from django.http import HttpResponse
from django.utils.cache import add_never_cache_headers

from convertoimg.converttoimg import slice_image
from tiler.models.Document import TiledDocument
import pdb
import cv2
import numpy as np

def index(request):
    return HttpResponse("Index page of tiler")


rows_per_image = 500

# todo: this is got from what leaflet sends as x & y
# TODO: This changes for every Zoom level!
# start_x = 4091
# start_y = 2722

# According to mapbox: for a point on increasing zoom level x, y double
# => 1,2 at zoom level 3 is 2,4 at zoom level 4
start_z = 3
start_x = {10: 4096, 9: 2048, 11: 8184, 8: 1024, 7: 512, 6: 256, 5: 128, 4: 64, 3: 32, 2: 16, 1: 8}
start_y = {10: 4096, 9: 2048, 11: 5447, 8: 1024, 7: 512, 6: 256, 5: 128, 4: 64, 3: 32, 2: 16, 1: 8}

# 8 1020,680
# 7 508, 338
# 6  252 , 168
# 5 124,83
# 4 60,40
# 3 28,19
# 2
# 1

# TODO: Find correct value. multiprocessing.cpu_count()-1 as a heuristic
multiprocessing_limit = multiprocessing.cpu_count() - 1

# TODO: change this based on zoom level
max_chars_per_column = 40


# this is the function that will return a tile based on x, y, z
# TODO try different image formats
# TODO: add an inmemory caching layer ex: memcached
# TODO: see if we can bunch a number of requests together rather than 1 per tile
# TODO mapbox uses 256 by 256 squares: so we need to pad our generated image to fit that
def tile_request(request, id, z, x, y):
    file_name = request.GET.get("file")
    z = int(z) - start_z
    print("{0},{1} zoom {2}".format(x, y, z))
    #return empty_response()
    # if int(z) < 0 or int(z) > 10:
    #     return empty_response()
    #x = int(x) - start_x.get(z)
    #y = int(y) - start_y.get(z)
    x = int(x)
    y = int(y)
    if x < 0 or y < 0:
        return empty_response()
    #x = int(math.fabs(x))
    #y = int(math.fabs(y))

    tiled_document = TiledDocument.objects.get(document__file_name=file_name, zoom_level=z)
    i = coordinate(x, y, tiled_document.tile_count_on_x)
    if int(i) > tiled_document.total_tile_count or int(x) >= tiled_document.tile_count_on_x or \
            int(y) >= tiled_document.tile_count_on_y:
        return empty_response()

    path = os.path.join(settings.MEDIA_ROOT, 'tiles', str(z), file_name + i + ".jpg");

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

    for zoom_level in range(6,11):
        tiled_document = TiledDocument(document=document, tile_count_on_x=0, tile_count_on_y=0,
                                   total_tile_count=0, profile_file_name=csv_name[:-4] + ".html", zoom_level=zoom_level)
        tiled_document.save()
    
    number_of_subtables = math.ceil(total_row_count / rows_per_image)

    for subtable_number in range(0, number_of_subtables):
        df = csv[subtable_number * rows_per_image: (subtable_number * rows_per_image) + rows_per_image]
        convert_subtable_html(df, csv_name, subtable_number=subtable_number)

    number_of_subtables = adjust_subtable_images(csv_name, number_of_subtables)
    create_tiles(csv_name, number_of_subtables)

def convert_subtable_html(df, csv_name, subtable_number):
    pd.set_option('max_colwidth', 40)
    df = df.astype(str)
    html = df.style.set_table_styles(get_style_for_zoom_level(10)).hide_index().render()
    imgkit.from_string(html, os.path.join(settings.MEDIA_ROOT, "documents", csv_name + str(subtable_number) + '.jpg'))

def adjust_subtable_images(csv_name, number_of_subtables):
    subtable_number = 0
    nst = number_of_subtables
    while subtable_number < nst:
        img1_path = os.path.join(settings.MEDIA_ROOT, "documents", csv_name + str(subtable_number) + '.jpg')
        img1 = cv2.imread(img1_path)
        tile_size = 2 ** (18 - 6)
        number_of_rows = int(math.ceil(img1.shape[0] / (tile_size * 1.0)))
        number_of_cols = int(math.ceil(img1.shape[1] / (tile_size * 1.0)))
        if subtable_number == nst - 1:
            img1 = pad_img(img1, tile_size * number_of_rows, tile_size * number_of_cols)
        else:
            img2_path = os.path.join(settings.MEDIA_ROOT, "documents", csv_name + str(subtable_number + 1) + '.jpg')
            img2 = cv2.imread(img2_path)
            diff = tile_size * number_of_rows - img1.shape[0]
            if img2.shape[0] < diff:
                number_of_subtables = number_of_subtables - 1
                subtable_number = subtable_number + 1
                img = np.zeros((img1.shape[0] + img2.shape[0], img1.shape[1], 3), dtype=np.uint8)
                img[0:img1.shape[0],0:img1.shape[1]] = img1
                img[img1.shape[0]: img1.shape[0] + img2.shape[0], 0:img1.shape[1]] = img2
                img = pad_img(img, tile_size * number_of_rows, img.shape[1])
            else:
                diff_img = img2[0 : diff, 0: img2.shape[1]]
                img2 = img2[diff: img2.shape[0], 0 : img2.shape[1]]
                cv2.imwrite(img2_path, img2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                img = np.zeros((tile_size * number_of_rows, img1.shape[1], 3), dtype=np.uint8)
                img[0:img1.shape[0],0:img1.shape[1]] = img1
                img[img1.shape[0]: img1.shape[0] + diff, 0:img1.shape[1]] = diff_img
            img1 = img
            if img1.shape[1] % tile_size != 0:
                img1 = pad_img(img1, img1.shape[0], tile_size * number_of_cols)
        cv2.imwrite(img1_path, img1, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        subtable_number = subtable_number + 1

    return number_of_subtables

def create_tiles(csv_name, number_of_subtables):
    for subtable_number in range(0, number_of_subtables):
        for zoom_level in range(6,11):
            img_path = os.path.join(settings.MEDIA_ROOT, "documents", csv_name + str(subtable_number) + '.jpg')
            img = cv2.imread(img_path)
            tile_size = 2 ** (18 - zoom_level)
            number_of_rows = int(math.ceil(img.shape[0] / (tile_size * 1.0)))
            number_of_cols = int(math.ceil(img.shape[1] / (tile_size * 1.0)))

            tiled_document = TiledDocument.objects.get(document__file_name=csv_name, zoom_level=zoom_level)
            tile_count = tiled_document.total_tile_count
            tiled_document.tile_count_on_y = F('tile_count_on_y') + number_of_rows
            tiled_document.zoom_level = zoom_level
            if tiled_document.tile_count_on_x == 0:
                tiled_document.tile_count_on_x = number_of_cols
            tiled_document.total_tile_count = F('total_tile_count') + number_of_rows * number_of_cols
            tiled_document.save()
            
            for i in range(0, number_of_rows):
                for j in range(0, number_of_cols):
                    cropped_img = img[i * tile_size : (i+1) * tile_size, j * tile_size : (j+1) * tile_size]
                    tile_path = os.path.join(settings.MEDIA_ROOT, 'tiles', str(zoom_level),
                                                csv_name + str(tile_count).zfill(5).replace("-", "0") + ".jpg")
                    cropped_img = cv2.resize(cropped_img, (256, 256))
                    cv2.imwrite(tile_path, cropped_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    tile_count = tile_count + 1
            
            print("Subtable {0}, Level {1} done".format(subtable_number, zoom_level))

def pad_img(img, h, w):
    height, width, channels = img.shape
    bottom_padding = h - height
    right_padding = w - width
    img = cv2.copyMakeBorder(img, top=0, bottom=bottom_padding, left=0, right=right_padding,
        borderType=cv2.BORDER_CONSTANT, value=[221, 221, 221])
    return img

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
