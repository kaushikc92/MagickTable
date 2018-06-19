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
    x = int(x) - start_x.get(z)
    y = int(y) - start_y.get(z)
    if x < 0 or y < 0:
        return empty_response()
    x = int(math.fabs(x))
    y = int(math.fabs(y))

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


def convert_subtable_html(df, csv_name, subtable_number, zoom_level, starting_tile_number=0):
    pd.set_option('max_colwidth', 40)
    df = df.astype(str).apply(lambda x: x.str[:max_chars_per_column])
    html = df.style.set_table_styles(get_style_for_zoom_level(10)).render()
    tile_count = starting_tile_number
    imgkit.from_string(html, os.path.join(settings.MEDIA_ROOT, "documents", str(zoom_level),
                                          csv_name + str(subtable_number) + '.jpg'))
    number_of_cols, number_of_rows, tile_count = slice_image(csv_name, os.path.join(settings.MEDIA_ROOT, "documents",
                                                                                    str(zoom_level),
                                                                                    csv_name + str(
                                                                                        subtable_number) + '.jpg'),
                                                             tile_count, zoom_level)
    print("zoom level " + str(zoom_level))
    tiled_document = TiledDocument.objects.get(document__file_name=csv_name, zoom_level=zoom_level)
    tiled_document.tile_count_on_y = F('tile_count_on_y') + number_of_rows
    tiled_document.zoom_level = zoom_level
    if tiled_document.tile_count_on_x == 0:
        tiled_document.tile_count_on_x = number_of_cols
    tiled_document.total_tile_count = F('total_tile_count') + tile_count
    tiled_document.save()

    return number_of_cols, number_of_rows, tile_count


def convert_html(document, csv_name):
    csv = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "documents", csv_name))
    total_row_count = csv.shape[0]
    x = 0
    tiled_document = TiledDocument(document=document, tile_count_on_x=0, tile_count_on_y=0,
                                   total_tile_count=0, profile_file_name=csv_name[:-4] + ".html", zoom_level=10)
    tiled_document.save()
    df = csv[x:x + rows_per_image]
    # convert the first set to get a count of the tiles per set
    number_of_cols, number_of_rows, tile_count = convert_subtable_html(df, csv_name, subtable_number=0,
                                                                       starting_tile_number=0, zoom_level=10)
    number_of_subtables = math.ceil(total_row_count / rows_per_image)

    thread_pool = multiprocessing.Pool(processes=multiprocessing_limit)

    # start from 1 because we already did the first set
    for subtable_number in range(1, number_of_subtables):
        df = csv[subtable_number * rows_per_image: (subtable_number * rows_per_image) + rows_per_image]
        thread_pool.apply_async(convert_subtable_html,
                                args=(df, csv_name, subtable_number, 10, tile_count * subtable_number))

    profile_df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "documents", csv_name))
    profile = pf.ProfileReport(profile_df)
    profile_output_path = os.path.join(settings.MEDIA_ROOT, "documents", csv_name[:-4] + "_profile.html")
    profile.to_file(outputfile=profile_output_path)

    for i in range(1, 10):
        convert_html_2(document, csv_name, zoom_level=i)


def convert_html_2(document, csv_name, zoom_level):
    csv = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "documents", csv_name))
    total_row_count = csv.shape[0]
    x = 0
    tiled_document = TiledDocument(document=document, tile_count_on_x=0, tile_count_on_y=0,
                                   total_tile_count=0, profile_file_name=csv_name[:-4] + ".html", zoom_level=zoom_level)
    tiled_document.save()
    df = csv[x:x + rows_per_image]
    # convert the first set to get a count of the tiles per set
    number_of_cols, number_of_rows, tile_count = convert_subtable_html(df, csv_name, subtable_number=0,
                                                                       starting_tile_number=0, zoom_level=zoom_level)
    number_of_subtables = math.ceil(total_row_count / rows_per_image)

    thread_pool = multiprocessing.Pool(processes=multiprocessing_limit)

    # start from 1 because we already did the first set
    for subtable_number in range(1, number_of_subtables):
        df = csv[subtable_number * rows_per_image: (subtable_number * rows_per_image) + rows_per_image]
        thread_pool.apply_async(convert_subtable_html,
                                args=(df, csv_name, subtable_number, zoom_level, tile_count * subtable_number))

    profile_df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, "documents", csv_name))
    profile = pf.ProfileReport(profile_df)
    profile_output_path = os.path.join(settings.MEDIA_ROOT, "documents", csv_name[:-4] + "_profile.html")
    profile.to_file(outputfile=profile_output_path)


def empty_response():
    red = Image.new('RGBA', (1, 1), (255, 0, 0, 0))
    response = HttpResponse(content_type="image/png")
    red.save(response, "png")
    return response


def coordinate(x, y, tile_count_on_x):
    # i + nx * (j + ny * k)
    # print("{0}\n".format(tile_count_on_x))
    tile_number = x + tile_count_on_x * y
    return str(tile_number).zfill(3).replace("-", "0")
