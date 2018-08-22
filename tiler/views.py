import math
import queue
import os
import threading
import io

import imgkit
import pandas as pd
from PIL import Image
from django.conf import settings
from django.db.models import F, Sum, Max
from django.http import HttpResponse, JsonResponse
from django.utils.cache import add_never_cache_headers

from tiler.models.Document import TiledDocument
import numpy as np

max_chars_per_column = 2000
st_images = {}
st_images_max = 25
cv = threading.Lock()
write_q = queue.Queue()
progressStatus = "Uploading File"
progressValue = 10

def index(request):
    return HttpResponse("Index page of tiler")

def get_subtable_number(file_name, x, y):
    """
    Given coordinates of a tile, this function computes the subtable image number for the requested tile.
    Args:
        file_name (str): Name of the csv files
        x, y (int): The x, y coordinates of the tile at the highest zoom level
    Returns:
        Subtable Number (Int): The subtable image that the requested tile belongs to.
        -1 if no tile image is available for the requested coordinates.
        y: The y coordinate is adjusted relative to the coordinate system centered at
        top left corner of the subtable image. No change is required for the x coordinate
    """
    if x < 0 or y < 0:
        return -1, None

    if y > TiledDocument.objects.filter(document__file_name=file_name).aggregate(Sum('tile_count_on_y'))['tile_count_on_y__sum']:
        return -1, None
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
            return -1, None
    
    tile_count_on_x = tile_details[subtable_number].tile_count_on_x
    if x >= tile_count_on_x:
        return -1, None

    return subtable_number, y

def tile_request(request, id, z, x, y):
    """
    Tile service that responds to leaflet JS tile requests. Takes in x, y coordinates and
    zoom level of requested tiles and returns the appropriate tile. Tiling is done in memory.
    Each request is mapped to the appropriate subtable image. If the image is already available
    in memory, appropriate tile is cut out and returned as a JPEG, else the image is loaded into
    memory first. As multiple simultaneous tile requests are made to the same subtable image, a 
    lock is used to ensure that only one thread reads the image from disk into memory. The other
    threads spin wait for this thread to complete and then use the loaded image.
    
    Args:
        request (HTTPRequestObject)
        x, y, z (int): Coordinates and zoom level of tile
    Returns: 
        HTTPResponse Object containing the tile image
    """
    file_name = request.GET.get("file")
    z = int(z) - 3
    x = int(x) * (2 ** (10 - z))
    y = int(y) * (2 ** (10 - z))

    subtable_number, y = get_subtable_number(file_name, x, y)
    if subtable_number == -1:
        return empty_response()
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
        return error_response()

def progress(request):
    """
    Backend service for the upload file progress bar. Updates are made to the global variables
    'progressValue' and 'progressStatus' during the subtable image generation process. These updates
    are sent to the client side by this function

    Args:
        request (HTTPRequestObject)
    Returns:
        A JSON response containing a status message and a progress percentage for csv uploads.
    """
    return JsonResponse({'progressValue': progressValue, 'progressStatus': progressStatus})

def get_subtable_dimensions(csv):
    """
    Processes structure and data within csv files in order to calculate how many rows to use per subtable
    image and the width that should be set for the generated html table.
    Args:
        csv (Pandas.DataFrame): CSV data stored in a pandas dataframe
    Returns:
        The number of rows per image and width of html table to be generated
    """
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
    """
    Creates the subtable image for the initial few rows of the csv, adds a database record for it
    if required. The function then hands off the task of creating the remaining subtable images to 
    convert_remaining_html and returns, thus allowing users to browse the initial rows of the csv 
    while the rest of it is being processed.
    Args:
        document (DocumentObject): Object corresponding to the file's record in the database
        csv_name (string): Name of the csv file
    Returns:
        Number of rows and columns in the csv
    """
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
    """
    Creates subtable images for the remaining rows in the csv. The csv is converted to a set of images by calls to
    multi-threaded calls to convert_subtable_html using batches. The converted
    images are joined with the top portion of the next image by create_subtable_image in order to create evenly sized
    tiles. Writing each such subtable image to disk is considered a task and is added to a write queue. A number of worker 
    threads are started. Each worker thread picks a task from the write queue and performs it.
    Args:
        document
        csv_name
        rows_per_image, width
        img1
        start_row
        add_entries
    Return:
        None
    """
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
    """
    Converts a dataframe into an image. The dataframe is first converted to html and the html
    is the converted to an image using the 'from_string' of wkhtmltoimage library. The image is read into
    a numpy array and returned or stored into a results array if provided.
    Args:
        df
        csv_name
        subtable_number
        max_width
        results
    Returns:
        An image as a numpy array if a results array is provided as an argument. Else returns None.
    """
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
    """
    Creates subtable images with dimensions that are mutliples of tiles sizes at the lowest zoom level. Takes
    in two images as numpy arrays and rounds out the first image by concatenating the top part of the second 
    image.
    Args:
        img1, img2
        start_row
    Returns:
        img: A vertically concatenated and horizontally padded image with dimensions that are multiples of the largest
        tile size.
        diff: Number of rows of pixels from the second image that are concatenated with the first image. -1 is returned 
        if the second image does not exist or if it does not have sufficient rows of pixels to pad out the first image.
    """
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
    """
    Writes subtable images to disk. Called by worker threads.
    Args:
        pil_img
        subtable_path
    Returns:
        None
    """
    pil_img.save(subtable_path, 'jpeg', quality=60)
    print("{0} written".format(subtable_path))

def worker():
    """
    A worker abstraction that iterates over the write queue and performs the tasks present in it.
    """
    while True:
        item = write_q.get()
        if item is None:
            break
        write_subtable_image(item[0], item[1])
        write_q.task_done()

def add_subtable_entries(document, csv_name, start_st_no, images):
    """
    Adds bulk database entries corresponding to a list of subtable images.
    Args:
        document
        csv_name
        start_st_no
        images
    Returns:
        None
    """
    entries = []
    for i, img in enumerate(images):
        tile_size = 2 ** 12
        nrows = int(math.ceil(img.shape[0]/tile_size)) * (2 ** 4) 
        ncols = int(math.ceil(img.shape[1]/tile_size)) * (2 ** 4)
        entries.append(TiledDocument(document=document, subtable_number=start_st_no+i,
            tile_count_on_x=ncols, tile_count_on_y=nrows, total_tile_count=ncols*nrows))

    TiledDocument.objects.bulk_create(entries)

def pad_img(img, h, w):
    """
    Utility function in order to pad out an image to the required size
    Args:
        img
        h, w
    Returns:
        The padded image as a numpy array
    """
    height, width, channels = img.shape
    new_img = np.full((h, w, channels), 221, dtype=np.uint8)
    new_img[0:height, 0:width] = img
    return new_img

def empty_response():
    """
    Utility function for returning an empty tile as a HTTPResponse
    """
    red = Image.new('RGBA', (1, 1), (255, 0, 0, 0))
    response = HttpResponse(content_type="image/png")
    red.save(response, "png")
    return response

def error_response():
    """
    Utility function for returning a red tile as an error response
    """
    red = Image.new('RGB', (256, 256), (255, 0, 0))
    response = HttpResponse(content_type="image/jpg")
    red.save(response, "jpeg")
    return response
