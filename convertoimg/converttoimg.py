import math
import os

import pandas as pd
import six
from django.conf import settings

import pdb

max_pixel_col = 5
width_per_char = 0.1
height_per_char = 0.2


# num_rows = df.shape[0]
# num_cols = df.shape[1]


def render_mpl_table(data, csv_name, col_width=10.0, row_height=0.625, font_size=5,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    num_rows = data.shape[0]
    num_cols = data.shape[1]

    # print("hello world")
    width_list = [0 for x in range(num_cols)]
    # calculate width for each column
    for x in range(0, num_cols):
        for y in range(0, num_rows):
            width = len(str(data.iloc[y][x])) * width_per_char
            if width < max_pixel_col:
                width_list[x] = max(width_list[x], width)
            else:
                width_list[x] = max_pixel_col

    height_list = [0 for x in range(num_rows)]
    for x in range(0, num_rows):
        for y in range(0, num_cols):
            height = len(str(data.iloc[x][y])) / 50 * height_per_char
            if len(str(data.iloc[x][y])) % 50 != 0:
                height += height_per_char
            height_list[x] = max(height_list[x], height)

    total_height = sum(height_list)
    total_width = sum(width_list)

    if ax is None:
        fig, ax = plt.subplots(figsize=[total_width, total_height], dpi=200)
        ax.axis('off')

    values = data.values

    for x in range(0, num_rows):
        for y in range(0, num_cols):
            original_st = str(values[x][y])
            if len(original_st) > 50:
                st = ""
                for index in range(0, len(original_st)):
                    st += original_st[index]
                    if index % 50 == 0 and index != 0:
                        st += '\n'
                values[x][y] = st

    mpl_table = ax.table(cellText=values, cellLoc="center", bbox=bbox, colLabels=data.columns, colWidths=width_list,
                         **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w', wrap=True)
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    plt.savefig(os.path.join(settings.MEDIA_ROOT, csv_name + '.png'))
    slice_image(csv_name, os.path.join(settings.MEDIA_ROOT, csv_name + '.png'))
    return ax

def convert(csv_name):
    df = pd.read_csv(os.path.join(settings.MEDIA_ROOT, csv_name))
    ax = render_mpl_table(df, header_columns=0, col_width=2.0, csv_name=csv_name)
