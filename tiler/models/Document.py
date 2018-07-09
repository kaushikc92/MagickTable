from django.db import models


class Document(models.Model):
    # TODO change this to username?
    file_name = models.CharField(max_length=200)
    docfile = models.FileField(upload_to='documents/')


class TiledDocument(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    subtable_number = models.IntegerField()
    zoom_level = models.IntegerField()
    end_image = models.IntegerField()
    end_row = models.IntegerField()
    tile_count_on_x = models.IntegerField()
    tile_count_on_y = models.IntegerField()
    total_tile_count = models.IntegerField()
    profile_file_name = models.CharField(max_length=200)
