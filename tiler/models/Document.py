from django.db import models


class Document(models.Model):
    # TODO change this to username?
    file_name = models.CharField(max_length=200)
    docfile = models.FileField(upload_to='documents/')
    last_access = models.DateTimeField(auto_now=True)


class TiledDocument(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    subtable_number = models.IntegerField()
    tile_count_on_x = models.IntegerField()
    tile_count_on_y = models.IntegerField()
    total_tile_count = models.IntegerField()
