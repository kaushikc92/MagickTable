from django.contrib import admin

from .models.Document import Document
from .models.Document import TiledDocument

# Register your models here.

admin.site.register(Document)
admin.site.register(TiledDocument)
