from django.http import HttpResponse
from django.shortcuts import render, redirect

from tiler.forms import DocumentForm
from tiler.models.Document import Document

def file_with_same_name_exists(request):
    return HttpResponse(False)

def file_exists_in_db(file_name):
    """
    Check if file already exists in database
    Args:
        file_name
    Returns:
        True if file exists, else False
    """
    docs = Document.objects.filter(file_name=file_name)
    if not docs:
        return False
    else:
        return True

def list_files(request):
    """
    Request handler for listing csv files on the index page. Handles GET requests used for listing all the
    files and POST requests for adding a new file.
    Args:
        request: The HTTP request object.
    Returns: HTTP response object that contains html for the index page in case of GET request. Redirects to
    the csv viewer page in case of POST request.
    """
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid() and not file_exists_in_db(request.FILES['docfile'].name):
            newdoc = Document(file_name=request.FILES['docfile'].name, docfile=request.FILES['docfile'],
                rows=0, columns=0)
            newdoc.save()
            return redirect('/map/leaflet?file=' + request.FILES['docfile'].name)
    else:
        form = DocumentForm()
    documents = Document.objects.all()
    return render(request, 'list.html', {'documents': documents, 'form': form})
