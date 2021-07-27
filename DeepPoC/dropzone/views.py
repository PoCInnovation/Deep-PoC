from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import HttpResponse, JsonResponse
from .models import dropzone
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image
# Create your views here.

class MainView(TemplateView):
    template_name = 'docs/main.html'

def file_upload_view(request):
    print("ddddddddddddddddddd", request.FILES)
    if (request.method == 'POST'):
        with open('name.txt', 'wb+') as destination:
            for chunk in request.FILES['file'].chunks():
                destination.write(chunk)
    return JsonResponse({'post': 'false'})