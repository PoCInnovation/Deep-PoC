import sys
from django.shortcuts import redirect
from django.shortcuts import render
from django.http import HttpResponse
from os import path, getcwd
from django.urls import reverse
sys.path.insert(0, getcwd() + "/bob")
from test_opencvbob import main_opencv

def main(request):
    if (request.method == 'POST' and request.FILES):
        name = request.FILES['file'].name
        if (name.split(".")[-1] != "mp4") or request.FILES['file'].size > 100000000:
            return render(request, 'main.html')
        print(request.FILES['file'].size)
        with open('vieo.mp4', 'wb+') as destination:
            for chunk in request.FILES['file'].chunks():
                destination.write(chunk)
        result = main_opencv("../eye_model/10.0,1.0,1.0,10")
        if (result >= 0.5):
            return render(request, 'real.html', {'fileName': name, "status": "real"})
        else:
            return render(request, 'main.html', {'fileName': name, "status": "fake"})
    return render(request, 'main.html')

def real(request):
    return render(request, 'real.html')

def fake(request):
    return render(request, 'fake.html')