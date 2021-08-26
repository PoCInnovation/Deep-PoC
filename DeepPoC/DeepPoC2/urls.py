from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^$', views.main, name='main'),
    url(r'^fake/$', views.fake, name='fake'),
    url(r'^real/$', views.real, name='real'),
]