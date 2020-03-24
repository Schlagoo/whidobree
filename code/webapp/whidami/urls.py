from django.urls import path
from django.conf.urls import url, include
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

from .views import *

urlpatterns = [
    path('', homepage_view, name='home'),
    path('result', result_view, name='result'),
    path('about', about_view, name='about'),
    path('imprint', imprint_view, name='imprint'),
    path('privacy', privacy_view, name='privacy')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)