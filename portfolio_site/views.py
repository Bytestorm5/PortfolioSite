from django.shortcuts import render
from django.http import HttpResponse

from django.shortcuts import get_object_or_404, render

def index(request):
    return render(request, "portfolio_site/base_section.html")
