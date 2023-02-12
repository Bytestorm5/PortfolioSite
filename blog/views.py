from django.shortcuts import render
from django.http import HttpResponse
from .models import BlogPost

from django.shortcuts import get_object_or_404, render

def index(request):
    latest_blog_list = BlogPost.objects.order_by('-pub_date')[:5]
    #template = loader.get_template("polls/index.html")
    context = {
        "latest_blog_list": latest_blog_list
    }
    return render(request, "blog/index.html", context) #HttpResponse(template.render(context, request))

def post(request, post_id):
    post = get_object_or_404(BlogPost, pk=post_id)
    return render(request, "blog/post.html", {"post": post})