from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse


def base(request):
    # Redirect root to rankingfacts landing page
    return HttpResponseRedirect(reverse('rankingfacts:base'))