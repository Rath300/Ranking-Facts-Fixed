from django.http import HttpResponse


def index(request):
    return HttpResponse("Synthesizer placeholder", content_type="text/plain")



