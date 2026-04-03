from django.urls import path
from . import views

app_name = 'rankingfacts'
urlpatterns = [
    path('', views.base, name='base'),
    path('upload_data/', views.upload_data, name='upload_data'),
    path('feature_select/', views.feature_select, name='feature_select'),
    path('gene_analysis/', views.gene_analysis, name='gene_analysis'),
    path('json_gene_ranking/', views.json_gene_ranking, name='json_gene_ranking'),
]
