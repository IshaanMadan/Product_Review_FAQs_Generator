from django.urls import path
from . import views

urlpatterns=[
    path('review',views.review,name='review'),
    path('api/review',views.get_bert,name="api/review"),
        # path('',views.home,name='home'),
    path('api/ditillbertreview',views.get_ditillbertreview,name="api/ditillbertreview"),

    path('distillbert',views.distillbert,name="distillbert")

]