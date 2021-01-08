from django.urls import path
from . import views

urlpatterns=[
    path('review',views.review,name='review'),
    path('api/review',views.get_bert,name="api/review"),
    path('api/ditillbertreview',views.get_ditillbertreview,name="api/ditillbertreview"),
    path('distillbert',views.distillbert,name="distillbert"),
    path('api/mobilebertreview',views.get_mobilebertreview,name="api/mobilebertreview"),
    path('mobilebert',views.mobilebert,name="mobilebert")

]