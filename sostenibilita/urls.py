from django.urls import path
from sostenibilita import views

urlpatterns = [
    path('', views.index, name='index'),
    path('sostenibilita/trainingFile', views.trainingFile, name='trainingFile'),
    path('sostenibilita/uploadFile', views.uploadFile, name='uploadFile'),
    path('sostenibilita/machineLearningTraining',views.machineLearningTraining, name='machineLearningTraining'),
    path('sostenibilita/modelView', views.modelView, name='modelView')

]