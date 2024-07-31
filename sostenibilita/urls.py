from django.urls import path
from sostenibilita import views

urlpatterns = [
    path('', views.index, name='index'),
    path('sostenibilita/trainingFile', views.trainingFile, name='trainingFile'),
    path('sostenibilita/uploadFile', views.uploadFile, name='uploadFile'),
    path('sostenibilita/machineLearningTraining',views.machineLearningTraining, name='machineLearningTraining'),
    path('sostenibilita/modelView', views.modelView, name='modelView'),
    path('sostenibilita/uploadFileSocial', views.uploadFileSocial, name='uploadFileSocial'),
    path('sostenibilita/uploadDataset',views.uploadDataset, name='uploadDataset'),
    path('sostenibilita/redirectUploadDataset',views.redirectUploadDataset, name='redirectUploadDataset'),
    path('sostenibilita/downloadFileEmission',views.downloadFileEmission, name='downloadFileEmission'),
    path('sostenibilita/redirectModelsPreaddestratedSocial',views.redirectModelsPreaddestratedSocial, name='redirectModelsPreaddestratedSocial'),
    path('sostenibilita/loadDefaultModel',views.loadDefaultModel,name='loadDefaultModel'),
    path('sostenibilita/redirectResult',views.redirectResult,name='redirectResult'),
    path('sostenibilita/deleteModel',views.deleteModel,name='deleteModel'),
]