from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload/', views.upload_audio, name='upload_audio'),
    path('jobs/', views.all_jobs, name='all_jobs'),
    path('jobs/<uuid:job_id>/', views.job_status, name='job_status'),
    path('jobs/<uuid:job_id>/result/', views.job_result, name='job_result'),
    path('jobs/<uuid:job_id>/delete/', views.job_delete, name='job_delete'),
    path('api/jobs/<uuid:job_id>/status/', views.job_status_api, name='job_status_api'),
    path('metrics/', views.metrics_view, name='metrics'),
]
