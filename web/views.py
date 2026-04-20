import json
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.utils import timezone

from .forms import RegisterForm, LoginForm, AudioUploadForm
from .models import AudioJob


def index(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return redirect('login')


def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome, {user.first_name or user.username}! Your account has been created.')
            return redirect('dashboard')
    else:
        form = RegisterForm()
    return render(request, 'web/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
    else:
        form = LoginForm()
    return render(request, 'web/login.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def dashboard(request):
    jobs = AudioJob.objects.filter(user=request.user)
    stats = {
        'total': jobs.count(),
        'completed': jobs.filter(status='completed').count(),
        'processing': jobs.exclude(status__in=['completed', 'failed', 'uploaded']).count(),
        'failed': jobs.filter(status='failed').count(),
    }
    recent_jobs = jobs[:10]
    upload_form = AudioUploadForm()
    return render(request, 'web/dashboard.html', {
        'jobs': recent_jobs,
        'stats': stats,
        'upload_form': upload_form,
    })


@login_required
@require_POST
def upload_audio(request):
    form = AudioUploadForm(request.POST, request.FILES)
    if form.is_valid():
        audio_file = form.cleaned_data['audio_file']
        max_speakers = int(form.cleaned_data['max_speakers'])

        job = AudioJob.objects.create(
            user=request.user,
            original_file=audio_file,
            original_filename=audio_file.name,
            max_speakers=max_speakers,
            status='uploaded',
            status_message='File uploaded. Queuing for processing...',
        )

        # Start background task
        import threading
        def run_in_background(job_id):
            try:
                from pipeline.orchestrator import run_pipeline
                run_pipeline(job_id)
            except Exception as e:
                from web.models import AudioJob
                AudioJob.objects.filter(id=job_id).update(
                    status='failed',
                    error_message=str(e),
                    status_message='Processing failed.',
                )

        job.status_message = 'Processing started...'
        job.save()
        t = threading.Thread(target=run_in_background, args=(str(job.id),), daemon=True)
        t.start()

        return redirect('job_status', job_id=str(job.id))
    else:
        messages.error(request, 'Upload failed: ' + str(form.errors))
        return redirect('dashboard')


@login_required
def job_status(request, job_id):
    job = get_object_or_404(AudioJob, id=job_id, user=request.user)
    return render(request, 'web/status.html', {'job': job})


@login_required
def job_result(request, job_id):
    job = get_object_or_404(AudioJob, id=job_id, user=request.user)
    if job.status != 'completed':
        return redirect('job_status', job_id=str(job.id))
    return render(request, 'web/result.html', {'job': job})


@login_required
def job_status_api(request, job_id):
    """AJAX endpoint for polling job status."""
    job = get_object_or_404(AudioJob, id=job_id, user=request.user)
    return JsonResponse({
        'status': job.status,
        'progress': job.progress,
        'status_message': job.status_message,
        'num_speakers': job.num_speakers,
        'error_message': job.error_message,
        'completed': job.status == 'completed',
        'failed': job.status == 'failed',
        'result_url': f'/jobs/{job_id}/result/' if job.status == 'completed' else None,
    })


@login_required
def job_delete(request, job_id):
    job = get_object_or_404(AudioJob, id=job_id, user=request.user)
    if request.method == 'POST':
        # Delete files
        for field in [job.original_file, job.processed_wav, job.dubbed_audio]:
            if field:
                try:
                    if os.path.exists(field.path):
                        os.remove(field.path)
                except Exception:
                    pass
        job.delete()
        messages.success(request, 'Job deleted successfully.')
    return redirect('dashboard')


@login_required
def all_jobs(request):
    jobs = AudioJob.objects.filter(user=request.user)
    return render(request, 'web/all_jobs.html', {'jobs': jobs})

@login_required
def metrics_view(request):
    jobs = AudioJob.objects.filter(user=request.user)
    total_jobs = jobs.count()
    completed_jobs = jobs.filter(status='completed').count()
    failed_jobs = jobs.filter(status='failed').count()
    processing_jobs = jobs.exclude(status__in=['completed', 'failed', 'uploaded']).count()

    # Calculate average processing time for completed jobs
    completed_times = []
    for job in jobs.filter(status='completed'):
        if job.created_at and job.updated_at:
            completed_times.append((job.updated_at - job.created_at).total_seconds())
    avg_time = sum(completed_times) / len(completed_times) if completed_times else 0

    return render(request, 'web/metrics.html', {
        'total_jobs': total_jobs,
        'completed_jobs': completed_jobs,
        'failed_jobs': failed_jobs,
        'processing_jobs': processing_jobs,
        'avg_time': avg_time,
    })