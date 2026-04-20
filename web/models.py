from django.db import models
from django.contrib.auth.models import User
import uuid


class AudioJob(models.Model):
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('preprocessing', 'Preprocessing'),
        ('diarizing', 'Diarizing'),
        ('transcribing', 'Transcribing'),
        ('translating', 'Translating'),
        ('synthesizing', 'Synthesizing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='audio_jobs')
    original_file = models.FileField(upload_to='uploads/')
    original_filename = models.CharField(max_length=255)
    processed_wav = models.FileField(upload_to='processed/', null=True, blank=True)
    dubbed_audio = models.FileField(upload_to='dubbed/', null=True, blank=True)

    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default='uploaded')
    progress = models.IntegerField(default=0)
    status_message = models.CharField(max_length=500, blank=True)

    num_speakers = models.IntegerField(null=True, blank=True)
    max_speakers = models.IntegerField(default=4)

    transcript = models.JSONField(null=True, blank=True)
    speaker_profiles = models.JSONField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.original_filename} [{self.status}]"

    def get_status_display_class(self):
        mapping = {
            'uploaded': 'info',
            'preprocessing': 'warning',
            'diarizing': 'warning',
            'transcribing': 'warning',
            'translating': 'warning',
            'synthesizing': 'warning',
            'completed': 'success',
            'failed': 'danger',
        }
        return mapping.get(self.status, 'secondary')
