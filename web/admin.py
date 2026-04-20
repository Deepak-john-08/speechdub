from django.contrib import admin
from .models import AudioJob


@admin.register(AudioJob)
class AudioJobAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'user', 'status', 'num_speakers', 'progress', 'created_at']
    list_filter = ['status']
    search_fields = ['original_filename', 'user__username']
    readonly_fields = ['id', 'created_at', 'updated_at', 'completed_at']
