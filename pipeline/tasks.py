"""
Celery tasks for background audio processing.
"""
import logging
from speechdub.celery import app

logger = logging.getLogger(__name__)


@app.task(bind=True, name='pipeline.tasks.process_audio_job',
          max_retries=2, soft_time_limit=1800, time_limit=2000)
def process_audio_job(self, job_id: str):
    """
    Celery task: Run the full audio processing pipeline.
    """
    logger.info(f"Celery task started for job: {job_id}")
    try:
        from pipeline.orchestrator import run_pipeline
        run_pipeline(job_id)
        logger.info(f"Celery task completed for job: {job_id}")
    except Exception as e:
        logger.error(f"Celery task failed for job {job_id}: {e}")
        try:
            self.retry(exc=e, countdown=10)
        except self.MaxRetriesExceededError:
            from web.models import AudioJob
            AudioJob.objects.filter(id=job_id).update(
                status='failed',
                error_message=str(e),
                status_message='Processing failed after retries.',
            )
