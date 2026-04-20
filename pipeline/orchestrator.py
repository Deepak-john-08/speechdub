"""
Pipeline Orchestrator
Central controller that runs all modules in sequence.
"""
import os
import logging
import traceback
from pathlib import Path
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


def update_job(job_id: str, **kwargs):
    """Update job status in database."""
    try:
        import django
        if not django.conf.settings.configured:
            django.setup()
        from web.models import AudioJob
        AudioJob.objects.filter(id=job_id).update(**kwargs)
        logger.info(f"[{job_id}] Job updated: {kwargs}")
    except Exception as e:
        logger.error(f"Job update failed: {e}")


def get_job(job_id: str):
    """Retrieve job from database."""
    from web.models import AudioJob
    return AudioJob.objects.get(id=job_id)


def run_pipeline(job_id: str):
    """
    Full pipeline:
    1. Preprocessing
    2. Diarization
    3. ASR (Transcription)
    4. Normalization
    5. Translation
    6. Speaker Profiling
    7. TTS Synthesis
    8. Audio Stitching
    9. Save results
    """
    logger.info(f"[{job_id}] ========== PIPELINE START ==========")

    # Setup directories
    media_root = str(settings.MEDIA_ROOT)
    job_dir = os.path.join(media_root, 'jobs', str(job_id))
    os.makedirs(job_dir, exist_ok=True)

    try:
        job = get_job(job_id)
        input_path = job.original_file.path

        # ─────────────────────────────────────────────────────────
        # STAGE 1: Preprocessing
        # ─────────────────────────────────────────────────────────
        update_job(job_id,
                   status='preprocessing',
                   progress=5,
                   status_message='Preprocessing audio (convert, denoise, normalize)...')

        from audio_processing.preprocessor import preprocess_audio
        preprocess_result = preprocess_audio(
            input_path=input_path,
            output_dir=job_dir,
            job_id=str(job_id),
        )
        processed_wav = preprocess_result['processed_wav']
        duration = preprocess_result['duration']
        logger.info(f"[{job_id}] Preprocessing done. Duration={duration:.1f}s")

        # Save processed wav path to job
        relative_processed = os.path.relpath(processed_wav, media_root)
        update_job(job_id,
                   processed_wav=relative_processed,
                   progress=15,
                   status_message=f'Audio ready ({duration:.1f}s). Running speaker diarization...')

        # ─────────────────────────────────────────────────────────
        # STAGE 2: Diarization
        # ─────────────────────────────────────────────────────────
        update_job(job_id, status='diarizing', progress=20,
                   status_message='Detecting speakers (this may take 1-2 min)...')

        from diarization.diarizer import diarize_audio
        job = get_job(job_id)
        diarization_result = diarize_audio(
            wav_path=processed_wav,
            max_speakers=job.max_speakers,
            job_id=str(job_id),
        )

        segments = diarization_result['segments']
        num_speakers = diarization_result['num_speakers']

        update_job(job_id,
                   num_speakers=num_speakers,
                   progress=35,
                   status_message=f'Detected {num_speakers} speaker(s), {len(segments)} segments. Transcribing...')

        if not segments:
            raise ValueError("No speech segments detected in audio.")

        # ─────────────────────────────────────────────────────────
        # STAGE 3: Transcription (ASR)
        # ─────────────────────────────────────────────────────────
        update_job(job_id, status='transcribing', progress=40,
                   status_message='Transcribing speech with Whisper...')

        asr_dir = os.path.join(job_dir, 'asr_segments')
        from asr.transcriber import transcribe_all_segments
        segments = transcribe_all_segments(
            wav_path=processed_wav,
            diarization_segments=segments,
            output_dir=asr_dir,
            job_id=str(job_id),
        )

        update_job(job_id, progress=55,
                   status_message='Transcription done. Normalizing Tamil text...')

        # ─────────────────────────────────────────────────────────
        # STAGE 4: Tamil Normalization
        # ─────────────────────────────────────────────────────────
        from normalization.tamil_normalizer import process_segments_normalization
        segments = process_segments_normalization(segments)

        update_job(job_id, progress=60,
                   status_message='Normalization done. Translating to English...')

        # ─────────────────────────────────────────────────────────
        # STAGE 5: Translation
        # ─────────────────────────────────────────────────────────
        update_job(job_id, status='translating', progress=62,
                   status_message='Translating Tamil -> English...')

        from translation.translator import translate_all_segments
        segments = translate_all_segments(segments)

        update_job(job_id, progress=70,
                   status_message='Translation done. Profiling speakers...')

        # ─────────────────────────────────────────────────────────
        # STAGE 6: Speaker Profiling (Gender + Voice)
        # ─────────────────────────────────────────────────────────
        from diarization.speaker_profiler import (
            profile_all_speakers, enrich_segments_with_profiles
        )
        profiles = profile_all_speakers(
            segments=segments,
            wav_path=processed_wav,
            output_dir=job_dir,
            job_id=str(job_id),
        )
        segments = enrich_segments_with_profiles(segments, profiles)

        update_job(job_id, progress=75,
                   status_message='Profiles ready. Synthesizing English speech...')

        # ─────────────────────────────────────────────────────────
        # STAGE 7: TTS Synthesis
        # ─────────────────────────────────────────────────────────
        update_job(job_id, status='synthesizing', progress=78,
                   status_message='Generating dubbed English speech...')

        tts_dir = os.path.join(job_dir, 'tts_segments')
        from tts.synthesizer import synthesize_all_segments
        segments = synthesize_all_segments(
            segments=segments,
            output_dir=tts_dir,
            job_id=str(job_id),
        )

        update_job(job_id, progress=90,
                   status_message='Speech synthesis done. Stitching audio...')

        # ─────────────────────────────────────────────────────────
        # STAGE 8: Audio Stitching
        # ─────────────────────────────────────────────────────────
        dubbed_wav = os.path.join(job_dir, f'{job_id}_dubbed.wav')
        from tts.stitcher import stitch_audio_segments
        stitch_audio_segments(
            segments=segments,
            output_path=dubbed_wav,
            job_id=str(job_id),
        )

        # Save dubbed audio path
        relative_dubbed = os.path.relpath(dubbed_wav, media_root)

        update_job(job_id, progress=95,
                   status_message='Audio ready. Building transcript...')

        # ─────────────────────────────────────────────────────────
        # STAGE 9: Build Final Transcript
        # ─────────────────────────────────────────────────────────
        transcript = []
        for seg in segments:
            transcript.append({
                'speaker': seg.get('speaker', 'UNKNOWN'),
                'speaker_label': seg.get('speaker_label', seg.get('speaker', 'Speaker')),
                'speaker_idx': seg.get('speaker_idx', 0),
                'start': round(seg.get('start', 0), 2),
                'end': round(seg.get('end', 0), 2),
                'language': seg.get('language', 'unknown'),
                'original_text': seg.get('text', ''),
                'normalized_text': seg.get('normalized_text', ''),
                'english_text': seg.get('english_text', ''),
                'gender': seg.get('gender', 'unknown'),
                'confidence': round(seg.get('confidence', 0), 3),
                'translated': seg.get('translated', False),
                'is_tamil': seg.get('is_tamil', False),
            })

        # Build speaker_profiles for UI
        speaker_profiles_ui = {}
        for spk_id, profile in profiles.items():
            speaker_profiles_ui[spk_id] = {
                'label': profile.get('label', spk_id),
                'gender': profile.get('gender', 'unknown'),
                'voice': profile.get('voice_speaker') or 'default',
                'avg_pitch': round(profile.get('avg_pitch', 0), 1),
                'total_duration': round(profile.get('total_duration', 0), 1),
            }

        # ─────────────────────────────────────────────────────────
        # STAGE 10: Complete Job
        # ─────────────────────────────────────────────────────────
        from web.models import AudioJob
        AudioJob.objects.filter(id=job_id).update(
            status='completed',
            progress=100,
            status_message='Done! Your audio has been dubbed successfully.',
            dubbed_audio=relative_dubbed,
            transcript=transcript,
            speaker_profiles=speaker_profiles_ui,
            num_speakers=num_speakers,
            completed_at=timezone.now(),
            error_message='',
        )

        logger.info(f"[{job_id}] ========== PIPELINE COMPLETE ==========")
        logger.info(f"[{job_id}] Speakers: {num_speakers}, Segments: {len(transcript)}")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[{job_id}] PIPELINE FAILED: {error_msg}")
        update_job(job_id,
                   status='failed',
                   progress=0,
                   status_message='Processing failed.',
                   error_message=error_msg[:2000])
