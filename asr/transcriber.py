"""
ASR Module - Speech to Text using OpenAI Whisper (local)
Handles Tamil, English, and code-mixed speech.
"""
import os
import logging
import tempfile
import subprocess
import numpy as np

logger = logging.getLogger(__name__)

_whisper_model = None
WHISPER_MODEL_SIZE = os.environ.get('WHISPER_MODEL_SIZE', 'medium')


def get_whisper_model():
    """Load Whisper model once and cache it."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
            _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            logger.info("Whisper model loaded successfully")
        except ImportError:
            logger.error("openai-whisper not installed. Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise
    return _whisper_model


def detect_language(audio_path: str) -> str:
    """Detect language of audio using Whisper."""
    try:
        model = get_whisper_model()
        import whisper
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        detected = max(probs, key=probs.get)
        logger.info(f"Detected language: {detected} (confidence: {probs[detected]:.2f})")
        return detected
    except Exception as e:
        logger.warning(f"Language detection failed: {e}. Defaulting to Tamil.")
        return 'ta'


def transcribe_segment(audio_path: str, language_hint: str = None,
                        initial_prompt: str = None) -> dict:
    """
    Transcribe a single audio segment using Whisper.
    Returns dict with text, language, confidence.
    """
    try:
        model = get_whisper_model()

        # Build transcription options
        options = {
            'task': 'transcribe',
            'verbose': False,
            'word_timestamps': False,
        }

        if language_hint:
            options['language'] = language_hint

        if initial_prompt:
            options['initial_prompt'] = initial_prompt
        else:
            # Helpful prompt for Tamil/English code-mixed speech
            options['initial_prompt'] = (
                "This audio may contain Tamil language, English language, "
                "or a mix of Tamil and English (Tanglish). "
                "Please transcribe accurately including colloquial Tamil words."
            )

        result = model.transcribe(audio_path, **options)

        text = result.get('text', '').strip()
        language = result.get('language', 'unknown')

        # Calculate average confidence from segments
        segments = result.get('segments', [])
        if segments:
            avg_conf = np.mean([s.get('avg_logprob', 0) for s in segments])
            # Convert log prob to 0-1 scale (roughly)
            confidence = float(np.clip(np.exp(avg_conf), 0, 1))
        else:
            confidence = 0.5

        logger.info(f"Transcribed: '{text[:60].encode('ascii', errors='replace').decode()}...' lang={language} conf={confidence:.2f}")
        return {
            'text': text,
            'language': language,
            'confidence': confidence,
            'segments': segments,
        }

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {
            'text': '',
            'language': 'unknown',
            'confidence': 0.0,
            'segments': [],
            'error': str(e),
        }


def transcribe_all_segments(wav_path: str, diarization_segments: list,
                             output_dir: str, job_id: str) -> list:
    """
    Transcribe each speaker segment.
    Returns enriched segments with transcription.
    """
    import subprocess
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # First pass: detect overall language
    overall_lang = detect_language(wav_path)
    logger.info(f"[{job_id}] Overall audio language: {overall_lang}")

    for i, seg in enumerate(diarization_segments):
        start = seg['start']
        end = seg['end']
        speaker = seg['speaker']
        duration = end - start

        logger.info(f"[{job_id}] Transcribing segment {i+1}/{len(diarization_segments)}: "
                    f"{speaker} [{start:.1f}s - {end:.1f}s]")

        # Extract segment audio
        seg_path = os.path.join(output_dir, f"{job_id}_seg_{i:04d}.wav")
        try:
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start),
                '-t', str(max(duration, 0.5)),
                '-i', wav_path,
                '-ar', '16000', '-ac', '1',
                seg_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        except Exception as e:
            logger.error(f"Segment extraction failed for segment {i}: {e}")
            results.append({**seg, 'text': '', 'language': 'unknown', 'confidence': 0.0})
            continue

        # Skip very short segments
        if duration < 0.5:
            results.append({**seg, 'text': '', 'language': 'unknown', 'confidence': 0.0})
            try:
                os.remove(seg_path)
            except Exception:
                pass
            continue

        # Transcribe
        transcription = transcribe_segment(seg_path, language_hint=overall_lang)

        result_seg = {
            **seg,
            'text': transcription['text'],
            'language': transcription['language'],
            'confidence': transcription['confidence'],
            'seg_audio_path': seg_path,
            'index': i,
        }
        results.append(result_seg)

    logger.info(f"[{job_id}] Transcription complete: {len(results)} segments")
    return results
