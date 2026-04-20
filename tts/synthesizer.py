"""
Text-to-Speech Module using Coqui TTS (local)
Generates English speech with gender-matched voices per speaker.
Falls back to pyttsx3 or gTTS-offline if Coqui unavailable.
"""
import os
import logging
import subprocess
import numpy as np
import tempfile
import wave

logger = logging.getLogger(__name__)

_tts_models = {}   # cache: model_name → TTS instance


def get_coqui_tts(model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
    """Load and cache a Coqui TTS model."""
    global _tts_models
    if model_name in _tts_models:
        return _tts_models[model_name]

    try:
        from TTS.api import TTS
        logger.info(f"Loading Coqui TTS model: {model_name}")
        tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
        _tts_models[model_name] = tts
        logger.info(f"Coqui TTS loaded: {model_name}")
        return tts
    except ImportError:
        logger.warning("Coqui TTS not installed. Install with: pip install TTS")
        return None
    except Exception as e:
        logger.warning(f"Coqui TTS model {model_name} failed: {e}")
        return None


def synthesize_with_coqui(text: str, output_path: str,
                           model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
                           speaker: str = None) -> bool:
    """Synthesize speech using Coqui TTS."""
    try:
        tts = get_coqui_tts(model_name)
        if tts is None:
            return False

        kwargs = {'text': text, 'file_path': output_path}
        if speaker and hasattr(tts, 'speakers') and tts.speakers:
            if speaker in tts.speakers:
                kwargs['speaker'] = speaker
            else:
                kwargs['speaker'] = tts.speakers[0]

        tts.tts_to_file(**kwargs)
        logger.info(f"Coqui TTS: synthesized '{text[:40].encode('ascii', errors='replace').decode()}...' -> {output_path}")
        return True

    except Exception as e:
        logger.error(f"Coqui TTS synthesis failed: {e}")
        return False


def synthesize_with_pyttsx3(text: str, output_path: str, gender: str = 'male') -> bool:
    """Fallback TTS using pyttsx3."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')

        # Try to select gender-appropriate voice
        target_gender = 'female' if gender == 'female' else 'male'
        for voice in voices:
            voice_name = voice.name.lower()
            if target_gender == 'female' and any(n in voice_name for n in ['female', 'zira', 'hazel', 'victoria']):
                engine.setProperty('voice', voice.id)
                break
            elif target_gender == 'male' and any(n in voice_name for n in ['male', 'david', 'mark', 'daniel']):
                engine.setProperty('voice', voice.id)
                break

        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)

        # pyttsx3 save to file
        engine.save_to_file(text, output_path)
        engine.runAndWait()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            logger.info(f"pyttsx3 TTS: synthesized to {output_path}")
            return True
        return False

    except ImportError:
        logger.warning("pyttsx3 not installed")
        return False
    except Exception as e:
        logger.error(f"pyttsx3 failed: {e}")
        return False


def synthesize_with_espeak(text: str, output_path: str, gender: str = 'male') -> bool:
    """Fallback TTS using espeak (system command)."""
    try:
        voice = 'en+m1' if gender != 'female' else 'en+f1'
        # Use aiff/wav via espeak
        wav_tmp = output_path.replace('.wav', '_esp.wav')
        cmd = ['espeak', '-v', voice, '-s', '140', '-w', output_path, text]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and os.path.exists(output_path):
            logger.info(f"espeak TTS: synthesized to {output_path}")
            return True
        return False
    except FileNotFoundError:
        logger.warning("espeak not found")
        return False
    except Exception as e:
        logger.error(f"espeak failed: {e}")
        return False


def generate_silent_wav(output_path: str, duration: float = 1.0, sr: int = 16000):
    """Generate a silent WAV file as last resort placeholder."""
    n_samples = int(sr * duration)
    samples = np.zeros(n_samples, dtype=np.int16)
    with wave.open(output_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def synthesize_segment(text: str, output_path: str,
                        voice_model: str = None,
                        voice_speaker: str = None,
                        gender: str = 'male') -> bool:
    """
    Synthesize a single text segment to audio.
    Tries Coqui → pyttsx3 → espeak → silent placeholder.
    """
    if not text or not text.strip():
        generate_silent_wav(output_path, duration=0.5)
        return True

    # Clean text for TTS
    text = text.strip()
    if len(text) > 500:
        text = text[:500] + '.'

    # Try Coqui TTS
    model = voice_model or "tts_models/en/ljspeech/tacotron2-DDC"
    if synthesize_with_coqui(text, output_path, model, voice_speaker):
        return True

    # Try fallback models
    simple_model = "tts_models/en/ljspeech/tacotron2-DDC"
    if model != simple_model:
        if synthesize_with_coqui(text, output_path, simple_model):
            return True

    # Try pyttsx3
    if synthesize_with_pyttsx3(text, output_path, gender):
        return True

    # Try espeak
    if synthesize_with_espeak(text, output_path, gender):
        return True

    # Last resort: silent audio
    logger.warning(f"All TTS methods failed for: '{text[:40]}'. Generating silence.")
    generate_silent_wav(output_path, duration=max(len(text) / 15, 1.0))
    return False


def synthesize_all_segments(segments: list, output_dir: str, job_id: str) -> list:
    """
    Generate speech for all transcript segments.
    Returns segments enriched with tts_audio_path.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        english_text = seg.get('english_text', '') or seg.get('text', '')
        if not english_text or not english_text.strip():
            seg['tts_audio_path'] = None
            continue

        tts_path = os.path.join(output_dir, f"{job_id}_tts_{i:04d}.wav")

        logger.info(f"[{job_id}] Synthesizing segment {i+1}/{len(segments)}: "
            f"{seg['speaker']} - '{seg['text'][:40].encode('ascii', errors='replace').decode()}...'")

        success = synthesize_segment(
            text=english_text,
            output_path=tts_path,
            voice_model=seg.get('voice_model'),
            voice_speaker=seg.get('voice_speaker'),
            gender=seg.get('gender', 'male'),
        )

        seg['tts_audio_path'] = tts_path if (success and os.path.exists(tts_path)) else None

    return segments
