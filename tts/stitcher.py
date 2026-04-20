"""
Audio Stitching Module
Combines TTS-generated audio segments into a single dubbed audio file.
"""
import os
import logging
import wave
import numpy as np
import subprocess

logger = logging.getLogger(__name__)


def load_wav_samples(wav_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load WAV file, resample to target_sr if needed."""
    try:
        # Try to resample with ffmpeg first for reliability
        tmp_path = wav_path + '_resampled.wav'
        cmd = [
            'ffmpeg', '-y', '-i', wav_path,
            '-ar', str(target_sr), '-ac', '1',
            tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            wav_path = tmp_path

        with wave.open(wav_path, 'r') as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        # Clean up temp
        try:
            if '_resampled.wav' in wav_path:
                os.remove(wav_path)
        except Exception:
            pass

        return samples
    except Exception as e:
        logger.error(f"Failed to load WAV {wav_path}: {e}")
        return np.zeros(target_sr, dtype=np.float32)


def create_silence(duration_sec: float, sr: int = 16000) -> np.ndarray:
    """Generate silence array."""
    return np.zeros(int(sr * duration_sec), dtype=np.float32)


def save_combined_wav(samples: np.ndarray, output_path: str, sr: int = 16000):
    """Save float32 numpy array as 16-bit WAV."""
    # Normalize
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val * 0.9

    int_samples = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(output_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())
    logger.info(f"Saved combined audio: {output_path} ({len(samples)/sr:.1f}s)")


def stitch_audio_segments(segments: list, output_path: str,
                            job_id: str, sr: int = 16000,
                            gap_silence: float = 0.3) -> str:
    """
    Stitch all TTS segments in order, with silence gaps between.
    Returns path to stitched audio.
    """
    logger.info(f"[{job_id}] Stitching {len(segments)} audio segments...")

    audio_parts = []
    silence_gap = create_silence(gap_silence, sr)

    valid_segments = 0
    for i, seg in enumerate(segments):
        tts_path = seg.get('tts_audio_path')

        if tts_path and os.path.exists(tts_path):
            try:
                seg_samples = load_wav_samples(tts_path, target_sr=sr)
                if len(seg_samples) > 0:
                    audio_parts.append(seg_samples)
                    audio_parts.append(silence_gap.copy())
                    valid_segments += 1
            except Exception as e:
                logger.warning(f"Failed to load TTS segment {i}: {e}")
                audio_parts.append(create_silence(0.5, sr))
        else:
            # Generate silence placeholder based on text length
            text = seg.get('english_text', '')
            est_duration = max(len(text.split()) * 0.4, 0.5)
            audio_parts.append(create_silence(est_duration, sr))
            audio_parts.append(silence_gap.copy())

    if not audio_parts:
        logger.warning("No audio parts to stitch. Generating 3s silence.")
        audio_parts = [create_silence(3.0, sr)]

    combined = np.concatenate(audio_parts)
    save_combined_wav(combined, output_path, sr)

    logger.info(f"[{job_id}] Stitching done: {valid_segments}/{len(segments)} valid segments, "
                f"total duration: {len(combined)/sr:.1f}s")
    return output_path


def convert_wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = '128k') -> str:
    """Optionally convert final WAV to MP3."""
    try:
        cmd = ['ffmpeg', '-y', '-i', wav_path, '-ab', bitrate, mp3_path]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            return mp3_path
    except Exception as e:
        logger.warning(f"MP3 conversion failed: {e}")
    return wav_path
