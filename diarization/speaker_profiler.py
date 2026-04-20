"""
Speaker Profiling Module
Detects speaker gender using pitch analysis and assigns TTS voices.
"""
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Voice assignments for Coqui TTS
MALE_VOICES = [
    "tts_models/en/ljspeech/tacotron2-DDC",  # fallback
    "tts_models/en/vctk/vits",
]

FEMALE_VOICES = [
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/vctk/vits",
]

# VCTK speaker IDs (if using VCTK model)
VCTK_MALE_SPEAKERS = ["p226", "p227", "p232", "p237", "p241", "p243"]
VCTK_FEMALE_SPEAKERS = ["p225", "p228", "p229", "p230", "p231", "p233"]


def estimate_pitch(wav_path: str, sr: int = 16000) -> float:
    """
    Estimate fundamental frequency (F0) using autocorrelation.
    More accurate pitch detection for gender classification.
    """
    try:
        import wave
        with wave.open(wav_path, 'r') as wf:
            file_sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        if len(samples) < file_sr * 0.1:
            return 0.0

        # Use more audio for better accuracy (up to 10 seconds)
        max_samples = int(file_sr * 10)
        if len(samples) > max_samples:
            # Take middle portion — more representative than start
            start = (len(samples) - max_samples) // 2
            samples = samples[start:start + max_samples]

        # Remove silence before pitch detection
        energy_threshold = 0.02
        frame_len = int(file_sr * 0.02)
        voiced_frames = []
        for i in range(0, len(samples) - frame_len, frame_len):
            frame = samples[i:i + frame_len]
            if np.sqrt(np.mean(frame ** 2)) > energy_threshold:
                voiced_frames.append(frame)

        if len(voiced_frames) < 5:
            return 0.0

        samples = np.concatenate(voiced_frames)

        # Autocorrelation pitch detection
        frame_length = int(file_sr * 0.04)   # 40ms frames
        hop_length   = int(file_sr * 0.01)   # 10ms hop
        min_period   = int(file_sr / 350)    # max 350 Hz
        max_period   = int(file_sr / 50)     # min 50 Hz

        pitches = []
        for i in range(0, len(samples) - frame_length, hop_length):
            frame = samples[i:i + frame_length]

            # Skip silent frames
            if np.sqrt(np.mean(frame ** 2)) < energy_threshold:
                continue

            # Normalize frame
            frame = frame - np.mean(frame)

            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            if len(autocorr) <= max_period:
                continue

            # Normalize by zero-lag
            if autocorr[0] == 0:
                continue
            autocorr = autocorr / autocorr[0]

            # Find peak in voiced range
            segment = autocorr[min_period:max_period]
            if len(segment) == 0:
                continue

            peak_idx = np.argmax(segment) + min_period
            peak_val = autocorr[peak_idx]

            # Only accept strong peaks (voiced speech)
            if peak_val > 0.4:
                pitch = file_sr / peak_idx
                if 50 <= pitch <= 350:
                    pitches.append(pitch)

        if len(pitches) < 3:
            return 0.0

        # Use median to ignore outliers
        return float(np.median(pitches))

    except Exception as e:
        logger.warning(f"Pitch estimation failed: {e}")
        return 0.0


def detect_gender_from_pitch(avg_pitch: float) -> str:
    """
    Classify gender based on median fundamental frequency.
    
    Typical ranges:
      Male voice:   85  - 180 Hz
      Female voice: 185 - 300 Hz
    """
    if avg_pitch <= 0:
        return 'unknown'
    elif avg_pitch < 185:
        return 'male'
    else:
        return 'female'


def profile_speaker_from_segments(speaker_id: str, segments: list,
                                   wav_path: str, output_dir: str) -> dict:
    """
    Profile a single speaker using their audio segments.
    Returns gender, pitch stats, and assigned TTS voice.
    """
    speaker_segments = [s for s in segments if s['speaker'] == speaker_id]

    if not speaker_segments:
        return {
            'speaker': speaker_id,
            'gender': 'unknown',
            'avg_pitch': 0.0,
            'total_duration': 0.0,
            'voice_model': MALE_VOICES[0],
            'voice_speaker': None,
        }

    pitches = []
    total_duration = 0.0

    for seg in speaker_segments:
        duration = seg['end'] - seg['start']
        total_duration += duration

        # Skip very short segments for pitch detection
        if duration < 1.0:
            continue

        seg_audio = seg.get('seg_audio_path')
        if seg_audio and os.path.exists(seg_audio):
            pitch = estimate_pitch(seg_audio)
            logger.info(f"  Segment pitch: {pitch:.1f} Hz (duration={duration:.1f}s)")
            if pitch > 0:
                pitches.append(pitch)

    # Log all collected pitches for debugging
    if pitches:
        logger.info(f"Speaker {speaker_id} pitches: {[round(p,1) for p in pitches]}")
        logger.info(f"Speaker {speaker_id} median pitch: {np.median(pitches):.1f} Hz")

    avg_pitch = float(np.median(pitches)) if pitches else 0.0
    gender = detect_gender_from_pitch(avg_pitch)

    # Assign TTS voice
    voice_model, voice_speaker = assign_voice(gender, speaker_id)

    logger.info(f"Speaker {speaker_id}: gender={gender}, pitch={avg_pitch:.1f}Hz, "
                f"duration={total_duration:.1f}s")

    return {
        'speaker': speaker_id,
        'gender': gender,
        'avg_pitch': avg_pitch,
        'total_duration': total_duration,
        'voice_model': voice_model,
        'voice_speaker': voice_speaker,
        'label': speaker_id.replace('SPEAKER_', 'Speaker ').replace('_0', ' ').strip(),
        'voice': voice_speaker or 'default',
    }


def assign_voice(gender: str, speaker_id: str) -> tuple:
    """
    Assign a TTS voice model and speaker ID based on gender.
    Returns (model_name, speaker_id_or_none).
    """
    speaker_idx = 0
    try:
        num = int(speaker_id.split('_')[-1])
        speaker_idx = num % 3
    except Exception:
        pass

    if gender == 'female':
        speakers = VCTK_FEMALE_SPEAKERS
        voice_model = "tts_models/en/vctk/vits"
    else:
        speakers = VCTK_MALE_SPEAKERS
        voice_model = "tts_models/en/vctk/vits"

    voice_speaker = speakers[speaker_idx % len(speakers)]
    return voice_model, voice_speaker


def profile_all_speakers(segments: list, wav_path: str,
                          output_dir: str, job_id: str) -> dict:
    """
    Profile all unique speakers in the session.
    Returns dict keyed by speaker_id.
    """
    unique_speakers = sorted(set(s['speaker'] for s in segments))
    profiles = {}

    for speaker_id in unique_speakers:
        logger.info(f"[{job_id}] Profiling speaker: {speaker_id}")
        profile = profile_speaker_from_segments(
            speaker_id, segments, wav_path, output_dir
        )
        profiles[speaker_id] = profile

    return profiles


def enrich_segments_with_profiles(segments: list, profiles: dict) -> list:
    """Add speaker profile info to each segment."""
    for i, seg in enumerate(segments):
        speaker_id = seg['speaker']
        profile = profiles.get(speaker_id, {})
        seg['gender'] = profile.get('gender', 'unknown')
        seg['voice_model'] = profile.get('voice_model', MALE_VOICES[0])
        seg['voice_speaker'] = profile.get('voice_speaker')
        seg['speaker_label'] = profile.get('label', speaker_id)

        # Assign numeric index for UI coloring
        try:
            seg['speaker_idx'] = int(speaker_id.split('_')[-1])
        except Exception:
            seg['speaker_idx'] = i % 4

    return segments
