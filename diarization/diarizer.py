"""
Speaker Diarization Module
Uses pyannote.audio locally for speaker diarization.
Falls back to energy-based segmentation if pyannote is unavailable.
"""
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

PYANNOTE_MODEL_PATH = os.environ.get(
    'PYANNOTE_MODEL_PATH',
    os.path.expanduser('~/.cache/pyannote/speaker-diarization')
)

# HuggingFace token for accessing gated models
HUGGINGFACE_TOKEN = os.environ.get(
    'HUGGINGFACE_TOKEN',
    "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"  # Update with your token
)

def diarize_with_pyannote(wav_path: str, max_speakers: int = 4,
                           min_speakers: int = 1) -> list:
    """
    Run pyannote.audio speaker diarization.
    Returns list of segments: [{'start': float, 'end': float, 'speaker': str}, ...]
    """
    try:
        from pyannote.audio import Pipeline
        import torch
        import wave

        logger.info("Loading pyannote pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HUGGINGFACE_TOKEN,
            cache_dir=os.path.expanduser("~/.cache/pyannote")
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = pipeline.to(device)

        # Load audio as tensor to avoid torchcodec issue on Windows
        with wave.open(wav_path, 'r') as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        waveform = torch.tensor(samples).unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": sr}

        logger.info(f"Running diarization on {wav_path}, max_speakers={max_speakers}")
        diarization = pipeline(
            audio_input,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        segments = []

        # pyannote 4.x returns DiarizeOutput wrapper — extract .speaker_diarization
        # pyannote 3.x returns Annotation directly — getattr handles both
        annotation = getattr(diarization, 'speaker_diarization', diarization)

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
            })

        logger.info(f"Diarization done: {len(segments)} segments")
        return segments if segments else None

    except ImportError:
        logger.warning("pyannote.audio not installed. Using fallback diarization.")
        return None
    except Exception as e:
        logger.warning(f"Pyannote diarization failed: {e}. Using fallback.")
        return None


def energy_based_diarization(wav_path: str, max_speakers: int = 4) -> list:
    """
    Fallback: energy-based speaker segmentation.
    Groups chunks by energy level into pseudo-speaker clusters.
    """
    import wave

    logger.info("Using energy-based fallback diarization...")

    with wave.open(wav_path, 'r') as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    frame_ms = 500  # 500ms frames
    frame_len = int(sr * frame_ms / 1000)
    energies = []

    for i in range(0, len(samples) - frame_len, frame_len):
        frame = samples[i:i + frame_len]
        energy = float(np.sqrt(np.mean(frame ** 2)))
        energies.append(energy)

    if not energies:
        return [{'start': 0.0, 'end': 5.0, 'speaker': 'SPEAKER_00'}]

    energy_arr = np.array(energies)
    mean_e = np.mean(energy_arr)
    threshold = mean_e * 0.5

    n_speakers = max_speakers  # respect caller's intent
    segments = []
    current_speaker = 0
    silent_count = 0

    for idx, energy in enumerate(energy_arr):
        start = idx * frame_ms / 1000.0
        end = start + frame_ms / 1000.0

        if energy < threshold:
            silent_count += 1
            if silent_count >= 2:
                current_speaker = (current_speaker + 1) % n_speakers
                silent_count = 0
        else:
            silent_count = 0

        speaker_label = f"SPEAKER_{current_speaker:02d}"
        if segments and segments[-1]['speaker'] == speaker_label:
            segments[-1]['end'] = end
        else:
            segments.append({'start': start, 'end': end, 'speaker': speaker_label})

    return segments


def merge_short_segments(segments: list, min_duration: float = 1.0) -> list:
    """Remove segments shorter than min_duration seconds."""
    return [s for s in segments if (s['end'] - s['start']) >= min_duration]


def merge_consecutive_same_speaker(segments: list, gap_tolerance: float = 0.5) -> list:
    """Merge consecutive segments from the same speaker if gap < tolerance."""
    if not segments:
        return segments

    merged = [dict(segments[0])]
    for seg in segments[1:]:
        last = merged[-1]
        gap = seg['start'] - last['end']
        if seg['speaker'] == last['speaker'] and gap <= gap_tolerance:
            last['end'] = seg['end']
        else:
            merged.append(dict(seg))
    return merged


def get_unique_speakers(segments: list) -> list:
    """Get sorted unique speaker labels."""
    speakers = sorted(set(s['speaker'] for s in segments))
    return speakers


def normalize_speaker_labels(segments: list) -> list:
    """Remap speaker labels to SPEAKER_00, SPEAKER_01, etc."""
    unique = get_unique_speakers(segments)
    remap = {old: f"SPEAKER_{i:02d}" for i, old in enumerate(unique)}
    for seg in segments:
        seg['speaker'] = remap[seg['speaker']]
    return segments


def diarize_audio(wav_path: str, max_speakers: int = 4, job_id: str = '') -> dict:
    """
    Main diarization entry point.
    Returns structured diarization result.
    """
    logger.info(f"[{job_id}] Starting diarization: {wav_path}")

    # Try pyannote first
    segments = diarize_with_pyannote(wav_path, max_speakers=max_speakers)

    # Fallback
    if segments is None:
        segments = energy_based_diarization(wav_path, max_speakers=max_speakers)

    # Post-process
    segments = merge_consecutive_same_speaker(segments)
    segments = merge_short_segments(segments, min_duration=0.8)
    segments = normalize_speaker_labels(segments)

    speakers = get_unique_speakers(segments)
    num_speakers = len(speakers)
    num_speakers = min(max(num_speakers, 1), max_speakers)

    logger.info(f"[{job_id}] Detected {num_speakers} speaker(s), {len(segments)} segments")

    return {
        'segments': segments,
        'num_speakers': num_speakers,
        'speakers': speakers,
    }


def extract_speaker_segment_audio(wav_path: str, start: float, end: float,
                                   output_path: str) -> str:
    """Extract a time segment from WAV file using ffmpeg."""
    import subprocess
    duration = end - start
    if duration <= 0:
        duration = 0.5

    FFMPEG_PATH = r'C:\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe'
    if not os.path.exists(FFMPEG_PATH):
        FFMPEG_PATH = 'ffmpeg'

    cmd = [
        FFMPEG_PATH, '-y',
        '-ss', str(start),
        '-t', str(duration),
        '-i', wav_path,
        '-ar', '16000', '-ac', '1',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Segment extraction failed: {result.stderr}")
    return output_path