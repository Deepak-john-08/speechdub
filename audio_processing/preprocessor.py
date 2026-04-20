"""
Audio Preprocessing Module
Converts uploaded audio to WAV 16kHz mono, applies noise reduction, VAD, normalization.
"""
import os
import logging
import subprocess
import tempfile
import numpy as np

logger = logging.getLogger(__name__)


def convert_to_wav(input_path: str, output_path: str) -> str:
    """Convert any audio format to WAV 16kHz mono using ffmpeg."""
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ar', '16000',   # 16kHz sample rate
        '-ac', '1',        # mono
        '-acodec', 'pcm_s16le',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
    logger.info(f"Converted {input_path} -> {output_path}")
    return output_path


def get_audio_duration(wav_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        import wave
        with wave.open(wav_path, 'r') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0


def load_wav(wav_path: str):
    """Load WAV file as numpy array, returns (samples, sample_rate)."""
    try:
        import wave, array as arr
        with wave.open(wav_path, 'r') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sr
    except Exception as e:
        logger.error(f"Could not load wav with wave module: {e}")
        return np.zeros(16000, dtype=np.float32), 16000


def save_wav(wav_path: str, samples: np.ndarray, sr: int = 16000):
    """Save numpy float32 array as 16-bit WAV."""
    import wave
    int_samples = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(wav_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())


def spectral_noise_reduction(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simple spectral gating noise reduction.
    Estimates noise from first 0.5s of audio.
    """
    try:
        noise_len = min(int(sr * 0.5), len(samples))
        if noise_len < 100:
            return samples

        # Use FFT-based spectral gating
        frame_size = 512
        hop_size = 256
        noise_profile = samples[:noise_len]

        # Compute noise power spectrum
        noise_frames = []
        for i in range(0, len(noise_profile) - frame_size, hop_size):
            frame = noise_profile[i:i + frame_size] * np.hanning(frame_size)
            noise_frames.append(np.abs(np.fft.rfft(frame)))

        if not noise_frames:
            return samples

        noise_mag = np.mean(noise_frames, axis=0)

        # Apply spectral gating to full audio
        output = np.zeros_like(samples)
        for i in range(0, len(samples) - frame_size, hop_size):
            frame = samples[i:i + frame_size] * np.hanning(frame_size)
            spectrum = np.fft.rfft(frame)
            mag = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Gate: suppress frequencies below 2x noise floor
            gate = np.maximum(mag - 2.0 * noise_mag, 0)
            cleaned = gate * np.exp(1j * phase)
            reconstructed = np.fft.irfft(cleaned)
            output[i:i + frame_size] += reconstructed * np.hanning(frame_size)

        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.9
        return output

    except Exception as e:
        logger.warning(f"Noise reduction failed, returning original: {e}")
        return samples


def apply_vad_silence_removal(samples: np.ndarray, sr: int = 16000,
                               frame_ms: int = 30, aggressiveness: int = 2) -> np.ndarray:
    """
    Voice Activity Detection using webrtcvad if available,
    otherwise falls back to energy-based VAD.
    """
    try:
        import webrtcvad
        vad = webrtcvad.Vad(aggressiveness)
        frame_length = int(sr * frame_ms / 1000)
        int_samples = (samples * 32767).clip(-32768, 32767).astype(np.int16)

        voiced_frames = []
        for i in range(0, len(int_samples) - frame_length, frame_length):
            frame = int_samples[i:i + frame_length]
            if len(frame) < frame_length:
                break
            try:
                is_speech = vad.is_speech(frame.tobytes(), sr)
                if is_speech:
                    voiced_frames.append(samples[i:i + frame_length])
            except Exception:
                voiced_frames.append(samples[i:i + frame_length])

        if voiced_frames:
            return np.concatenate(voiced_frames)
        return samples

    except ImportError:
        logger.info("webrtcvad not installed, using energy-based VAD")
        return _energy_vad(samples, sr)
    except Exception as e:
        logger.warning(f"VAD failed: {e}, returning original")
        return samples


def _energy_vad(samples: np.ndarray, sr: int = 16000,
                frame_ms: int = 30, threshold: float = 0.01) -> np.ndarray:
    """Simple energy-based voice activity detection."""
    frame_length = int(sr * frame_ms / 1000)
    voiced_frames = []
    for i in range(0, len(samples) - frame_length, frame_length):
        frame = samples[i:i + frame_length]
        energy = np.sqrt(np.mean(frame ** 2))
        if energy > threshold:
            voiced_frames.append(frame)
    if voiced_frames:
        return np.concatenate(voiced_frames)
    return samples


def normalize_audio(samples: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target RMS dB level."""
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < 1e-8:
        return samples
    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / rms
    normalized = samples * gain
    # Peak limiting
    peak = np.max(np.abs(normalized))
    if peak > 0.95:
        normalized = normalized * (0.95 / peak)
    return normalized


def preprocess_audio(input_path: str, output_dir: str, job_id: str) -> dict:
    """
    Full preprocessing pipeline:
    1. Convert to WAV 16kHz mono
    2. Noise reduction
    3. VAD / silence removal
    4. Normalization
    Returns dict with paths and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_wav = os.path.join(output_dir, f"{job_id}_raw.wav")
    processed_wav = os.path.join(output_dir, f"{job_id}_processed.wav")

    logger.info(f"[{job_id}] Starting preprocessing: {input_path}")

    # Step 1: Convert
    convert_to_wav(input_path, raw_wav)
    duration = get_audio_duration(raw_wav)
    logger.info(f"[{job_id}] Duration: {duration:.1f}s")

    # Step 2: Load
    samples, sr = load_wav(raw_wav)
    logger.info(f"[{job_id}] Loaded {len(samples)} samples at {sr}Hz")

    # Step 3: Noise reduction
    samples = spectral_noise_reduction(samples, sr)
    logger.info(f"[{job_id}] Noise reduction done")

    # Step 4: Normalize
    samples = normalize_audio(samples)
    logger.info(f"[{job_id}] Normalization done")

    # Step 5: Save processed file (keep full audio for diarization)
    save_wav(processed_wav, samples, sr)
    logger.info(f"[{job_id}] Saved processed WAV: {processed_wav}")

    # Clean up raw
    try:
        os.remove(raw_wav)
    except Exception:
        pass

    return {
        'processed_wav': processed_wav,
        'duration': duration,
        'sample_rate': sr,
        'num_samples': len(samples),
    }
