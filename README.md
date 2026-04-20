# SpeechDub — Multilingual Multi-Speaker AI Dubbing System

A fully local Django web application that processes multilingual (Tamil + English)
multi-speaker audio and outputs English transcripts + dubbed speech.

---

## Features

- **Register / Login** user accounts
- **Dashboard** with upload form and job history
- **Automatic speaker detection** (2–4 speakers)
- **Speaker diarization** using pyannote.audio
- **ASR** using OpenAI Whisper (medium model)
- **Colloquial Tamil normalization** (Tanglish → formal Tamil)
- **Translation** Tamil → English via IndicTrans2 / Whisper
- **Gender detection** via pitch analysis
- **TTS dubbing** with Coqui TTS (gender-matched voices)
- **Audio stitching** into final dubbed WAV
- **Live progress tracking** via AJAX polling
- **Download** dubbed audio

---

## System Requirements

| Tool       | Requirement         |
|------------|---------------------|
| Python     | 3.10 or 3.11        |
| ffmpeg     | System-installed    |
| Redis      | For Celery (optional) |
| RAM        | 8GB+ recommended    |
| GPU        | Optional (CUDA)     |

---

## Quick Setup

### 1. Install system dependencies

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y ffmpeg redis-server espeak python3-venv

# macOS
brew install ffmpeg redis espeak
```

### 2. Create virtual environment

```bash
cd speechdub
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows
```

### 3. Install Python packages

```bash
pip install --upgrade pip

# Core requirements (install progressively)
pip install django celery redis numpy

# Whisper (ASR)
pip install openai-whisper

# Coqui TTS (Text-to-Speech)
pip install TTS

# Speaker diarization
pip install pyannote.audio

# Translation (IndicTrans2)
pip install transformers sentencepiece torch

# Optional: better VAD
pip install webrtcvad

# Optional: better audio processing
pip install librosa soundfile scipy
```

> **Tip:** You can install just `django numpy` to run the app with fallback
> processing (no AI models). The pipeline degrades gracefully.

### 4. Apply database migrations

```bash
python manage.py migrate
```

### 5. Create a superuser (optional)

```bash
python manage.py createsuperuser
```

### 6. Create media folders

```bash
mkdir -p media/uploads media/processed media/dubbed media/jobs logs
```

### 7. Run the development server

```bash
python manage.py runserver
```

Open: **http://127.0.0.1:8000**

---

## Running with Celery (Background Processing)

Celery enables non-blocking audio processing. Open **two terminals**:

**Terminal 1 — Django server:**
```bash
source venv/bin/activate
python manage.py runserver
```

**Terminal 2 — Celery worker:**
```bash
source venv/bin/activate
celery -A speechdub worker --loglevel=info --concurrency=1
```

> If Redis is not running: `sudo service redis start` (Linux) or `brew services start redis` (Mac)
>
> **Without Celery**, the app falls back to synchronous processing (page will
> wait until done — works fine for testing).

---

## Model Downloads (First Run)

The following models are downloaded automatically on first use:

| Model | Size | Purpose |
|-------|------|---------|
| Whisper `medium` | ~1.5 GB | ASR (speech → text) |
| `pyannote/speaker-diarization-3.1` | ~200 MB | Speaker diarization |
| `ai4bharat/indictrans2-indic-en-1B` | ~2 GB | Tamil→English translation |
| Coqui `tts_models/en/ljspeech/tacotron2-DDC` | ~100 MB | Text-to-speech |

**Total ~4 GB** on first run. Subsequent runs use the cache.

To pre-download:
```bash
python -c "import whisper; whisper.load_model('medium')"
python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC')"
```

---

## Project Structure

```
speechdub/
├── manage.py
├── requirements.txt
├── README.md
│
├── speechdub/              # Django project config
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── celery.py
│
├── web/                    # Django app (UI + models)
│   ├── models.py           # AudioJob model
│   ├── views.py            # Register, login, dashboard, upload, status, result
│   ├── forms.py
│   ├── urls.py
│   ├── admin.py
│   └── templates/web/
│       ├── base.html
│       ├── login.html
│       ├── register.html
│       ├── dashboard.html
│       ├── status.html
│       ├── result.html
│       └── all_jobs.html
│
├── audio_processing/       # Preprocessing module
│   └── preprocessor.py     # ffmpeg convert, noise reduce, VAD, normalize
│
├── diarization/            # Speaker diarization
│   ├── diarizer.py         # pyannote + fallback
│   └── speaker_profiler.py # Gender + voice assignment
│
├── asr/                    # Speech recognition
│   └── transcriber.py      # Whisper medium
│
├── normalization/          # Tamil text normalization
│   └── tamil_normalizer.py # Colloquial → formal Tamil
│
├── translation/            # Language translation
│   └── translator.py       # IndicTrans2 + Whisper translate fallback
│
├── tts/                    # Text-to-speech
│   ├── synthesizer.py      # Coqui TTS + fallbacks
│   └── stitcher.py         # Audio stitching
│
├── pipeline/               # Orchestrator
│   ├── orchestrator.py     # Main pipeline controller
│   └── tasks.py            # Celery task definition
│
├── media/                  # Uploaded + processed files
│   ├── uploads/
│   ├── processed/
│   ├── dubbed/
│   └── jobs/
│
└── logs/
    └── app.log
```

---

## Usage Walkthrough

1. **Register** at `/register/`
2. **Login** at `/login/`
3. **Dashboard** — drag & drop or browse for an audio file
4. Choose max speakers (2 / 3 / 4)
5. Click **Process Audio**
6. Watch the **live progress bar** on the status page
7. When done → **View Results**:
   - See speaker profiles (gender, voice, duration)
   - Play or download the **dubbed English audio**
   - Read the **English transcript** with speaker labels and timestamps

---

## Accuracy Expectations

| Speakers | Expected Accuracy |
|----------|-------------------|
| 2        | ~85–90%           |
| 3        | ~75%              |
| 4        | ~70%              |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `medium` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `PYANNOTE_MODEL_PATH` | `~/.cache/pyannote/...` | Local path for pyannote |
| `INDICTRANS_MODEL_PATH` | `~/.cache/indictrans2` | Local path for IndicTrans2 |

Set via shell:
```bash
export WHISPER_MODEL_SIZE=small  # faster, less accurate
python manage.py runserver
```

---

## Troubleshooting

**ffmpeg not found:**
```bash
sudo apt install ffmpeg    # Ubuntu
brew install ffmpeg        # macOS
```

**Redis connection error:**
The app falls back to synchronous mode automatically if Redis is unavailable.

**Out of memory (OOM):**
Use `WHISPER_MODEL_SIZE=small` and disable pyannote:
```bash
export WHISPER_MODEL_SIZE=small
```

**Coqui TTS fails:**
The pipeline falls back to `pyttsx3` → `espeak` → silent audio automatically.

**pyannote requires HuggingFace token:**
If pyannote requires auth, the system uses energy-based fallback diarization.
To use pyannote with auth:
```bash
pip install huggingface_hub
huggingface-cli login
```

---

## Admin Panel

Access at `/admin/` after creating a superuser.
Manage all audio jobs, users, and statuses.

---

## License

MIT — free for personal and commercial use.
