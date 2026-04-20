"""
Translation Module - Tamil to English using IndicTrans2 (local)
Falls back to simple dictionary-based translation if model unavailable.
"""
import os
import logging
import re

logger = logging.getLogger(__name__)

_translator = None
INDICTRANS_MODEL_PATH = os.environ.get(
    'INDICTRANS_MODEL_PATH',
    os.path.expanduser('~/.cache/indictrans2')
)

# HuggingFace token for accessing gated models
HUGGINGFACE_TOKEN = os.environ.get(
    'HUGGINGFACE_TOKEN',
    "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"  # Update with your token
)


def get_indictrans_translator():
    """Load IndicTrans2 model once and cache."""
    global _translator
    if _translator is not None:
        return _translator

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        model_name = "ai4bharat/indictrans2-indic-en-1B"
        logger.info(f"Loading IndicTrans2 model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=HUGGINGFACE_TOKEN,
            cache_dir=INDICTRANS_MODEL_PATH
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=HUGGINGFACE_TOKEN,
            cache_dir=INDICTRANS_MODEL_PATH
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        _translator = {'model': model, 'tokenizer': tokenizer, 'device': device}
        logger.info("IndicTrans2 loaded successfully")
        return _translator

    except ImportError:
        logger.warning("transformers not installed for IndicTrans2")
        return None
    except Exception as e:
        logger.warning(f"IndicTrans2 load failed: {e}")
        return None


def translate_with_indictrans2(text: str) -> str:
    """Translate Tamil text to English using IndicTrans2."""
    translator = get_indictrans_translator()
    if translator is None:
        return None

    try:
        from transformers import pipeline as hf_pipeline

        tokenizer = translator['tokenizer']
        model = translator['model']
        device = translator['device']

        # IndicTrans2 format: add language tags
        src_text = f">>ta<< {text}"

        inputs = tokenizer(
            src_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=4,
                max_length=512,
                early_stopping=True,
            )

        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated.strip()

    except Exception as e:
        logger.error(f"IndicTrans2 translation error: {e}")
        return None


def translate_with_whisper_task(text: str, audio_path: str = None) -> str:
    """
    Use Whisper's translate task for Tamil → English.
    More reliable than IndicTrans2 for Whisper-transcribed content.
    """
    if not audio_path or not os.path.exists(audio_path):
        return None

    try:
        import whisper
        from asr.transcriber import get_whisper_model
        model = get_whisper_model()

        result = model.transcribe(
            audio_path,
            task='translate',
            language='ta',
            initial_prompt="Translate this Tamil speech to English."
        )
        translated = result.get('text', '').strip()
        if translated:
            logger.info(f"Whisper translated: '{text[:30]}' → '{translated[:50]}'")
        return translated

    except Exception as e:
        logger.warning(f"Whisper translate failed: {e}")
        return None


# Basic Tamil→English word dictionary as ultimate fallback
BASIC_TAMIL_EN_DICT = {
    "நான்": "I", "நீ": "you", "அவன்": "he", "அவள்": "she",
    "அவர்கள்": "they", "நாங்கள்": "we", "நாம்": "we",
    "என்ன": "what", "எங்கே": "where", "எப்படி": "how",
    "எப்போது": "when", "ஏன்": "why", "யார்": "who",
    "இருக்கிறேன்": "am", "இருக்கிறது": "is", "இருக்கிறார்கள்": "are",
    "செய்கிறேன்": "doing", "போகிறேன்": "going", "வருகிறேன்": "coming",
    "சொல்கிறேன்": "saying", "பார்க்கிறேன்": "seeing",
    "நல்ல": "good", "கெட்ட": "bad", "பெரிய": "big", "சின்ன": "small",
    "மிகவும்": "very", "கொஞ்சம்": "little", "நன்றி": "thank you",
    "சரி": "okay", "ஆம்": "yes", "இல்லை": "no",
    "வணக்கம்": "hello", "விடைபெறுகிறேன்": "goodbye",
    "தண்ணீர்": "water", "உணவு": "food", "வீடு": "house",
    "வேலை": "work", "நேரம்": "time", "பணம்": "money",
    "தாமதமாக": "late", "விரைவாக": "quickly", "நன்றாக": "well",
}


def fallback_translate(text: str) -> str:
    """Simple word-by-word fallback translation."""
    if not text:
        return text

    # If already mostly English, return as-is
    english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha > 0 and english_chars / total_alpha > 0.7:
        return text

    words = text.split()
    translated = []
    for word in words:
        clean = re.sub(r'[^\u0B80-\u0BFF\w]', '', word)
        if clean in BASIC_TAMIL_EN_DICT:
            translated.append(BASIC_TAMIL_EN_DICT[clean])
        elif any('\u0B80' <= c <= '\u0BFF' for c in word):
            translated.append(f"[{word}]")
        else:
            translated.append(word)
    return ' '.join(translated)


def translate_tamil_to_english(text: str, audio_path: str = None) -> str:
    """
    Main translation entry point.
    Tries IndicTrans2 → Whisper translate → fallback.
    """
    if not text or not text.strip():
        return text

    # Check if translation is needed
    english_ratio = sum(1 for c in text if c.isascii() and c.isalpha()) / max(len(text), 1)
    if english_ratio > 0.8:
        logger.info("Text already mostly English, skipping translation.")
        return text

    # Try IndicTrans2
    result = translate_with_indictrans2(text)
    if result and len(result) > 2:
        return result

    # Try Whisper translation on the segment audio
    if audio_path:
        result = translate_with_whisper_task(text, audio_path)
        if result and len(result) > 2:
            return result

    # Fallback
    logger.warning("Using fallback dictionary translation")
    return fallback_translate(text)


def translate_all_segments(segments: list) -> list:
    """Translate all Tamil segments to English."""
    for seg in segments:
        normalized_text = seg.get('normalized_text') or seg.get('text', '')
        language = seg.get('language', '')
        is_tamil = seg.get('is_tamil', False)

        if is_tamil or language in ('ta', 'tamil'):
            audio_path = seg.get('seg_audio_path')
            english_text = translate_tamil_to_english(normalized_text, audio_path)
            seg['english_text'] = english_text
            seg['translated'] = True
        else:
            # Already English
            seg['english_text'] = normalized_text
            seg['translated'] = False

    return segments
