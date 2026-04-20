"""
Colloquial Tamil Normalization Module
Converts spoken/colloquial Tamil and Tanglish to formal Tamil text.
"""
import re
import logging

logger = logging.getLogger(__name__)

# ============================================================
# RULE-BASED DICTIONARY: Colloquial Tamil → Formal Tamil
# ============================================================
COLLOQUIAL_DICT = {
    # Pronouns
    "naan": "நான்",
    "nee": "நீ",
    "avan": "அவன்",
    "aval": "அவள்",
    "avanga": "அவர்கள்",
    "naanga": "நாங்கள்",
    "neengalum": "நீங்களும்",
    "neengal": "நீங்கள்",
    "namma": "நாம்",
    "antha": "அந்த",
    "indha": "இந்த",
    "avar": "அவர்",

    # Common verbs colloquial → formal
    "solluven": "சொல்கிறேன்",
    "solren": "சொல்கிறேன்",
    "sollu": "சொல்",
    "paaru": "பார்",
    "paarunga": "பாருங்கள்",
    "poru": "போ",
    "va": "வா",
    "vaa": "வா",
    "varuven": "வருவேன்",
    "varen": "வருகிறேன்",
    "iruken": "இருக்கிறேன்",
    "irukku": "இருக்கிறது",
    "irukken": "இருக்கிறேன்",
    "irukka": "இருக்கிறார்களா",
    "pannuven": "செய்வேன்",
    "panren": "செய்கிறேன்",
    "pannu": "செய்",
    "poren": "போகிறேன்",
    "poranga": "போகிறார்கள்",
    "pooven": "போவேன்",
    "tharen": "தருகிறேன்",
    "tharuven": "தருவேன்",
    "ketken": "கேட்கிறேன்",
    "ketkuven": "கேட்பேன்",
    "padichen": "படித்தேன்",
    "saapduven": "சாப்பிடுவேன்",
    "saapren": "சாப்பிடுகிறேன்",
    "thoonguren": "தூங்குகிறேன்",

    # Common words
    "late": "தாமதமாக",
    "enga": "எங்கே",
    "enna": "என்ன",
    "epdi": "எப்படி",
    "eppov": "எப்போது",
    "edhuku": "ஏன்",
    "yaaruku": "யாருக்கு",
    "yaaru": "யார்",
    "evlo": "எவ்வளவு",
    "oru": "ஒரு",
    "romba": "மிகவும்",
    "konjam": "கொஞ்சம்",
    "nalla": "நல்ல",
    "nallavela": "நல்லவேளை",
    "ketta": "கெட்ட",
    "periya": "பெரிய",
    "sinna": "சின்ன",
    "pudhu": "புது",
    "pazhaya": "பழைய",
    "azhaga": "அழகாக",
    "seekiram": "விரைவாக",
    "maela": "மேலே",
    "keela": "கீழே",
    "munn": "முன்னால்",
    "pinn": "பின்னால்",

    # Discourse markers
    "da": "",       # male vocative (drop)
    "di": "",       # female vocative (drop)
    "pa": "",       # informal suffix (drop)
    "nga": "கள்",   # plural marker
    "la": "இல்",    # locative
    "le": "இல்",
    "ku": "க்கு",   # dative
    "kku": "க்கு",
    "odey": "உடன்",
    "oda": "உடன்",
    "kita": "கிட்டே",

    # Common Tanglish words
    "ok": "சரி",
    "okay": "சரி",
    "bro": "நண்பா",
    "anna": "அண்ணா",
    "akka": "அக்கா",
    "amma": "அம்மா",
    "appa": "அப்பா",
    "thambi": "தம்பி",
    "thangachi": "தங்கை",

    # Numbers
    "onnu": "ஒன்று",
    "rendu": "இரண்டு",
    "moonu": "மூன்று",
    "naalu": "நான்கு",
    "anju": "ஐந்து",
    "aaru": "ஆறு",
    "ezhu": "ஏழு",
    "ettu": "எட்டு",
    "ombodhu": "ஒன்பது",
    "pathu": "பத்து",

    # Question words
    "enna panre": "என்ன செய்கிறீர்கள்",
    "enga pore": "எங்கே போகிறீர்கள்",
    "epdi iruke": "எப்படி இருக்கிறீர்கள்",

    # Common phrases
    "naan late da": "நான் தாமதமாக வந்தேன்",
    "romba nalla irukku": "மிகவும் நன்றாக இருக்கிறது",
    "enna panre nee": "நீ என்ன செய்கிறாய்",
    "enga pore": "நீங்கள் எங்கே போகிறீர்கள்",
    "konjam wait pannu": "கொஞ்சம் காத்திரு",
    "seekiram va": "விரைவாக வா",
}

# Tanglish patterns (English words mixed in Tamil sentences)
TANGLISH_PATTERNS = [
    (r'\bwait\b', 'காத்திரு'),
    (r'\bplease\b', 'தயவுசெய்து'),
    (r'\bthanks\b', 'நன்றி'),
    (r'\bsorry\b', 'மன்னிக்கவும்'),
    (r'\bhello\b', 'வணக்கம்'),
    (r'\bbye\b', 'விடைபெறுகிறேன்'),
    (r'\byes\b', 'ஆம்'),
    (r'\bno\b', 'இல்லை'),
    (r'\bmoney\b', 'பணம்'),
    (r'\btime\b', 'நேரம்'),
    (r'\bwork\b', 'வேலை'),
    (r'\boffice\b', 'அலுவலகம்'),
    (r'\bschool\b', 'பள்ளி'),
    (r'\bcollege\b', 'கல்லூரி'),
    (r'\bphone\b', 'தொலைபேசி'),
    (r'\bhospital\b', 'மருத்துவமனை'),
    (r'\bhouse\b', 'வீடு'),
    (r'\bcar\b', 'கார்'),
    (r'\bbike\b', 'இருசக்கர வாகனம்'),
    (r'\bfood\b', 'உணவு'),
    (r'\bwater\b', 'தண்ணீர்'),
    (r'\bbook\b', 'புத்தகம்'),
]


def normalize_colloquial_tamil(text: str) -> str:
    """
    Convert colloquial/spoken Tamil to formal Tamil.
    Handles:
    - Romanized Tamil (Tanglish) → Tamil script
    - Common spoken contractions
    - Dropped endings / suffix restoration
    """
    if not text or not text.strip():
        return text

    original = text
    text_lower = text.lower().strip()

    # 1. Try full-phrase matches first (longest match wins)
    sorted_phrases = sorted(COLLOQUIAL_DICT.keys(), key=len, reverse=True)
    for phrase in sorted_phrases:
        if phrase in text_lower and len(phrase.split()) > 1:
            formal = COLLOQUIAL_DICT[phrase]
            if formal:
                text_lower = text_lower.replace(phrase, formal)
            else:
                text_lower = text_lower.replace(phrase, '')

    # 2. Word-level normalization
    words = text_lower.split()
    normalized_words = []
    for word in words:
        # Strip punctuation for lookup
        clean = re.sub(r'[^\w]', '', word)
        if clean in COLLOQUIAL_DICT:
            replacement = COLLOQUIAL_DICT[clean]
            if replacement:  # Not empty (dropped suffix)
                normalized_words.append(replacement)
        else:
            normalized_words.append(word)

    text_normalized = ' '.join(normalized_words)

    # 3. Apply Tanglish patterns for remaining English words
    for pattern, replacement in TANGLISH_PATTERNS:
        text_normalized = re.sub(pattern, replacement, text_normalized, flags=re.IGNORECASE)

    # 4. Clean up extra spaces
    text_normalized = re.sub(r'\s+', ' ', text_normalized).strip()

    if text_normalized != original:
        logger.debug(f"Normalized: '{original}' → '{text_normalized}'")

    return text_normalized if text_normalized else original


def is_tamil_script(text: str) -> bool:
    """Check if text contains Tamil Unicode characters."""
    if not text:
        return False
    tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    return tamil_chars > len(text) * 0.1


def is_likely_tamil(text: str, language_code: str = '') -> bool:
    """Determine if text is Tamil (script or romanized)."""
    if language_code in ('ta', 'tamil'):
        return True
    if is_tamil_script(text):
        return True

    # Check for common Tamil romanization markers
    tamil_markers = [
        'naan', 'nee', 'avan', 'aval', 'avanga', 'enna', 'enga',
        'epdi', 'romba', 'konjam', 'irukku', 'iruken', 'poren',
        'varen', 'panren', 'da', 'di', 'pa', 'anna', 'akka',
    ]
    text_lower = text.lower()
    marker_count = sum(1 for m in tamil_markers if re.search(r'\b' + m + r'\b', text_lower))
    return marker_count >= 2


def process_segments_normalization(segments: list) -> list:
    """Apply normalization to all transcribed segments."""
    for seg in segments:
        raw_text = seg.get('text', '')
        language = seg.get('language', 'unknown')

        if is_likely_tamil(raw_text, language):
            seg['normalized_text'] = normalize_colloquial_tamil(raw_text)
            seg['is_tamil'] = True
        else:
            seg['normalized_text'] = raw_text
            seg['is_tamil'] = False

    return segments
