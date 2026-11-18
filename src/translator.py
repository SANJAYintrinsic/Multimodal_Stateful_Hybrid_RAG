"""
Cross-lingual support: detects whether a query is Tamil, Hindi, or English,
translates non-English queries -> English for retrieval, and translates
answers back to the user's original language. Free, local library, no API key.
"""

from deep_translator import GoogleTranslator, MyMemoryTranslator


_MYMEMORY_CODES = {
    "en": "en-GB",
    "ta": "ta-IN",
    "hi": "hi-IN",
}

def _split_into_safe_chunks(text: str, max_chars: int = 450) -> list[str]:
    """
    MyMemory rejects any single request over 500 characters. Gemini's
    answers are often longer than that, so this splits text into pieces
    at sentence boundaries (never mid-sentence), each safely under the
    limit, so long answers can still be translated in full.
    """
    sentences = text.replace("\n", " ").split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        piece = sentence if sentence.endswith(".") else sentence + "."
        if len(current) + len(piece) + 1 <= max_chars:
            current = f"{current} {piece}".strip()
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks or [text]


def _translate(text: str, source: str, target: str) -> str:
    """
    Translate using MyMemory (a free translation service, separate from
    Google) as the primary backend, since Google's free endpoint can hit
    an aggressive rate limit after heavy use. Falls back to Google
    Translate if MyMemory ever fails for some reason.

    Long text is split into sentence-based chunks first, since MyMemory
    rejects any single request over 500 characters.
    """
    mm_source = _MYMEMORY_CODES.get(source, source)
    mm_target = _MYMEMORY_CODES.get(target, target)

    chunks = _split_into_safe_chunks(text)
    translated_parts = []

    for chunk in chunks:
        try:
            translated_parts.append(
                MyMemoryTranslator(source=mm_source, target=mm_target).translate(chunk)
            )
        except Exception:
            translated_parts.append(
                GoogleTranslator(source=source, target=target).translate(chunk)
            )

    return " ".join(translated_parts)

def detect_language(text: str) -> str:
    """
    Lightweight language detection via Unicode script ranges:
      - Tamil script:     U+0B80 to U+0BFF
      - Devanagari (Hindi) script: U+0900 to U+097F
    This avoids needing a separate detection library/API for a
    small, fixed set of supported languages (English/Tamil/Hindi).
    """
    for ch in text:
        if "\u0b80" <= ch <= "\u0bff":
            return "ta"
        if "\u0900" <= ch <= "\u097f":
            return "hi"
    return "en"


def translate_to_english(text: str) -> str:
    """Translate a Tamil or Hindi query -> English (for retrieval against English docs)."""
    lang = detect_language(text)
    if lang in ("ta", "hi"):
        return _translate(text, source=lang, target="en")
    return text


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate the final English answer -> the target language (ta or hi)."""
    return _translate(text, source="en", target=target_lang)


def respond_in_original_language(english_answer: str, original_query: str) -> str:
    """
    Given the LLM's English answer and the user's original query,
    return the answer in whichever language the user asked in
    (English, Tamil, or Hindi).
    """
    lang = detect_language(original_query)
    if lang in ("ta", "hi"):
        return translate_from_english(english_answer, lang)
    return english_answer