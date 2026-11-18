
import os
import tempfile

from pypdf import PdfReader
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
import speech_recognition as sr


def read_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text.strip()


def read_docx(filepath: str) -> str:
    doc = DocxDocument(filepath)
    return "\n".join(p.text for p in doc.paragraphs).strip()


def read_image(filepath: str) -> str:
    """OCR: extract any text visible in an image."""
    img = Image.open(filepath)
    return pytesseract.image_to_string(img).strip()

def read_audio(filepath: str, target_chunk_seconds: int = 15) -> str:
    """
    Transcribe speech from an audio file (wav/flac work best).

    Google's FREE Web Speech API (used here, no key needed) tends to cut off
    or stop transcribing after roughly 15-20 seconds of continuous audio -
    a known limitation of this particular free endpoint. To work around it,
    we split the audio into chunks of roughly EQUAL length (close to
    target_chunk_seconds), rather than fixed-size chunks that can leave an
    awkward, too-short final chunk (e.g. a leftover 2-second piece) that's
    too brief for the API to recognize anything in - which is what was
    silently dropping the last few seconds of content before this fix.
    """
    recognizer = sr.Recognizer()
    transcript_parts = []

    with sr.AudioFile(filepath) as source:
        total_duration = source.DURATION

        # Work out how many chunks we need, then split evenly across the
        # full duration so every chunk - including the last one - is a
        # similar, comfortably-transcribable length.
        num_chunks = max(1, round(total_duration / target_chunk_seconds))
        chunk_length = total_duration / num_chunks

        elapsed = 0
        for _ in range(num_chunks):
            remaining = total_duration - elapsed
            duration = min(chunk_length, remaining) if remaining > 0 else chunk_length
            audio_chunk = recognizer.record(source, duration=duration)
            try:
                text = recognizer.recognize_google(audio_chunk)
                transcript_parts.append(text)
            except sr.UnknownValueError:
                pass  # this chunk was silent/unclear - just skip it, not a failure
            except sr.RequestError as e:
                transcript_parts.append(f"[Speech recognition service error: {e}]")
            elapsed += duration

    return " ".join(transcript_parts).strip()

def read_mp3(filepath: str) -> str:
    """Convert MP3 -> WAV (SpeechRecognition needs WAV/FLAC), then transcribe."""
    from moviepy import AudioFileClip

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    clip = AudioFileClip(filepath)
    clip.write_audiofile(wav_path, logger=None)
    clip.close()

    text = read_audio(wav_path)
    os.remove(wav_path)
    return text


def read_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def extract_text(filepath: str) -> str:
    """Dispatch to the right reader based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return read_pdf(filepath)
    elif ext == ".docx":
        return read_docx(filepath)
    elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
        return read_image(filepath)
    elif ext in (".wav", ".flac"):
        return read_audio(filepath)
    elif ext == ".mp3":
        return read_mp3(filepath)
    elif ext == ".txt":
        return read_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Simple sliding-window chunking so each piece fits well for embedding/search."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]