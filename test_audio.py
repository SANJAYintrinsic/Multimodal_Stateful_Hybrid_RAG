from src.ingestion import extract_text

# Update this path to point to your actual audio file
AUDIO_FILE_PATH = "uploaded_docs/javaa.mp3"

print("Extracting text from audio... (this calls Google's free speech API, needs internet)")
text = extract_text(AUDIO_FILE_PATH)

print("\n--- TRANSCRIBED TEXT ---")
print(text)
print("--- END ---")