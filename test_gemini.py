from google import genai
from src.config import GEMINI_API_KEY, GEMINI_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents="Say hello and confirm you are working, in one short sentence."
)
print(response.text)