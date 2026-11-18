from src.translator import detect_language, translate_to_english, translate_from_english

# A Tamil question meaning "what is Java?"
tamil_query = "ஜாவா என்றால் என்ன?"

# A Hindi question meaning "what is Java?"
hindi_query = "जावा क्या है"

print("--- TAMIL TEST ---")
print("Detected language:", detect_language(tamil_query))
english_from_tamil = translate_to_english(tamil_query)
print("Translated to English:", english_from_tamil)

print("\n--- HINDI TEST ---")
print("Detected language:", detect_language(hindi_query))
english_from_hindi = translate_to_english(hindi_query)
print("Translated to English:", english_from_hindi)

print("\n--- TRANSLATE BACK TEST ---")
sample_answer = "Ben works at TCS."
print("English -> Tamil:", translate_from_english(sample_answer, "ta"))
print("English -> Hindi:", translate_from_english(sample_answer, "hi"))

#  தரவு மையம் என்றால் என்ன?
#  डेटा सेंटर क्या है