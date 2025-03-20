from langdetect import detect # Language detection

# Node function to check if the input is in English
def is_text_in_english(text):
    """
    This function checks if the text is in English.
    It uses langdetect to detect the language of the text.
    If the language is English, it returns True.
    Else it returns False
    """
    # Use langdetect to detect language and check if text is in English
    try:
        detected_language = detect(text)
        if detected_language == 'en':  # If the language is English
            return True
        else:
            print("Text is not in English. This pipeline only supports English text.")
            return False
    except Exception as e:
        print(f"Error detecting language: {e}")