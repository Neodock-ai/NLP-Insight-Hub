import re

def clean_text(text):
    """
    Cleans and normalizes input text.
    """
    # Remove extra whitespace and unwanted characters
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()
