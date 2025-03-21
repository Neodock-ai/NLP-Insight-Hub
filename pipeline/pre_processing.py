import re
import logging

# Configure module logger
logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Enhanced text cleaning and normalization for better NLP performance.
    
    Args:
        text (str): The raw input text
        
    Returns:
        str: Cleaned and normalized text
    """
    try:
        # Handle None or empty text
        if not text:
            logger.warning("Empty text received for cleaning")
            return ""
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', text)
        
        # Remove unwanted special characters but keep important punctuation
        cleaned = re.sub(r'[^\w\s.,;:!?\"\'()\[\]{}*&%$#@-]', '', cleaned)
        
        # Fix common OCR issues and errors
        cleaned = _fix_common_ocr_errors(cleaned)
        
        # Fix sentence spacing
        cleaned = _fix_sentence_spacing(cleaned)
        
        # Remove duplicate sentences that might be artifacts of PDF extraction
        cleaned = _remove_duplicate_sentences(cleaned)
        
        # Normalize quotes and apostrophes
        cleaned = _normalize_quotes(cleaned)
        
        return cleaned.strip()
        
    except Exception as e:
        logger.error(f"Error during text cleaning: {str(e)}")
        # Return original text if cleaning fails
        return text.strip() if isinstance(text, str) else str(text).strip()

def preprocess_for_summarization(text):
    """
    Special preprocessing for summarization tasks.
    
    Args:
        text (str): The input text
        
    Returns:
        str: Text optimized for summarization
    """
    try:
        # Basic cleaning first
        cleaned = clean_text(text)
        
        # Remove boilerplate text that often appears in documents
        cleaned = _remove_boilerplate(cleaned)
        
        # Ensure proper paragraph breaks for better structure recognition
        cleaned = _normalize_paragraphs(cleaned)
        
        # Limit text length for large documents (focus on beginning and end which often contain key info)
        if len(cleaned) > 10000:
            logger.info("Long text detected, truncating for summarization")
            # Take first 6000 chars and last 4000 chars
            cleaned = cleaned[:6000] + "\n\n[...]\n\n" + cleaned[-4000:]
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Error during summarization preprocessing: {str(e)}")
        return clean_text(text)

def preprocess_for_sentiment(text):
    """
    Special preprocessing for sentiment analysis tasks.
    
    Args:
        text (str): The input text
        
    Returns:
        str: Text optimized for sentiment analysis
    """
    try:
        # Basic cleaning first
        cleaned = clean_text(text)
        
        # Normalize common emoticons and emoji representations
        cleaned = _normalize_emoticons(cleaned)
        
        # Normalize emphasized text (e.g., ALL CAPS, excessive punctuation)
        cleaned = _normalize_emphasis(cleaned)
        
        # For very long texts, use strategic sampling for sentiment
        if len(cleaned) > 5000:
            logger.info("Long text detected, sampling for sentiment analysis")
            cleaned = _strategic_sample_for_sentiment(cleaned)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Error during sentiment preprocessing: {str(e)}")
        return clean_text(text)

def preprocess_for_keywords(text):
    """
    Special preprocessing for keyword extraction tasks.
    
    Args:
        text (str): The input text
        
    Returns:
        str: Text optimized for keyword extraction
    """
    try:
        # Basic cleaning first
        cleaned = clean_text(text)
        
        # Remove generic headers and footers
        cleaned = _remove_headers_footers(cleaned)
        
        # For very long texts, use strategic sampling for keywords
        if len(cleaned) > 10000:
            logger.info("Long text detected, sampling for keyword extraction")
            # Focus on headings, first sentences of paragraphs, and beginning/end
            cleaned = _strategic_sample_for_keywords(cleaned)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Error during keyword preprocessing: {str(e)}")
        return clean_text(text)

def _fix_common_ocr_errors(text):
    """
    Fixes common OCR errors in text.
    """
    # Common OCR error patterns
    replacements = [
        (r'l\b', 'i'),           # "l" at end of word often should be "i"
        (r'\bl\b', 'I'),         # "l" as single letter often should be "I"
        (r'0', 'o'),             # "0" often should be "o" in words
        (r'1', 'l'),             # "1" often should be "l" or "I" in words
        (r'S', '5'),             # "S" vs "5" confusion
        (r'rn', 'm'),            # "rn" often misread as "m"
        (r'IVI', 'M'),           # "IVI" often misread as "M"
        (r'\bI\s+([a-z])', r'I \1'),  # Space after "I"
    ]
    
    # Only apply OCR fixes if the text appears to have OCR issues
    ocr_indicators = ['l l', 'l l l', '0n', 'c0m', 'f0r', '0f', 'l1fe', 'lS', 'l5']
    if any(indicator in text for indicator in ocr_indicators):
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
    
    return text

def _fix_sentence_spacing(text):
    """
    Ensures proper spacing between sentences.
    """
    # Fix spacing after periods, question marks, and exclamation points
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Fix missing space after comma
    text = re.sub(r',([A-Za-z])', r', \1', text)
    
    return text

def _remove_duplicate_sentences(text):
    """
    Removes duplicate sentences that often appear in PDFs.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Remove duplicates while preserving order
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        # Normalize for comparison
        normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
        if len(normalized) > 20 and normalized not in seen:
            seen.add(normalized)
            unique_sentences.append(sentence)
        elif len(normalized) <= 20:
            # Keep short sentences as they might be headers or important fragments
            unique_sentences.append(sentence)
    
    # Reassemble text
    return ' '.join(unique_sentences)

def _normalize_quotes(text):
    """
    Normalizes different quote styles to standard format.
    """
    # Replace various quote types with standard double quotes
    text = re.sub(r'[""]', '"', text)
    
    # Replace various apostrophe types with standard apostrophe
    text = re.sub(r'['']', "'", text)
    
    return text

def _remove_boilerplate(text):
    """
    Removes common boilerplate text found in documents.
    """
    # Common boilerplate patterns to remove
    boilerplate_patterns = [
        r'(?i)all rights reserved\.?',
        r'(?i)confidential(?:ity)? notice:?.*?,
        r'(?i)copyright ©.*?(?:\d{4}|\[year\]).*?,
        r'(?i)this (?:document|email) (?:is|contains).*?confidential.*?,
        r'(?i)disclaimer:.*?,
        r'(?i)legal notice:.*?,
        r'(?i)produced by.*?,
        r'(?i)generated by.*?,
        r'(?i)page \d+ of \d+',
        r'(?i)https?://(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+/?[^\s]*'
    ]
    
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text)
    
    return text

def _normalize_paragraphs(text):
    """
    Normalizes paragraph breaks for better structure.
    """
    # Replace various paragraph break styles with standard double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Ensure paragraphs have proper breaks
    text = re.sub(r'([.!?])\s*(\n?)([A-Z])', r'\1\n\n\3', text)
    
    return text

def _normalize_emoticons(text):
    """
    Normalizes emoticons and emoji representations.
    """
    # Common emoticon patterns and their normalized forms
    emoticons = {
        r':-?\)': ' [POSITIVE_EMOJI] ',
        r':-?\(': ' [NEGATIVE_EMOJI] ',
        r':-?D': ' [VERY_POSITIVE_EMOJI] ',
        r':-?P': ' [PLAYFUL_EMOJI] ',
        r':-?/': ' [SKEPTICAL_EMOJI] ',
        r':-?\|': ' [NEUTRAL_EMOJI] ',
        r';-?\)': ' [WINK_EMOJI] ',
    }
    
    for pattern, replacement in emoticons.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def _normalize_emphasis(text):
    """
    Normalizes emphasized text like ALL CAPS or excessive punctuation.
    """
    # Mark ALL CAPS text (often indicates shouting/emphasis)
    def mark_caps(match):
        text = match.group(0)
        if len(text) >= 3:  # Only mark if at least 3 characters
            return f" [EMPHASIS] {text.lower()} [/EMPHASIS] "
        return text
    
    text = re.sub(r'\b[A-Z]{3,}\b', mark_caps, text)
    
    # Normalize multiple exclamation or question marks
    text = re.sub(r'!{2,}', ' [STRONG_EMPHASIS] ! ', text)
    text = re.sub(r'\?{2,}', ' [STRONG_QUESTION] ? ', text)
    
    return text

def _strategic_sample_for_sentiment(text):
    """
    Strategically samples text for sentiment analysis on long documents.
    """
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    if len(paragraphs) <= 5:
        return text
    
    # Take the first two paragraphs (often introductory)
    result = '\n\n'.join(paragraphs[:2]) + '\n\n'
    
    # Take sample paragraphs from the middle
    middle_start = max(2, len(paragraphs) // 4)
    middle_end = min(len(paragraphs) - 2, 3 * len(paragraphs) // 4)
    
    # Sample up to 4 paragraphs from the middle
    sample_count = min(4, middle_end - middle_start)
    if sample_count > 0:
        step_size = (middle_end - middle_start) // sample_count
        for i in range(middle_start, middle_end, step_size):
            if i < len(paragraphs):
                result += paragraphs[i] + '\n\n'
    
    # Take the last two paragraphs (often conclusive)
    result += '\n\n' + '\n\n'.join(paragraphs[-2:])
    
    return result

def _remove_headers_footers(text):
    """
    Removes common headers and footers from documents.
    """
    # Common header/footer patterns
    patterns = [
        r'^\s*\d+\s*,  # Just a page number
        r'^\s*Page \d+ of \d+\s*,  # Page X of Y
        r'^\s*[^a-zA-Z]*\d+[^a-zA-Z]*\s*,  # Numbers with symbols
        r'^\s*[A-Za-z0-9_\.-]+@[A-Za-z0-9_\.-]+\.\w+\s*,  # Email address only
        r'^\s*https?://\S+\s*,  # URLs only
        r'^\s*www\.\S+\s*,  # www URLs only
        r'^\s*[A-Za-z\s]+ \| [A-Za-z\s]+\s*,  # Title | Subtitle format
        r'^\s*CONFIDENTIAL\s*,  # Just "CONFIDENTIAL"
        r'^\s*DRAFT\s*,  # Just "DRAFT"
    ]
    
    # Process line by line
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        is_header_footer = False
        for pattern in patterns:
            if re.match(pattern, line):
                is_header_footer = True
                break
        
        if not is_header_footer:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def _strategic_sample_for_keywords(text):
    """
    Strategically samples text for keyword extraction on long documents.
    """
    # Extract headings (assuming headings don't end with periods)
    headings = re.findall(r'(?m)^[A-Z][\w\s:]{3,60}(?<!\.), text)
    
    # Extract first sentences of paragraphs
    paragraphs = text.split('\n\n')
    first_sentences = []
    
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)
        if sentences:
            first_sentences.append(sentences[0])
    
    # Prepare the sampled text
    sample = ""
    
    # Add headings
    if headings:
        sample += "HEADINGS:\n" + "\n".join(headings) + "\n\n"
    
    # Add first sentences
    if first_sentences:
        sample += "KEY SENTENCES:\n" + " ".join(first_sentences[:15]) + "\n\n"
    
    # Add beginning (first 2000 chars)
    beginning = text[:2000].strip()
    if beginning:
        sample += "BEGINNING:\n" + beginning + "\n\n"
    
    # Add end (last 2000 chars)
    end = text[-2000:].strip()
    if end:
        sample += "END:\n" + end
    
    return sample
