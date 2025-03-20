def get_summary_prompt(text):
    """
    Returns a prompt template for summarizing text.
    """
    return f"Summarize the following text:\n\n{text}\n\nSummary:"

def get_sentiment_prompt(text):
    """
    Returns a prompt template for sentiment analysis.
    """
    return f"Analyze the sentiment of the following text and respond with Positive, Negative, or Neutral:\n\n{text}\n\nSentiment:"

def get_keywords_prompt(text):
    """
    Returns a prompt template for extracting keywords.
    """
    return f"Extract the main keywords from the following text, separated by commas:\n\n{text}\n\nKeywords:"

def get_qa_prompt(text, question):
    """
    Returns a prompt template for Q&A.
    """
    return f"Based on the following text, answer the question.\n\nText: {text}\n\nQuestion: {question}\n\nAnswer:"
