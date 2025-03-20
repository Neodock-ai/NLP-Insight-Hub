def format_summary(summary):
    """
    Formats the summary output.
    """
    return summary.strip()

def format_sentiment(sentiment):
    """
    Formats the sentiment analysis output.
    """
    return sentiment.strip()

def format_keywords(keywords):
    """
    Formats the keyword extraction output as a list.
    """
    return [keyword.strip() for keyword in keywords.split(',') if keyword.strip()]

def format_qa(answer):
    """
    Formats the Q&A output.
    """
    return answer.strip()
