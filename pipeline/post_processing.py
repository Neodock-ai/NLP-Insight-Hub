def format_summary(summary):
    """
    Formats the summary output.
    """
    return summary.strip()

def format_sentiment(sentiment):
    """
    Formats the sentiment analysis output.
    """
    # Return a more detailed sentiment analysis
    sentiment = sentiment.strip()
    if sentiment.lower() == "positive":
        return "The text has a **positive** sentiment overall."
    elif sentiment.lower() == "negative":
        return "The text has a **negative** sentiment overall."
    else:
        return "The text has a **neutral** sentiment overall."

def format_keywords(keywords):
    """
    Formats the keyword extraction output as a markdown list.
    """
    # First check if keywords is already a list
    if isinstance(keywords, list):
        keyword_list = keywords
    else:
        # Otherwise split the string
        keyword_list = [keyword.strip() for keyword in keywords.split(',') if keyword.strip()]
    
    # Format as markdown
    if keyword_list:
        return "**Key concepts extracted:**\n- " + "\n- ".join(keyword_list)
    else:
        return "No keywords were extracted."

def format_qa(answer):
    """
    Formats the Q&A output.
    """
    return f"**Answer:** {answer.strip()}"
