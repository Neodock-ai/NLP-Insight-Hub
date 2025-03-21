def format_summary(summary):
    """
    Formats the summary output.
    """
    summary = summary.strip()
    
    # Add some styling to make it more presentable
    return f"""
    ## Text Summary
    
    {summary}
    """

def format_sentiment(sentiment):
    """
    Formats the sentiment analysis output.
    """
    sentiment = sentiment.strip().lower()
    
    # Create a more visual representation
    if "positive" in sentiment:
        emoji = "😃"
        color = "green"
        sentiment_text = "Positive"
    elif "negative" in sentiment:
        emoji = "😞"
        color = "red"
        sentiment_text = "Negative"
    else:
        emoji = "😐"
        color = "gray"
        sentiment_text = "Neutral"
    
    return f"""
    ## Sentiment Analysis
    
    Overall sentiment: <span style="color:{color};font-weight:bold">{sentiment_text} {emoji}</span>
    
    *Note: This is an automated sentiment analysis and may not capture nuanced emotions.*
    """

def format_keywords(keywords):
    """
    Formats the keyword extraction output as a visual list.
    """
    # First check if keywords is already a list
    if isinstance(keywords, list):
        keyword_list = keywords
    else:
        # Otherwise split the string
        keyword_list = [keyword.strip() for keyword in keywords.split(',') if keyword.strip()]
    
    # Format as markdown with some visual elements
    if keyword_list:
        keyword_html = "\n".join([f"- {keyword}" for keyword in keyword_list])
        return f"""
        ## Key Topics & Concepts
        
        The following keywords were extracted from the text:
        
        {keyword_html}
        """
    else:
        return "No keywords were extracted from the text."

def format_qa(answer):
    """
    Formats the Q&A output with better formatting.
    """
    answer = answer.strip()
    
    return f"""
    {answer}
    
    *Note: This answer is generated based on the provided text and may not be comprehensive.*
    """
