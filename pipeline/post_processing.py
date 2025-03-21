import re

def format_summary(summary):
    """
    Formats the summary output with improved formatting and structure.
    
    Args:
        summary (str): The raw summary text from the model
        
    Returns:
        str: Formatted summary with markdown styling
    """
    summary = summary.strip()
    
    # Remove any prefixes like "Summary:" that might be in the output
    summary = re.sub(r'^(Summary:?\s*)', '', summary, flags=re.IGNORECASE)
    
    # Split into paragraphs if it's a long summary
    paragraphs = summary.split('\n\n')
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        # Clean up the paragraph
        clean_paragraph = paragraph.strip()
        if clean_paragraph:
            formatted_paragraphs.append(clean_paragraph)
    
    # If there are multiple paragraphs, format them with bullet points
    if len(formatted_paragraphs) > 1:
        formatted_text = "## Text Summary\n\n"
        for para in formatted_paragraphs:
            formatted_text += f"• {para}\n\n"
        return formatted_text
    else:
        # For single paragraphs, use simpler formatting
        return f"""## Text Summary

{summary}
"""

def format_sentiment(sentiment_result):
    """
    Enhanced sentiment analysis formatting with better structure.
    
    Args:
        sentiment_result: Can be a string or a dictionary with detailed sentiment data
        
    Returns:
        dict: Structured sentiment data with formatted text
    """
    # Handle different input types
    if isinstance(sentiment_result, dict):
        # Use the sentiment data from the dict if available
        if 'sentiment' in sentiment_result:
            sentiment = sentiment_result['sentiment']
        else:
            # Extract sentiment from scores if available
            scores = sentiment_result.get('scores', {})
            if scores:
                max_sentiment = max(scores.items(), key=lambda x: x[1])
                sentiment = max_sentiment[0].capitalize()
            else:
                # Default if we can't determine
                sentiment = "Neutral"
    else:
        # Process the string input
        sentiment_text = sentiment_result.strip()
        
        # Extract the sentiment classification from the text
        if "positive" in sentiment_text.lower():
            sentiment = "Positive"
        elif "negative" in sentiment_text.lower():
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
    
    # Define emoji and color based on sentiment
    if sentiment.lower() == "positive":
        emoji = "😃"
        color = "green"
    elif sentiment.lower() == "negative":
        emoji = "😞"
        color = "red"
    else:
        emoji = "😐"
        color = "gray"
    
    # Extract explanation if available
    explanation = ""
    if isinstance(sentiment_result, str):
        # Try to extract explanation after the sentiment classification
        match = re.search(r'(positive|negative|neutral)[:\s]+(.+)', sentiment_result, re.IGNORECASE)
        if match:
            explanation = match.group(2).strip()
    
    # Create the formatted output
    markdown_text = f"""## Sentiment Analysis

**Overall sentiment:** <span style="color:{color}">**{sentiment}**</span> {emoji}

"""
    
    # Add explanation if available
    if explanation:
        markdown_text += f"**Analysis:** {explanation}\n\n"
    
    markdown_text += "*Note: This is an automated sentiment analysis that evaluates the emotional tone of the text.*"
    
    # Return a dictionary with both the raw data and formatted text
    return {
        "sentiment": sentiment,
        "emoji": emoji,
        "color": color,
        "text": markdown_text,
        "raw_sentiment": sentiment.lower()  # For compatibility with visualization
    }

def format_keywords(keywords):
    """
    Enhanced keyword formatting with better structure and presentation.
    
    Args:
        keywords: String of comma-separated keywords or list of keyword tuples
        
    Returns:
        dict: Structured keyword data with formatted text
    """
    # Process different input types
    if isinstance(keywords, list):
        if keywords and isinstance(keywords[0], tuple):
            # It's a list of (keyword, weight) tuples
            keyword_items = [kw for kw, _ in keywords]
        else:
            # It's a simple list of keywords
            keyword_items = keywords
    elif isinstance(keywords, str):
        # It's a comma-separated string
        keyword_items = [k.strip() for k in keywords.split(',') if k.strip()]
    else:
        # Default empty list
        keyword_items = []
    
    # Format the output
    if keyword_items:
        # Group keywords by importance (assume first ones are more important)
        primary_keywords = keyword_items[:min(5, len(keyword_items))]
        secondary_keywords = keyword_items[min(5, len(keyword_items)):min(10, len(keyword_items))]
        remaining_keywords = keyword_items[min(10, len(keyword_items)):]
        
        markdown_text = "## Key Topics & Concepts\n\n"
        
        if primary_keywords:
            markdown_text += "**Primary Topics:**\n"
            for kw in primary_keywords:
                markdown_text += f"- {kw}\n"
            markdown_text += "\n"
        
        if secondary_keywords:
            markdown_text += "**Secondary Topics:**\n"
            for kw in secondary_keywords:
                markdown_text += f"- {kw}\n"
            markdown_text += "\n"
        
        if remaining_keywords:
            markdown_text += "**Additional Concepts:**\n"
            for kw in remaining_keywords:
                markdown_text += f"- {kw}\n"
            markdown_text += "\n"
    else:
        markdown_text = "## Key Topics & Concepts\n\nNo significant keywords were identified in the text."
    
    # Return a dictionary with both the raw data and formatted text
    return {
        "text": markdown_text,
        "data": keywords  # Keep original data for visualization
    }

def format_qa(answer):
    """
    Enhanced Q&A formatting with better structure.
    
    Args:
        answer (str): The raw answer from the model
        
    Returns:
        str: Formatted answer with markdown styling
    """
    answer = answer.strip()
    
    # Remove any prefixes like "Answer:" that might be in the output
    answer = re.sub(r'^(Answer:?\s*)', '', answer, flags=re.IGNORECASE)
    
    # Format the answer
    formatted_answer = f"""### Answer

{answer}

*Note: This answer is generated based on the provided text and context.*
"""
    
    return formatted_answer
