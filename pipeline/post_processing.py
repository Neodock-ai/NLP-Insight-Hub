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
    Enhanced sentiment analysis formatting with comprehensive insight presentation.
    
    Args:
        sentiment_result: Can be a string or a dictionary with detailed sentiment data
        
    Returns:
        dict: Structured sentiment data with formatted text and visualization data
    """
    # Handle different input types
    if isinstance(sentiment_result, dict):
        # Use the sentiment data from the dict if available
        if 'sentiment' in sentiment_result:
            sentiment = sentiment_result['sentiment']
            scores = sentiment_result.get('scores', {})
            confidence = sentiment_result.get('confidence', 0.0)
            aspects = sentiment_result.get('aspects', {})
            explanation = sentiment_result.get('explanation', '')
            key_phrases = sentiment_result.get('key_phrases', [])
        else:
            # Extract sentiment from scores if available
            scores = sentiment_result.get('scores', {})
            if scores:
                max_sentiment = max(scores.items(), key=lambda x: x[1])
                sentiment = max_sentiment[0].capitalize()
            else:
                # Default if we can't determine
                sentiment = "Neutral"
            confidence = 0.0
            aspects = {}
            explanation = ''
            key_phrases = []
    else:
        # Process the string input
        sentiment_text = str(sentiment_result).strip()
        
        # Extract the sentiment classification from the text
        if "positive" in sentiment_text.lower():
            sentiment = "Positive"
            scores = {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
        elif "negative" in sentiment_text.lower():
            sentiment = "Negative"
            scores = {"positive": 0.1, "neutral": 0.2, "negative": 0.7}
        else:
            sentiment = "Neutral"
            scores = {"positive": 0.3, "neutral": 0.5, "negative": 0.2}
        
        confidence = max(scores.values())
        aspects = {}
        key_phrases = []
        
        # Try to extract explanation after the sentiment classification
        explanation = ""
        match = re.search(r'(positive|negative|neutral)[:\s]+(.+)', sentiment_text, re.IGNORECASE)
        if match:
            explanation = match.group(2).strip()
    
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
    
    # Calculate confidence description
    if isinstance(confidence, (int, float)):
        if confidence > 0.8:
            confidence_text = "High confidence"
        elif confidence > 0.6:
            confidence_text = "Moderate confidence"
        else:
            confidence_text = "Low confidence"
    else:
        confidence_text = ""
    
    # Create the formatted output with enhanced details
    markdown_text = f"""## Sentiment Analysis

**Overall sentiment:** <span style="color:{color}">**{sentiment}**</span> {emoji} {confidence_text}

"""
    
    # Add explanation if available
    if explanation:
        markdown_text += f"**Analysis:** {explanation}\n\n"
    
    # Add key sentiment phrases if available
    if key_phrases:
        markdown_text += "**Key sentiment phrases:**\n"
        for i, phrase_data in enumerate(key_phrases[:3]):  # Show top 3 phrases
            phrase = phrase_data.get("phrase", "")
            phrase_sentiment = phrase_data.get("sentiment", "")
            
            if phrase_sentiment.lower() == "positive":
                phrase_emoji = "✓"
                phrase_color = "green"
            elif phrase_sentiment.lower() == "negative":
                phrase_emoji = "✗"
                phrase_color = "red"
            else:
                phrase_emoji = "•"
                phrase_color = "gray"
                
            markdown_text += f"{phrase_emoji} <span style=\"color:{phrase_color}\">\"{phrase}\"</span>\n"
        markdown_text += "\n"
    
    # Add aspect-based sentiment if available
    if aspects:
        markdown_text += "**Aspect analysis:**\n"
        
        # Sort aspects by absolute sentiment value
        sorted_aspects = sorted(aspects.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for i, (aspect, value) in enumerate(sorted_aspects[:5]):  # Show top 5 aspects
            if value > 0:
                aspect_emoji = "👍"
                aspect_text = f"{aspect.capitalize()}: Positive"
            elif value < 0:
                aspect_emoji = "👎"
                aspect_text = f"{aspect.capitalize()}: Negative"
            else:
                aspect_emoji = "•"
                aspect_text = f"{aspect.capitalize()}: Neutral"
                
            markdown_text += f"{aspect_emoji} {aspect_text}\n"
        markdown_text += "\n"
    
    # Add confidence breakdown
    if scores:
        pos_pct = int(scores.get("positive", 0) * 100)
        neu_pct = int(scores.get("neutral", 0) * 100)
        neg_pct = int(scores.get("negative", 0) * 100)
        
        markdown_text += f"**Sentiment distribution:** {pos_pct}% Positive, {neu_pct}% Neutral, {neg_pct}% Negative\n\n"
    
    markdown_text += "*Note: This is an automated sentiment analysis that evaluates the emotional tone of the text.*"
    
    # Return a dictionary with both the raw data and formatted text
    return {
        "sentiment": sentiment,
        "emoji": emoji,
        "color": color,
        "confidence": confidence,
        "aspects": aspects,
        "key_phrases": key_phrases,
        "text": markdown_text,
        "raw_sentiment": sentiment.lower(),  # For compatibility with visualization
        "scores": scores,  # For visualization
        "explanation": explanation
    }
    
def format_keywords(keywords):
    """
    Enhanced keyword formatting with rich insights including themes and categories.
    
    Args:
        keywords: String of comma-separated keywords, list of keyword tuples, or rich keyword data
        
    Returns:
        dict: Structured keyword data with formatted text
    """
    # Process different input types
    keyword_items = []
    themes = {}
    categories = {"entity": [], "phrase": [], "concept": [], "technical": []}
    
    if isinstance(keywords, list):
        if keywords and isinstance(keywords[0], tuple):
            # It's a list of tuples
            if len(keywords[0]) >= 4:  # Enhanced format with category and theme
                for kw, weight, category, theme in keywords:
                    keyword_items.append(kw)
                    if theme and theme != "General":
                        if theme not in themes:
                            themes[theme] = []
                        themes[theme].append(kw)
                    if category in categories:
                        categories[category].append(kw)
            elif len(keywords[0]) >= 3:  # With category
                for kw, weight, category in keywords:
                    keyword_items.append(kw)
                    if category in categories:
                        categories[category].append(kw)
            else:  # Basic (keyword, weight) tuples
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
        markdown_text = "## Key Topics & Concepts\n\n"
        
        # Display themes if available
        if themes:
            markdown_text += "### Topic Clusters\n"
            for theme, theme_keywords in themes.items():
                markdown_text += f"**{theme}**: {', '.join(theme_keywords[:5])}"
                if len(theme_keywords) > 5:
                    markdown_text += f" and {len(theme_keywords) - 5} more"
                markdown_text += "\n"
            markdown_text += "\n"
        
        # Display categories
        if any(len(cat) > 0 for cat in categories.values()):
            if categories["entity"]:
                markdown_text += "### Key Entities\n"
                markdown_text += ", ".join(categories["entity"][:8])
                if len(categories["entity"]) > 8:
                    markdown_text += f" and {len(categories['entity']) - 8} more"
                markdown_text += "\n\n"
            
            if categories["phrase"]:
                markdown_text += "### Key Phrases\n"
                markdown_text += ", ".join(categories["phrase"][:8])
                if len(categories["phrase"]) > 8:
                    markdown_text += f" and {len(categories['phrase']) - 8} more"
                markdown_text += "\n\n"
            
            if categories["technical"]:
                markdown_text += "### Technical Terms\n"
                markdown_text += ", ".join(categories["technical"][:8])
                if len(categories["technical"]) > 8:
                    markdown_text += f" and {len(categories['technical']) - 8} more"
                markdown_text += "\n\n"
            
            if categories["concept"]:
                markdown_text += "### General Concepts\n"
                markdown_text += ", ".join(categories["concept"][:8])
                if len(categories["concept"]) > 8:
                    markdown_text += f" and {len(categories['concept']) - 8} more"
                markdown_text += "\n\n"
        else:
            # Simple keyword listing if categories not available
            # Group keywords by importance
            primary_keywords = keyword_items[:min(5, len(keyword_items))]
            secondary_keywords = keyword_items[min(5, len(keyword_items)):min(15, len(keyword_items))]
            remaining_keywords = keyword_items[min(15, len(keyword_items)):]
            
            if primary_keywords:
                markdown_text += "**Primary Topics:** "
                markdown_text += ", ".join(primary_keywords)
                markdown_text += "\n\n"
            
            if secondary_keywords:
                markdown_text += "**Secondary Topics:** "
                markdown_text += ", ".join(secondary_keywords)
                markdown_text += "\n\n"
            
            if remaining_keywords:
                markdown_text += "**Additional Terms:** "
                markdown_text += ", ".join(remaining_keywords[:10])
                if len(remaining_keywords) > 10:
                    markdown_text += f" and {len(remaining_keywords) - 10} more"
                markdown_text += "\n\n"
    else:
        markdown_text = "## Key Topics & Concepts\n\nNo significant keywords were identified in the text."
    
    # Return a dictionary with both the raw data and formatted text
    return {
        "text": markdown_text,
        "data": keywords,  # Keep original data for visualization
        "themes": themes,
        "categories": categories
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
