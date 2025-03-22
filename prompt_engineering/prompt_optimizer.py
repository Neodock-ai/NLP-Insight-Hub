import re

def optimize_prompt(prompt):
    """
    Optimizes prompts for better LLM performance by applying various improvements.
    
    Args:
        prompt (str): The original prompt
        
    Returns:
        str: The optimized prompt
    """
    # Identify task type
    task_type = _identify_task_type(prompt)
    
    # Apply task-specific optimizations
    if task_type == "summary":
        prompt = _optimize_summary_prompt(prompt)
    elif task_type == "sentiment":
        prompt = _optimize_sentiment_prompt(prompt)
    elif task_type == "keywords":
        prompt = _optimize_keywords_prompt(prompt)
    elif task_type == "qa":
        prompt = _optimize_qa_prompt(prompt)
    
    # Apply general optimizations
    prompt = _apply_general_optimizations(prompt)
    
    return prompt

def _identify_task_type(prompt):
    """
    Identifies the type of NLP task from the prompt.
    """
    if re.search(r'summar(y|ize|ization)', prompt, re.IGNORECASE):
        return "summary"
    elif re.search(r'sentiment|emotion|feeling|tone', prompt, re.IGNORECASE):
        return "sentiment"
    elif re.search(r'keyword|key phrase|main topic|concept', prompt, re.IGNORECASE):
        return "keywords"
    elif re.search(r'question|answer|Q&A', prompt, re.IGNORECASE):
        return "qa"
    else:
        return "general"

def _optimize_summary_prompt(prompt):
    """
    Optimizes a summarization prompt.
    """
    # Extract the text to be summarized
    match = re.search(r'text to summarize:(.*?)summary:', prompt, re.DOTALL | re.IGNORECASE)
    if not match:
        match = re.search(r'following text[:]?(.*?)summary:', prompt, re.DOTALL | re.IGNORECASE)
    
    if match:
        text = match.group(1).strip()
        
        # Analyze text length and adjust prompt accordingly
        word_count = len(text.split())
        
        if word_count > 1000:
            # For long texts, emphasize extractive summarization
            optimized = """Generate a concise, coherent summary of the following long text. 
Focus on the main points, key arguments, and crucial information.
Include only what's essential and maintain logical flow.

Text to summarize:
{}

Summary:""".format(text)
        elif word_count > 500:
            # For medium-length texts
            optimized = """Create a comprehensive summary of the following text. 
Capture the main ideas, significant details, and overall message.
Ensure the summary is coherent and well-structured.

Text to summarize:
{}

Summary:""".format(text)
        else:
            # For shorter texts
            optimized = """Provide a focused summary of this text that captures its essence.
Include the main points while maintaining the original meaning.

Text to summarize:
{}

Summary:""".format(text)
            
        return optimized
    
    # If we couldn't extract the text, return original prompt
    return prompt

def _optimize_sentiment_prompt(prompt):
    """
    Optimizes a sentiment analysis prompt.
    """
    # Extract the text for sentiment analysis
    match = re.search(r'text for sentiment analysis:(.*?)sentiment', prompt, re.DOTALL | re.IGNORECASE)
    if not match:
        match = re.search(r'following text[:]?(.*?)sentiment', prompt, re.DOTALL | re.IGNORECASE)
    
    if match:
        text = match.group(1).strip()
        
        # Create optimized prompt
        optimized = """Analyze the sentiment of this text as Positive, Negative, or Neutral.
Consider the overall emotional tone and provide a brief explanation for your analysis.
Pay attention to sentiment-bearing phrases, emotional language, and contextual cues.

Text:
{}

Sentiment analysis:""".format(text)
            
        return optimized
    
    # If we couldn't extract the text, return original prompt
    return prompt

def _optimize_keywords_prompt(prompt):
    """
    Optimizes a keyword extraction prompt.
    """
    # Extract the text for keyword extraction
    match = re.search(r'text for keyword extraction:(.*?)keywords:', prompt, re.DOTALL | re.IGNORECASE)
    if not match:
        match = re.search(r'following text[:]?(.*?)keywords:', prompt, re.DOTALL | re.IGNORECASE)
    
    if match:
        text = match.group(1).strip()
        
        # Determine if domain-specific keywords needed
        is_technical = _is_technical_text(text)
        
        if is_technical:
            # For technical content
            optimized = """Extract the most important technical terms, concepts, and key phrases from this text.
Identify domain-specific terminology and significant entities.
List keywords in order of importance, separated by commas.

Text:
{}

Keywords:""".format(text)
        else:
            # For general content
            optimized = """Extract the main topics, concepts, and key terms from this text.
Identify the most relevant and meaningful words and phrases.
List keywords in order of importance, separated by commas.

Text:
{}

Keywords:""".format(text)
            
        return optimized
    
    # If we couldn't extract the text, return original prompt
    return prompt

def _optimize_qa_prompt(prompt):
    """
    Optimizes a Q&A prompt.
    """
    # Extract the text and question
    text_match = re.search(r'context text:(.*?)question:', prompt, re.DOTALL | re.IGNORECASE)
    if not text_match:
        text_match = re.search(r'text:(.*?)question:', prompt, re.DOTALL | re.IGNORECASE)
    
    question_match = re.search(r'question:(.*?)answer:', prompt, re.DOTALL | re.IGNORECASE)
    
    if text_match and question_match:
        text = text_match.group(1).strip()
        question = question_match.group(1).strip()
        
        # Create optimized prompt
        optimized = """Answer the question precisely based on the provided context.
Use only information from the context and be specific.
If the context doesn't contain the answer, state that clearly.

Context:
{}

Question:
{}

Answer:""".format(text, question)
            
        return optimized
    
    # If we couldn't extract the text and question, return original prompt
    return prompt

def _apply_general_optimizations(prompt):
    """
    Applies general optimizations to any prompt.
    """
    # Ensure prompt ends with a clear indicator for completion
    if not prompt.rstrip().endswith(':'):
        prompt = re.sub(r'\s*$', '\n\n', prompt)
    
    # Remove excessive newlines or spaces
    prompt = re.sub(r'\n{3,}', '\n\n', prompt)
    prompt = re.sub(r' {2,}', ' ', prompt)
    
    return prompt

def _is_technical_text(text):
    """
    Checks if the text appears to be technical in nature.
    """
    # List of words that suggest technical content
    technical_indicators = [
        'algorithm', 'function', 'method', 'system', 'process', 'data',
        'analysis', 'research', 'study', 'experiment', 'result', 'conclusion',
        'theory', 'concept', 'framework', 'model', 'approach', 'technique',
        'technology', 'implementation', 'application', 'device', 'machine',
        'software', 'hardware', 'code', 'programming', 'development',
        'engineering', 'science', 'scientific', 'technical', 'protocol'
    ]
    
    # Count technical terms
    text_lower = text.lower()
    technical_count = sum(1 for term in technical_indicators if term in text_lower)
    
    # Estimate if text is technical based on density of technical terms
    word_count = len(text.split())
    technical_density = technical_count / max(1, word_count) * 100
    
    return technical_density > 1.5  # If more than 1.5% of words are technical
