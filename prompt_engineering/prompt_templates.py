def get_summary_prompt(text):
    """
    Returns an improved prompt template for summarizing text.
    Uses more specific instructions for more coherent and comprehensive summaries.
    """
    return f"""Summarize the following text with these requirements:
1. Create a coherent, well-structured summary that captures the main points
2. Maintain the logical flow of ideas from the original text
3. Include only the most important information and key details
4. Ensure the summary is self-contained and understandable without the original text
5. Use clear, concise language

Text to summarize:
{text}

Summary:"""

def get_sentiment_prompt(text):
    """
    Returns an improved prompt template for sentiment analysis.
    Provides more detailed instructions for better sentiment detection.
    """
    return f"""Analyze the sentiment of the following text with these requirements:
1. Classify the overall sentiment as Positive, Negative, or Neutral
2. Consider the emotional tone, opinion, and attitude expressed in the text
3. Pay attention to nuanced language, including idioms and expressions
4. Account for mixed sentiments if present, but provide a definitive final classification
5. Identify key sentiment-bearing phrases that support your classification

Text for sentiment analysis:
{text}

Sentiment (provide only the classification word followed by a brief explanation):"""

def get_keywords_prompt(text):
    """
    Returns an improved prompt template for extracting keywords.
    """
    return f"""Extract the main keywords and key concepts from the following text:
1. Identify the most important topics, entities, and concepts
2. Focus on meaningful terms that represent core ideas
3. Include relevant technical terms, proper nouns, and domain-specific vocabulary
4. List keywords in order of importance
5. Return only the keywords separated by commas, without numbering or explanation

Text for keyword extraction:
{text}

Keywords:"""

def get_qa_prompt(text, question):
    """
    Returns an improved prompt template for Q&A.
    """
    return f"""Answer the question based on the provided text:
1. Provide a direct, clear answer to the question
2. Use only information present in the text
3. If the answer isn't in the text, state that clearly
4. Include relevant context to support your answer
5. Be concise but thorough

Context text:
{text}

Question:
{question}

Answer:"""
