from models import llama, falcon, mistral, deepseek
from prompt_engineering import prompt_templates, prompt_optimizer
import logging

# Configure module logger
logger = logging.getLogger(__name__)

def get_model_instance(model_choice):
    """
    Returns the model instance based on the selected model.
    
    Args:
        model_choice (str): The name of the model to use
        
    Returns:
        object: The model instance
        
    Raises:
        ValueError: If the model choice is not recognized
    """
    try:
        if model_choice.lower() == "llama":
            return llama.LlamaModel()
        elif model_choice.lower() == "falcon":
            return falcon.FalconModel()
        elif model_choice.lower() == "mistral":
            return mistral.MistralModel()
        elif model_choice.lower() == "deepseek":
            return deepseek.DeepSeekModel()
        else:
            raise ValueError(f"Unknown model choice: {model_choice}")
    except Exception as e:
        logger.error(f"Error initializing model {model_choice}: {str(e)}")
        # Fallback to Llama if available
        try:
            logger.info("Attempting to fall back to Llama model")
            return llama.LlamaModel()
        except Exception:
            raise ValueError(f"Failed to initialize any model: {str(e)}")

def get_summary(text, model="llama", length=3):
    """
    Generates a summary for the input text with improved quality.
    
    Args:
        text (str): The text to summarize
        model (str): The model to use for summarization
        length (int): Desired summary length (1=very brief, 5=detailed)
        
    Returns:
        str: The generated summary
    """
    logger.info(f"Generating summary using {model} model, length={length}")
    
    # Adjust prompt based on desired length
    if length <= 2:
        prompt_modifier = "Create a brief and concise summary focusing only on the most essential points."
    elif length >= 4:
        prompt_modifier = "Create a comprehensive and detailed summary covering all important aspects."
    else:
        prompt_modifier = "Create a well-balanced summary covering the main points."
    
    # Get the base prompt template
    prompt = prompt_templates.get_summary_prompt(text)
    
    # Add length guidance
    prompt = prompt.replace("Text to summarize:", f"Text to summarize:\n{prompt_modifier}")
    
    # Optimize the prompt
    optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
    
    # Get model instance and generate summary
    try:
        model_instance = get_model_instance(model)
        summary = model_instance.generate_text(optimized_prompt)
        
        # If the summary is too short or low quality, try a different model
        if len(summary.split()) < 15 or "error" in summary.lower():
            logger.warning(f"Low quality summary from {model}, trying fallback model")
            try:
                fallback_model = "mistral" if model.lower() != "mistral" else "deepseek"
                model_instance = get_model_instance(fallback_model)
                fallback_summary = model_instance.generate_text(optimized_prompt)
                
                # Use the better summary (simple length heuristic)
                if len(fallback_summary.split()) > len(summary.split()) * 1.5:
                    logger.info(f"Using fallback summary from {fallback_model}")
                    return fallback_summary
                
            except Exception as e:
                logger.error(f"Error with fallback summary: {str(e)}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

def get_sentiment(text, model="llama"):
    """
    Analyzes sentiment for the input text with improved accuracy.
    
    Args:
        text (str): The text to analyze
        model (str): The model to use for sentiment analysis
        
    Returns:
        dict: Sentiment analysis results including sentiment classification and scores
    """
    logger.info(f"Analyzing sentiment using {model} model")
    
    # Get the base prompt template
    prompt = prompt_templates.get_sentiment_prompt(text)
    
    # Optimize the prompt
    optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
    
    # Get model instance and analyze sentiment
    try:
        model_instance = get_model_instance(model)
        sentiment_result = model_instance.generate_text(optimized_prompt)
        
        # If the first model gives a vague or uncertain result, try another
        if any(term in str(sentiment_result).lower() for term in ["unclear", "uncertain", "ambiguous", "mixed"]):
            logger.info(f"Uncertain sentiment from {model}, trying another model")
            try:
                # Try a different model
                fallback_model = "deepseek" if model.lower() != "deepseek" else "falcon"
                fallback_instance = get_model_instance(fallback_model)
                fallback_result = fallback_instance.generate_text(optimized_prompt)
                
                # Combine results for better accuracy
                if isinstance(sentiment_result, dict) and isinstance(fallback_result, dict):
                    # If both are dictionaries with scores, use weighted average
                    if "scores" in sentiment_result and "scores" in fallback_result:
                        combined_scores = {}
                        
                        for key in sentiment_result["scores"]:
                            combined_scores[key] = (
                                sentiment_result["scores"][key] * 0.6 + 
                                fallback_result["scores"].get(key, 0) * 0.4
                            )
                        
                        # Determine final sentiment
                        final_sentiment = max(combined_scores, key=combined_scores.get)
                        
                        sentiment_result = {
                            "sentiment": final_sentiment.capitalize(),
                            "scores": combined_scores,
                            "ensemble": True
                        }
                
            except Exception as e:
                logger.error(f"Error with fallback sentiment: {str(e)}")
        
        return sentiment_result
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return {"sentiment": "Neutral", "error": str(e)}

def get_keywords(text, model="llama"):
    """
    Extracts keywords from the input text with improved relevance.
    
    Args:
        text (str): The text to extract keywords from
        model (str): The model to use for keyword extraction
        
    Returns:
        list: List of keyword tuples (keyword, relevance_score)
    """
    logger.info(f"Extracting keywords using {model} model")
    
    # Get the base prompt template
    prompt = prompt_templates.get_keywords_prompt(text)
    
    # Optimize the prompt
    optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
    
    # Get model instance and extract keywords
    try:
        model_instance = get_model_instance(model)
        keywords = model_instance.generate_text(optimized_prompt)
        
        # For keywords, it's beneficial to combine results from multiple models
        # for more comprehensive coverage
        try:
            # Try an additional model
            second_model = "falcon" if model.lower() != "falcon" else "mistral"
            second_instance = get_model_instance(second_model)
            second_keywords = second_instance.generate_text(optimized_prompt)
            
            # Combine the keyword sets
            combined_keywords = _combine_keywords(keywords, second_keywords)
            if combined_keywords:
                logger.info(f"Using ensemble keywords from {model} and {second_model}")
                return combined_keywords
                
        except Exception as e:
            logger.error(f"Error with secondary keywords: {str(e)}")
        
        return keywords
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return [("Error", 10.0)]

def get_qa(text, question, model="llama"):
    """
    Answers a question based on the input text with improved accuracy.
    
    Args:
        text (str): The context text
        question (str): The question to answer
        model (str): The model to use for Q&A
        
    Returns:
        str: The answer to the question
    """
    logger.info(f"Answering question using {model} model")
    
    # Get the base prompt template
    prompt = prompt_templates.get_qa_prompt(text, question)
    
    # Optimize the prompt
    optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
    
    # Get model instance and generate answer
    try:
        model_instance = get_model_instance(model)
        answer = model_instance.generate_text(optimized_prompt)
        
        # Check for "I don't know" type responses that might benefit from a different model
        if any(phrase in answer.lower() for phrase in ["don't know", "cannot determine", "not stated", "unclear", "no information"]):
            # The question might be answerable by a different model
            logger.info(f"Checking answer with second model")
            try:
                second_model = "deepseek" if model.lower() != "deepseek" else "llama"
                second_instance = get_model_instance(second_model)
                second_answer = second_instance.generate_text(optimized_prompt)
                
                # If the second model gives a more definitive answer, use it
                if not any(phrase in second_answer.lower() for phrase in ["don't know", "cannot determine", "not stated", "unclear", "no information"]):
                    logger.info(f"Using answer from {second_model}")
                    return second_answer
                
            except Exception as e:
                logger.error(f"Error with second Q&A attempt: {str(e)}")
        
        return answer
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return f"Error answering question: {str(e)}"

def _combine_keywords(keywords1, keywords2):
    """
    Combines keyword results from multiple models.
    
    Args:
        keywords1: First set of keywords (string or list of tuples)
        keywords2: Second set of keywords (string or list of tuples)
        
    Returns:
        list: Combined list of (keyword, weight) tuples
    """
    # Convert both to lists of tuples (keyword, weight)
    kw_list1 = _normalize_keywords(keywords1)
    kw_list2 = _normalize_keywords(keywords2)
    
    # Combine the lists
    combined = {}
    
    # Add first set
    for kw, weight in kw_list1:
        combined[kw.lower()] = (kw, weight)
    
    # Add second set, averaging weights for duplicates
    for kw, weight in kw_list2:
        kw_lower = kw.lower()
        if kw_lower in combined:
            existing_kw, existing_weight = combined[kw_lower]
            # Use the better-cased version
            better_kw = kw if kw[0].isupper() and not existing_kw[0].isupper() else existing_kw
            # Average the weights
            avg_weight = (existing_weight + weight) / 2
            combined[kw_lower] = (better_kw, avg_weight)
        else:
            combined[kw_lower] = (kw, weight)
    
    # Convert back to list and sort by weight
    result = [item for _, item in combined.items()]
    result.sort(key=lambda x: x[1], reverse=True)
    
    return result

def _normalize_keywords(keywords):
    """
    Normalizes keywords to a list of (keyword, weight) tuples.
    
    Args:
        keywords: Keywords (string or list)
        
    Returns:
        list: List of (keyword, weight) tuples
    """
    if isinstance(keywords, str):
        # Convert comma-separated string to list
        kw_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        # Assign decreasing weights
        return [(kw, 10.0 - 0.1 * i) for i, kw in enumerate(kw_list)]
    
    elif isinstance(keywords, list):
        if all(isinstance(k, tuple) and len(k) == 2 for k in keywords):
            # Already in the right format
            return keywords
        else:
            # List of strings
            return [(kw, 10.0 - 0.1 * i) for i, kw in enumerate(keywords)]
    
    elif isinstance(keywords, dict):
        # Convert dict to list of tuples
        return [(k, v) for k, v in keywords.items()]
    
    # Default empty list
    return []
