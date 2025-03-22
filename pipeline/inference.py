# inference.py
# ----------------------------------------------------------------------------------
# An updated version of your inference pipeline that includes an "OpenAI GPT" model
# option, chunk-based logic, and user-supplied API keys.
# ----------------------------------------------------------------------------------

import logging
from models import llama, falcon, mistral, deepseek

# IMPORTANT: add your new openai_gpt model here.
# E.g., from models.openai_gpt import OpenAIGPTModel
# Make sure you have created that file with the chunk-based approach.
try:
    from models.openai_gpt import OpenAIGPTModel
except ImportError:
    # If you haven't created openai_gpt.py yet, remove or handle gracefully
    OpenAIGPTModel = None

from prompt_engineering import prompt_templates, prompt_optimizer

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
        # ADDED: If the user chooses "openai gpt," return our chunk-based GPT class
        if model_choice.lower() in ["openai gpt", "openai_gpt"]:
            if OpenAIGPTModel is None:
                raise ValueError("OpenAIGPTModel is not available. Make sure 'openai_gpt.py' is created.")
            return OpenAIGPTModel()

        elif model_choice.lower() == "llama":
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

# ----------------------------------------------------------------------
# NEW UTILITY TO SET OPENAI API KEY ON THE MODEL, IF SUPPORTED
# ----------------------------------------------------------------------
def set_openai_api_key(model_instance, api_key: str):
    """
    If the model is an OpenAIGPTModel, call its set_api_key(api_key).
    Otherwise, do nothing.
    """
    if hasattr(model_instance, "set_api_key"):
        # e.g. model_instance is an instance of OpenAIGPTModel
        model_instance.set_api_key(api_key)

# ----------------------------------------------------------------------
# SUMMARIZATION
# ----------------------------------------------------------------------
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
    
    # Base prompt template
    prompt = prompt_templates.get_summary_prompt(text)
    # Insert the length guidance
    prompt = prompt.replace("Text to summarize:", f"Text to summarize:\n{prompt_modifier}")
    
    # Optimize
    optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
    
    # Get model instance
    model_instance = get_model_instance(model)

    # If this is our OpenAI GPT model, do the chunk-based approach
    if isinstance(model_instance, OpenAIGPTModel):
        # We ignore the old T5 approach and call the chunk-based summarizer
        return model_instance.summarize(text)
    
    # Otherwise, do your old T5-based approach
    try:
        summary = model_instance.generate_text(optimized_prompt)
        
        # If the summary is too short or low quality, try a fallback
        if len(summary.split()) < 15 or "error" in summary.lower():
            logger.warning(f"Low quality summary from {model}, trying fallback model")
            try:
                fallback_model = "mistral" if model.lower() != "mistral" else "deepseek"
                fm_instance = get_model_instance(fallback_model)
                fallback_summary = fm_instance.generate_text(optimized_prompt)
                
                # Use the better summary
                if len(fallback_summary.split()) > len(summary.split()) * 1.5:
                    logger.info(f"Using fallback summary from {fallback_model}")
                    return fallback_summary
            except Exception as e:
                logger.error(f"Error with fallback summary: {str(e)}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

# ----------------------------------------------------------------------
# SENTIMENT
# ----------------------------------------------------------------------
def get_sentiment(text, model="llama"):
    """
    Analyzes sentiment for the input text with improved accuracy.
    
    Args:
        text (str): The text to analyze
        model (str): The model to use for sentiment analysis
        
    Returns:
        dict OR str: Sentiment results
    """
    logger.info(f"Analyzing sentiment using {model} model")
    
    prompt = prompt_templates.get_sentiment_prompt(text)
    optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
    
    model_instance = get_model_instance(model)

    # If OpenAIGPTModel, do chunk-based GPT sentiment
    if isinstance(model_instance, OpenAIGPTModel):
        return model_instance.sentiment(text)
    
    # Otherwise, old approach
    try:
        sentiment_result = model_instance.generate_text(optimized_prompt)
        
        # If it gives uncertain result, fallback to another model
        if any(term in str(sentiment_result).lower() for term in ["unclear", "uncertain", "ambiguous", "mixed"]):
            logger.info(f"Uncertain sentiment from {model}, trying another model")
            try:
                fallback_model = "deepseek" if model.lower() != "deepseek" else "falcon"
                fb_instance = get_model_instance(fallback_model)
                fallback_result = fb_instance.generate_text(optimized_prompt)
                
                # Combine if both are dict with "scores"
                if isinstance(sentiment_result, dict) and isinstance(fallback_result, dict):
                    if "scores" in sentiment_result and "scores" in fallback_result:
                        combined_scores = {}
                        for key in sentiment_result["scores"]:
                            combined_scores[key] = (
                                sentiment_result["scores"][key] * 0.6
                                + fallback_result["scores"].get(key, 0) * 0.4
                            )
                        # Highest
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

# ----------------------------------------------------------------------
# KEYWORDS
# ----------------------------------------------------------------------
def get_keywords(text, model="llama"):
    """
    Extracts keywords from the input text with improved relevance.
    
    Args:
        text (str): The text
        model (str): The model to use
        
    Returns:
        list OR str: Keywords
    """
    logger.info(f"Extracting keywords using {model} model")
    
    prompt = prompt_templates.get_keywords_prompt(text)
    optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
    
    model_instance = get_model_instance(model)

    # If OpenAIGPTModel, chunk-based keywords
    if isinstance(model_instance, OpenAIGPTModel):
        return model_instance.keywords(text)
    
    # Otherwise old approach
    try:
        keywords = model_instance.generate_text(optimized_prompt)
        
        # Attempt ensemble with second model
        try:
            second_model = "falcon" if model.lower() != "falcon" else "mistral"
            second_instance = get_model_instance(second_model)
            second_keywords = second_instance.generate_text(optimized_prompt)
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

# ----------------------------------------------------------------------
# Q&A
# ----------------------------------------------------------------------
def get_qa(text, question, model="llama"):
    """
    Answers a question based on the input text with improved accuracy.
    
    Args:
        text (str): The context text
        question (str): The question
        model (str): The model name
        
    Returns:
        str: The answer
    """
    logger.info(f"Answering question using {model} model")
    
    prompt = prompt_templates.get_qa_prompt(text, question)
    optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
    
    model_instance = get_model_instance(model)

    # If OpenAIGPTModel, chunk-based QA
    if isinstance(model_instance, OpenAIGPTModel):
        return model_instance.qa(text, question)
    
    # Otherwise old approach
    try:
        answer = model_instance.generate_text(optimized_prompt)
        # Check for "don't know" pattern, fallback to second model
        triggers = ["don't know", "cannot determine", "not stated", "unclear", "no information"]
        if any(t in answer.lower() for t in triggers):
            logger.info(f"Trying second model for possibly better QA.")
            try:
                second_model = "deepseek" if model.lower() != "deepseek" else "llama"
                second_instance = get_model_instance(second_model)
                second_answer = second_instance.generate_text(optimized_prompt)
                if not any(t in second_answer.lower() for t in triggers):
                    logger.info(f"Using answer from {second_model}")
                    return second_answer
            except Exception as e:
                logger.error(f"Error with second Q&A attempt: {str(e)}")
        return answer
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return f"Error answering question: {str(e)}"

# ----------------------------------------------------------------------
# INTERNAL KEYWORD COMBINING
# ----------------------------------------------------------------------
def _combine_keywords(keywords1, keywords2):
    """
    Combines keyword results from multiple models.
    
    Args:
        keywords1: First set of keywords (string or list of tuples)
        keywords2: Second set of keywords (string or list of tuples)
        
    Returns:
        list: Combined list of (keyword, weight) tuples
    """
    kw_list1 = _normalize_keywords(keywords1)
    kw_list2 = _normalize_keywords(keywords2)
    
    combined = {}
    
    # Add first set
    for kw, weight in kw_list1:
        combined[kw.lower()] = (kw, weight)
    
    # Add second set
    for kw, weight in kw_list2:
        low = kw.lower()
        if low in combined:
            existing_kw, existing_weight = combined[low]
            # Keep better cased version
            better_kw = kw if kw and kw[0].isupper() and (not existing_kw or not existing_kw[0].isupper()) else existing_kw
            avg_weight = (existing_weight + weight) / 2
            combined[low] = (better_kw, avg_weight)
        else:
            combined[low] = (kw, weight)
    
    # Sort by weight descending
    result = list(combined.values())
    result.sort(key=lambda x: x[1], reverse=True)
    return result

def _normalize_keywords(keywords):
    """
    Normalizes keywords to a list of (keyword, weight) tuples.
    We keep your existing logic for converting from strings/lists/dicts.
    """
    if isinstance(keywords, str):
        # string of comma-separated keywords
        kw_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        return [(kw, 10.0 - 0.1 * i) for i, kw in enumerate(kw_list)]
    
    elif isinstance(keywords, list):
        if all(isinstance(k, tuple) and len(k) == 2 for k in keywords):
            return keywords
        else:
            # list of strings
            return [(kw, 10.0 - 0.1 * i) for i, kw in enumerate(keywords)]
    
    elif isinstance(keywords, dict):
        return [(k, v) for k, v in keywords.items()]
    
    # Default
    return []

