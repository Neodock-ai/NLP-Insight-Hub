from models import llama, falcon, mistral, deepseek
from prompt_engineering import prompt_templates, prompt_optimizer

def get_model_instance(model_choice):
    """
    Returns the model instance based on the selected model.
    """
    if model_choice.lower() == "llama":
        return llama.LlamaModel()
    elif model_choice.lower() == "falcon":
        return falcon.FalconModel()
    elif model_choice.lower() == "mistral":
        return mistral.MistralModel()
    elif model_choice.lower() == "deepseek":
        return deepseek.DeepSeekModel()
    else:
        raise ValueError("Unknown model choice")

def get_summary(text, model="llama"):
    """
    Generates a summary for the input text.
    """
    model_instance = get_model_instance(model)
    prompt = prompt_templates.get_summary_prompt(text)
    optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
    return model_instance.generate_text(optimized_prompt)

def get_sentiment(text, model="llama"):
    """
    Analyzes sentiment for the input text.
    """
    model_instance = get_model_instance(model)
    prompt = prompt_templates.get_sentiment_prompt(text)
    optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
    return model_instance.generate_text(optimized_prompt)

def get_keywords(text, model="llama"):
    """
    Extracts keywords from the input text.
    """
    model_instance = get_model_instance(model)
    prompt = prompt_templates.get_keywords_prompt(text)
    optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
    return model_instance.generate_text(optimized_prompt)

def get_qa(text, question, model="llama"):
    """
    Answers a question based on the input text.
    """
    model_instance = get_model_instance(model)
    prompt = prompt_templates.get_qa_prompt(text, question)
    optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
    return model_instance.generate_text(optimized_prompt)
