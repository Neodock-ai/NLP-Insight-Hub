from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random

class MistralModel:
    def __init__(self):
        # Placeholder: using 't5-small' as a stand-in for Mistral
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        self.name = "Mistral"
    
    def generate_text(self, prompt):
        """
        Generates text using the Mistral model with task-specific handling.
        """
        task_type = self._determine_task_type(prompt)
        
        if task_type == "summary":
            # Extract the text to summarize
            text_to_process = prompt.split("Summarize the following text:\n\n")[1].split("\n\nSummary:")[0]
            # Mistral uses a different summary prompt format
            input_prompt = f"Create a concise summary: {text_to_process}"
        
        elif task_type == "sentiment":
            # Extract the text for sentiment analysis
            text_to_process = prompt.split("Analyze the sentiment of the following text")[1].split("Sentiment:")[0]
            input_prompt = f"Sentiment analysis: {text_to_process}"
        
        elif task_type == "keywords":
            # Extract the text for keyword extraction
            text_to_process = prompt.split("Extract the main keywords from the following text")[1].split("Keywords:")[0]
            input_prompt = f"Important keywords: {text_to_process}"
        
        elif task_type == "qa":
            # Extract the text and question for Q&A
            parts = prompt.split("Text: ")[1]
            text = parts.split("\n\nQuestion: ")[0]
            question = parts.split("\n\nQuestion: ")[1].split("\n\nAnswer:")[0]
            input_prompt = f"Question: {question} Context: {text}"
        
        else:
            # Default handling
            input_prompt = prompt
        
        # Process with the model (with different parameters from LlamaModel)
        inputs = self.tokenizer.encode(input_prompt, return_tensors="pt", max_length=512, truncation=True)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            inputs = inputs.to(device)
            self.model = self.model.to(device)
        
        outputs = self.model.generate(
            inputs, 
            max_length=150, 
            num_beams=5,  # Different from Llama
            temperature=0.8,  # Different temperature
            top_p=0.9,  # Add top_p sampling
            do_sample=True,
            early_stopping=True
        )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Task-specific post-processing with different behavior from LlamaModel
        if task_type == "sentiment":
            sentiments = ["Positive", "Negative", "Neutral"]
            # For demo purposes, this is more decisive than Llama
            if "positive" in result.lower():
                return "Positive"
            elif "negative" in result.lower():
                return "Negative"
            else:
                return "Neutral"
        
        elif task_type == "keywords":
            # Different approach from Llama - attempting to find noun phrases
            words = []
            for word in result.split():
                if len(word) > 4:  # Slightly different criteria
                    words.append(word.strip(".,;:\"'()[]{}"))
            
            # Ensure we have at least some keywords
            if len(words) < 3:
                words = [w for w in text_to_process.split() if len(w) > 5][:8]
                
            return ", ".join(words[:8])  # Fewer keywords than Llama
        
        return result
    
    def _determine_task_type(self, prompt):
        """
        Determines what type of NLP task is being requested.
        """
        if "Summarize the following" in prompt:
            return "summary"
        elif "Analyze the sentiment" in prompt:
            return "sentiment"
        elif "Extract the main keywords" in prompt:
            return "keywords"
        elif "Based on the following text, answer the question" in prompt:
            return "qa"
        else:
            return "general"
