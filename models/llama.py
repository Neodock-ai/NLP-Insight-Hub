from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LlamaModel:
    def __init__(self):
        # For demonstration, we use 't5-small' as a stand-in for Llama
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        self.name = "Llama"
    
    def generate_text(self, prompt):
        """
        Generates text using the Llama model with task-specific handling.
        """
        task_type = self._determine_task_type(prompt)
        
        if task_type == "summary":
            # Extract the text to summarize
            text_to_process = prompt.split("Summarize the following text:\n\n")[1].split("\n\nSummary:")[0]
            input_prompt = f"summarize: {text_to_process}"
        
        elif task_type == "sentiment":
            # Extract the text for sentiment analysis
            text_to_process = prompt.split("Analyze the sentiment of the following text")[1].split("Sentiment:")[0]
            input_prompt = f"classify sentiment: {text_to_process}"
        
        elif task_type == "keywords":
            # Extract the text for keyword extraction
            text_to_process = prompt.split("Extract the main keywords from the following text")[1].split("Keywords:")[0]
            input_prompt = f"extract keywords: {text_to_process}"
        
        elif task_type == "qa":
            # Extract the text and question for Q&A
            parts = prompt.split("Text: ")[1]
            text = parts.split("\n\nQuestion: ")[0]
            question = parts.split("\n\nQuestion: ")[1].split("\n\nAnswer:")[0]
            input_prompt = f"answer question: {question} context: {text}"
        
        else:
            # Default handling
            input_prompt = prompt
        
        # Process with the model
        inputs = self.tokenizer.encode(input_prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Add some randomness to ensure unique responses
        if torch.cuda.is_available():
            device = torch.device("cuda")
            inputs = inputs.to(device)
            self.model = self.model.to(device)
        
        outputs = self.model.generate(
            inputs, 
            max_length=150, 
            num_beams=4, 
            temperature=0.7,  # Add temperature for some randomness
            do_sample=True,   # Enable sampling
            early_stopping=True
        )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process based on task type
        if task_type == "sentiment":
            # Map t5 output to sentiment categories
            if "positive" in result.lower():
                return "Positive"
            elif "negative" in result.lower():
                return "Negative"
            else:
                return "Neutral"
        
        elif task_type == "keywords":
            # If not in a good format, extract words and format as comma-separated
            words = [word for word in result.split() if len(word) > 3]
            return ", ".join(words[:10])  # Limit to 10 keywords
            
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
