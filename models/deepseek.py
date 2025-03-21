from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class DeepSeekModel:
    def __init__(self):
        # Placeholder: using 't5-small' as a stand-in for DeepSeek.
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        self.name = "DeepSeek"
    
    def generate_text(self, prompt):
        """
        Generates text using the DeepSeek model with advanced task-specific handling.
        """
        task_type = self._determine_task_type(prompt)
        
        if task_type == "summary":
            # Extract text to summarize
            text_to_process = prompt.split("Summarize the following text:\n\n")[1].split("\n\nSummary:")[0]
            # DeepSeek uses a specialized summary format
            input_prompt = f"Generate a comprehensive summary of: {text_to_process}"
        
        elif task_type == "sentiment":
            # Extract text for sentiment analysis
            text_to_process = prompt.split("Analyze the sentiment of the following text")[1].split("Sentiment:")[0]
            input_prompt = f"Perform sentiment analysis and classify as Positive, Negative, or Neutral: {text_to_process}"
        
        elif task_type == "keywords":
            # Extract text for keyword extraction
            text_to_process = prompt.split("Extract the main keywords from the following text")[1].split("Keywords:")[0]
            input_prompt = f"Extract the most important keywords and concepts: {text_to_process}"
        
        elif task_type == "qa":
            # Extract text and question for Q&A
            parts = prompt.split("Text: ")[1]
            text = parts.split("\n\nQuestion: ")[0]
            question = parts.split("\n\nQuestion: ")[1].split("\n\nAnswer:")[0]
            input_prompt = f"Using only the provided context, answer the following question.\nContext: {text}\nQuestion: {question}\nAnswer:"
        
        else:
            # Default handling
            input_prompt = prompt
        
        # Process with the model (with different parameters for DeepSeek)
        inputs = self.tokenizer.encode(input_prompt, return_tensors="pt", max_length=512, truncation=True)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            inputs = inputs.to(device)
            self.model = self.model.to(device)
        
        # DeepSeek's generation parameters (would be different from other models)
        outputs = self.model.generate(
            inputs, 
            max_length=200,  # Longer max length than other models
            num_beams=6,     # More beams for better quality
            temperature=0.6, # Lower temperature for more focused outputs
            top_p=0.95,      # Different top_p
            top_k=50,        # Add top_k
            do_sample=True,
            early_stopping=True
        )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Task-specific post-processing with DeepSeek's behavior
        if task_type == "sentiment":
            # DeepSeek has high confidence classification
            if "positive" in result.lower() or "good" in result.lower():
                return "Positive"
            elif "negative" in result.lower() or "bad" in result.lower():
                return "Negative"
            else:
                return "Neutral"
        
        elif task_type == "keywords":
            # DeepSeek's keyword extraction should focus on important concepts
            try:
                # Attempt to parse comma-separated keywords
                keywords = [k.strip() for k in result.split(',')]
                # Filter out any too-short words
                keywords = [k for k in keywords if len(k) > 2]
                
                # If we have enough keywords, return them
                if len(keywords) >= 5:
                    return ", ".join(keywords[:12])  # DeepSeek provides more keywords
                
                # Otherwise, fall back to a more basic extraction
                words = set()
                for word in text_to_process.split():
                    word = word.strip(".,;:\"'()[]{}")
                    if len(word) > 5:
                        words.add(word)
                
                return ", ".join(list(words)[:10])
                
            except Exception:
                # Fallback for any parsing issues
                return result
        
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
