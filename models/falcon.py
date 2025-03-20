from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class FalconModel:
    def __init__(self):
        # Placeholder: using 't5-small' as a stand-in for Falcon.
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    def generate_text(self, prompt):
        """
        Generates text using the Falcon model.
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
