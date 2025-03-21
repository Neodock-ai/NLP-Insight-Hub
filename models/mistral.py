from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random
import re
from collections import Counter
import math

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
            input_prompt = f"Create a focused, concise summary: {text_to_process}"
        
        elif task_type == "sentiment":
            # Extract the text for sentiment analysis
            text_to_process = prompt.split("Analyze the sentiment of the following text")[1].split("Sentiment:")[0]
            input_prompt = f"Sentiment analysis of the following text, classify as Positive, Negative, or Neutral: {text_to_process}"
        
        elif task_type == "keywords":
            # Extract the text for keyword extraction
            text_to_process = prompt.split("Extract the main keywords from the following text")[1].split("Keywords:")[0]
            input_prompt = f"Extract most relevant keywords and key phrases: {text_to_process}"
        
        elif task_type == "qa":
            # Extract the text and question for Q&A
            parts = prompt.split("Text: ")[1]
            text = parts.split("\n\nQuestion: ")[0]
            question = parts.split("\n\nQuestion: ")[1].split("\n\nAnswer:")[0]
            input_prompt = f"Answer precisely based on the following text\nText: {text}\nQuestion: {question}\nAnswer:"
        
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
        
        # Task-specific post-processing
        if task_type == "sentiment":
            sentiment_result = self._dynamic_sentiment_analysis(text_to_process, result)
            return sentiment_result
        
        elif task_type == "keywords":
            keyword_result = self._enhanced_keyword_extraction(text_to_process, result)
            return keyword_result
        
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
    
    def _dynamic_sentiment_analysis(self, text, raw_result):
        """
        Enhanced sentiment analysis with TF-IDF inspired weighting.
        """
        # Process raw result to identify sentiment
        result_lower = raw_result.lower()
        
        # Determine base sentiment
        if any(pos in result_lower for pos in ["positive", "good", "great", "excellent"]):
            base_sentiment = "Positive"
        elif any(neg in result_lower for neg in ["negative", "bad", "poor", "terrible"]):
            base_sentiment = "Negative"
        else:
            base_sentiment = "Neutral"
        
        # Initialize scores
        scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        
        # Define sentiment lexicons
        positive_terms = {
            "good": 1.0, "great": 1.2, "excellent": 1.5, "amazing": 1.4, 
            "impressive": 1.1, "happy": 1.0, "love": 1.3, "best": 1.2,
            "wonderful": 1.3, "fantastic": 1.4, "perfect": 1.5, "enjoy": 1.0,
            "superior": 1.2, "outstanding": 1.4, "favorite": 1.1, "recommended": 1.0,
            "positive": 1.0, "pleased": 0.9, "glad": 0.9, "satisfied": 0.9,
            "beneficial": 0.8, "successful": 0.9, "excellent": 1.3
        }
        
        negative_terms = {
            "bad": 1.0, "poor": 1.1, "terrible": 1.5, "awful": 1.4,
            "horrible": 1.5, "disappointing": 1.2, "worst": 1.5, "hate": 1.4,
            "dislike": 1.0, "unfortunate": 0.9, "mediocre": 0.8, "negative": 1.0,
            "problem": 0.7, "difficult": 0.7, "hard": 0.6, "annoying": 1.1,
            "frustrating": 1.2, "failure": 1.3, "inferior": 1.1, "useless": 1.3,
            "waste": 1.2, "boring": 0.9, "slow": 0.7, "expensive": 0.7
        }
        
        neutral_terms = {
            "ok": 0.5, "okay": 0.5, "neutral": 0.8, "average": 0.6,
            "acceptable": 0.5, "fair": 0.5, "reasonable": 0.5, "moderate": 0.6,
            "typical": 0.5, "normal": 0.5, "standard": 0.5, "adequate": 0.5,
            "neither": 0.7, "mixed": 0.6, "balanced": 0.6, "common": 0.4
        }
        
        # Prepare text for analysis
        text_words = re.findall(r'\b\w+\b', text.lower())
        
        # Count term occurrences
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        # Apply a TF-IDF inspired weighting
        for word in text_words:
            if word in positive_terms:
                positive_count += positive_terms[word]
            elif word in negative_terms:
                negative_count += negative_terms[word]
            elif word in neutral_terms:
                neutral_count += neutral_terms[word]
        
        # Factor in negation
        text_lower = text.lower()
        negations = ["not", "no", "never", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't"]
        
        for negation in negations:
            # Look for patterns like "not good", "no excellent", etc.
            for pos_term in positive_terms:
                negation_pattern = f"{negation} {pos_term}"
                if negation_pattern in text_lower:
                    # Convert positive to negative
                    positive_count -= positive_terms[pos_term] * 0.8
                    negative_count += positive_terms[pos_term] * 0.7
            
            # Similarly for negative terms
            for neg_term in negative_terms:
                negation_pattern = f"{negation} {neg_term}"
                if negation_pattern in text_lower:
                    # Convert negative to positive
                    negative_count -= negative_terms[neg_term] * 0.8
                    positive_count += negative_terms[neg_term] * 0.6
        
        # Ensure counts are not negative
        positive_count = max(0, positive_count)
        negative_count = max(0, negative_count)
        
        # Calculate total to normalize
        total_count = positive_count + negative_count + neutral_count
        
        if total_count > 0:
            # Normalize to probabilities
            scores["positive"] = positive_count / total_count
            scores["negative"] = negative_count / total_count
            scores["neutral"] = neutral_count / total_count
        else:
            # Default to base sentiment
            if base_sentiment == "Positive":
                scores["positive"] = 0.7
                scores["neutral"] = 0.2
                scores["negative"] = 0.1
            elif base_sentiment == "Negative":
                scores["positive"] = 0.1
                scores["neutral"] = 0.2
                scores["negative"] = 0.7
            else:
                scores["positive"] = 0.25
                scores["neutral"] = 0.5
                scores["negative"] = 0.25
        
        # Add randomness for more natural feel
        for key in scores:
            scores[key] += random.uniform(-0.05, 0.05)
            scores[key] = max(0, min(1, scores[key]))  # Clamp between 0 and 1
        
        # Renormalize after randomness
        total = sum(scores.values())
        for key in scores:
            scores[key] /= total
        
        # Determine final sentiment based on scores
        if scores["positive"] > scores["negative"] and scores["positive"] > scores["neutral"]:
            sentiment = "Positive"
        elif scores["negative"] > scores["positive"] and scores["negative"] > scores["neutral"]:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return {
            "sentiment": sentiment,
            "scores": scores
        }
    
    def _enhanced_keyword_extraction(self, text, result):
        """
        Enhanced keyword extraction using TF-IDF principles.
        """
        # Clean and normalize text
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        # Remove stopwords
        stopwords = self._get_stopwords()
        filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Extract ngrams (phrases of 2-3 words)
        ngrams = []
        for i in range(len(words) - 1):
            if words[i] not in stopwords and words[i+1] not in stopwords:
                ngrams.append(f"{words[i]} {words[i+1]}")
        
        for i in range(len(words) - 2):
            if words[i] not in stopwords and words[i+2] not in stopwords:
                ngrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        ngram_counts = Counter(ngrams)
        
        # Try to use keywords from the result first
        keywords = []
        if "," in result:
            keywords = [k.strip() for k in result.split(",") if len(k.strip()) > 2]
        
        # If we don't have enough keywords from the result, use our extracted ones
        keyword_weights = []
        
        if len(keywords) < 5:
            # Combine single words and ngrams
            all_candidates = []
            
            # Add high-frequency single words
            for word, count in word_counts.most_common(20):
                if len(word) > 3:  # Longer words might be more meaningful
                    # Weight based on frequency and word length
                    weight = count * (1 + min(0.5, len(word) * 0.05))
                    all_candidates.append((word, weight))
            
            # Add ngrams with higher weight
            for ngram, count in ngram_counts.most_common(15):
                # Ngrams usually more meaningful
                weight = count * 1.5
                all_candidates.append((ngram, weight))
            
            # Sort by weight
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Take the top candidates
            keywords = [candidate[0] for candidate in all_candidates[:15]]
        
        # Convert to proper case
        proper_case_keywords = []
        for keyword in keywords:
            # Check if keyword exists in original text with proper case
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            matches = pattern.findall(text)
            
            if matches:
                # Use the most common case
                case_counts = Counter(matches)
                proper_case_keywords.append(case_counts.most_common(1)[0][0])
            else:
                proper_case_keywords.append(keyword)
        
        # Assign weights to keywords
        for i, keyword in enumerate(proper_case_keywords):
            # Base weight with variation
            base_weight = 10.2 - (i * 0.05)  # Decreasing weight based on position
            
            # Additional weighting factors
            frequency_factor = 0
            if keyword.lower() in word_counts:
                # Calculate term frequency normalized by document length
                tf = word_counts[keyword.lower()] / len(words)
                frequency_factor = min(0.3, tf * 10)  # Cap the frequency factor
            
            # Length factor - longer terms might be more specific
            length_factor = min(0.2, len(keyword) * 0.01)
            
            # Capitalization factor - proper nouns might be more important
            cap_factor = 0.1 if keyword[0].isupper() else 0
            
            # Calculate final weight with some randomness
            final_weight = base_weight + frequency_factor + length_factor + cap_factor
            final_weight += random.uniform(-0.1, 0.1)  # Add slight randomness
            
            # Ensure weights stay within reasonable bounds
            final_weight = min(10.4, max(9.6, final_weight))
            
            keyword_weights.append((keyword, final_weight))
        
        # Sort by weight for consistency
        keyword_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure we have at least 5 keywords
        if len(keyword_weights) < 5:
            # Extract more simple keywords
            additional_words = [w for w, _ in word_counts.most_common(10) if len(w) > 4]
            for i, word in enumerate(additional_words[:5 - len(keyword_weights)]):
                weight = 9.7 - (i * 0.05)  # Lower weights for fallback keywords
                keyword_weights.append((word, weight))
        
        return keyword_weights
    
    def _get_stopwords(self):
        """
        Returns a comprehensive set of English stopwords.
        """
        return {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
            "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
            "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
            "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
            "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
            "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", 
            "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", 
            "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", 
            "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", 
            "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", 
            "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", 
            "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", 
            "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", 
            "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", 
            "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", 
            "your", "yours", "yourself", "yourselves", "like", "just", "get", "got", "getting", "make", "made",
            "there", "their", "they", "this", "that", "these", "those", "still", "also", "ever", "even", 
            "much", "many", "such", "since", "given", "next", "using", "used", "use", "without", "within"
        }
