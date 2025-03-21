from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random
import re
from collections import Counter
import logging

# Setup logging
logger = logging.getLogger(__name__)

class LlamaModel:
    def __init__(self):
        try:
            # For demonstration, we use 't5-small' as a stand-in for Llama
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            self.name = "Llama"
            
            # Initialize sentiment lexicons as class variables for reuse
            self.positive_lexicon = {
                # Strong positive indicators (weight 2.0)
                "excellent": 2.0, "outstanding": 2.0, "exceptional": 2.0, "superb": 2.0,
                "perfect": 2.0, "brilliant": 2.0, "fantastic": 2.0, "extraordinary": 2.0,
                "wonderful": 2.0, "amazing": 2.0, "incredible": 2.0, "phenomenal": 2.0,
                
                # Moderate positive indicators (weight 1.5)
                "great": 1.5, "impressive": 1.5, "terrific": 1.5, "awesome": 1.5, 
                "delightful": 1.5, "favorable": 1.5, "remarkable": 1.5, "splendid": 1.5,
                "admirable": 1.5, "marvelous": 1.5, "excellent": 1.5,
                
                # Standard positive indicators (weight 1.0)
                "good": 1.0, "nice": 1.0, "positive": 1.0, "satisfactory": 1.0, 
                "pleasing": 1.0, "pleasant": 1.0, "enjoy": 1.0, "happy": 1.0,
                "glad": 1.0, "pleased": 1.0, "thankful": 1.0, "grateful": 1.0,
                "satisfied": 1.0, "like": 1.0, "love": 1.0, "appreciate": 1.0,
                "beneficial": 1.0, "helpful": 1.0, "useful": 1.0, "valuable": 1.0,
                
                # Mild positive indicators (weight 0.5)
                "decent": 0.5, "fine": 0.5, "acceptable": 0.5, "adequate": 0.5,
                "sufficient": 0.5, "fair": 0.5, "reasonable": 0.5, "better": 0.5,
                "improvement": 0.5, "improved": 0.5, "recommend": 0.5, "worth": 0.5
            }
            
            self.negative_lexicon = {
                # Strong negative indicators (weight 2.0)
                "terrible": 2.0, "horrible": 2.0, "awful": 2.0, "dreadful": 2.0,
                "abysmal": 2.0, "atrocious": 2.0, "appalling": 2.0, "disastrous": 2.0,
                "catastrophic": 2.0, "horrendous": 2.0, "unbearable": 2.0, "intolerable": 2.0,
                
                # Moderate negative indicators (weight 1.5)
                "bad": 1.5, "poor": 1.5, "disappointing": 1.5, "frustrating": 1.5,
                "annoying": 1.5, "irritating": 1.5, "unpleasant": 1.5, "unfavorable": 1.5,
                "inferior": 1.5, "unsatisfactory": 1.5, "subpar": 1.5, "unacceptable": 1.5,
                
                # Standard negative indicators (weight 1.0)
                "negative": 1.0, "problem": 1.0, "issue": 1.0, "concern": 1.0,
                "trouble": 1.0, "difficult": 1.0, "hard": 1.0, "challenging": 1.0,
                "dislike": 1.0, "hate": 1.0, "upset": 1.0, "sad": 1.0,
                "angry": 1.0, "mad": 1.0, "unhappy": 1.0, "disappointed": 1.0,
                "failure": 1.0, "failed": 1.0, "fail": 1.0, "broken": 1.0,
                
                # Mild negative indicators (weight 0.5)
                "slow": 0.5, "inconvenient": 0.5, "inconsistent": 0.5, "mediocre": 0.5,
                "overpriced": 0.5, "expensive": 0.5, "lacking": 0.5, "limited": 0.5,
                "confusing": 0.5, "uncomfortable": 0.5, "underwhelming": 0.5, "unfortunate": 0.5
            }
            
        except Exception as e:
            logger.error(f"Error initializing LlamaModel: {str(e)}")
            raise
    
    def generate_text(self, prompt):
        """
        Generates text using the Llama model with improved task-specific handling.
        """
        try:
            task_type = self._determine_task_type(prompt)
            
            if task_type == "summary":
                # Extract the text to summarize
                text_to_process = self._extract_text_for_summary(prompt)
                input_prompt = f"summarize: {text_to_process}"
            
            elif task_type == "sentiment":
                # Extract the text for sentiment analysis
                text_to_process = self._extract_text_for_sentiment(prompt)
                input_prompt = f"classify sentiment of the following text as positive, negative, or neutral and explain why: {text_to_process}"
            
            elif task_type == "keywords":
                # Extract the text for keyword extraction
                text_to_process = self._extract_text_for_keywords(prompt)
                input_prompt = f"extract important keywords and key phrases: {text_to_process}"
            
            elif task_type == "qa":
                # Extract the text and question for Q&A
                text, question = self._extract_text_and_question(prompt)
                input_prompt = f"context: {text} question: {question} answer:"
            
            else:
                # Default handling
                input_prompt = prompt
            
            # Process with the model with improved parameters
            inputs = self.tokenizer.encode(input_prompt, return_tensors="pt", max_length=1024, truncation=True)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                device = torch.device("cuda")
                inputs = inputs.to(device)
                self.model = self.model.to(device)
            
            # Generate with better parameters
            outputs = self.model.generate(
                inputs, 
                max_length=250,  # Longer outputs for better summaries
                num_beams=5,     # More beams for better quality
                length_penalty=1.5,  # Encourage longer outputs for summaries
                temperature=0.7,    # Controlled randomness
                top_p=0.95,         # Better nucleus sampling
                do_sample=True,    
                early_stopping=True,
                no_repeat_ngram_size=3  # Avoid repetition
            )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process based on task type
            if task_type == "summary":
                # Clean up and structure the summary
                return self._post_process_summary(result, text_to_process)
            
            elif task_type == "sentiment":
                # Enhanced sentiment analysis with better classification
                return self._analyze_sentiment_advanced(text_to_process, result)
            
            elif task_type == "keywords":
                # Enhanced keyword extraction with better relevance
                return self._extract_keywords_advanced(text_to_process, result)
            
            return result
        except Exception as e:
            logger.error(f"Error in generate_text: {str(e)}")
            if task_type == "sentiment":
                return {"sentiment": "Neutral", "scores": {"neutral": 1.0, "positive": 0.0, "negative": 0.0}, "explanation": f"Error during analysis: {str(e)}"}
            elif task_type == "keywords":
                return [("Error", 10.0)]
            else:
                return f"Error generating response: {str(e)}"
    
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
        elif "Answer the question" in prompt or ("Based on the" in prompt and "question" in prompt):
            return "qa"
        else:
            return "general"
    
    def _extract_text_for_summary(self, prompt):
        """Extract the text portion from the summarization prompt."""
        try:
            # Try different patterns to extract the text
            if "Text to summarize:" in prompt:
                text = prompt.split("Text to summarize:")[1].split("Summary:")[0].strip()
            else:
                text = prompt.split("Summarize the following text")[1].split("Summary:")[0].strip()
                # Remove any leading/trailing markers
                text = re.sub(r'^\s*:\s*', '', text)
                text = re.sub(r'^\s*with these requirements:[^:]+', '', text)
            
            return text
        except Exception as e:
            logger.warning(f"Error extracting summary text, using full prompt: {str(e)}")
            return prompt
    
    def _extract_text_for_sentiment(self, prompt):
        """Extract the text portion from the sentiment analysis prompt."""
        try:
            if "Text for sentiment analysis:" in prompt:
                text = prompt.split("Text for sentiment analysis:")[1].split("Sentiment")[0].strip()
            else:
                text = prompt.split("Analyze the sentiment of the following text")[1].split("Sentiment")[0].strip()
                # Remove any leading/trailing markers
                text = re.sub(r'^\s*:\s*', '', text)
                text = re.sub(r'^\s*with these requirements:[^:]+', '', text)
            
            return text
        except Exception as e:
            logger.warning(f"Error extracting sentiment text, using full prompt: {str(e)}")
            return prompt
    
    def _extract_text_for_keywords(self, prompt):
        """Extract the text portion from the keyword extraction prompt."""
        try:
            if "Text for keyword extraction:" in prompt:
                text = prompt.split("Text for keyword extraction:")[1].split("Keywords:")[0].strip()
            else:
                text = prompt.split("Extract the main keywords")[1].split("Keywords:")[0].strip()
                # Remove any leading/trailing markers
                text = re.sub(r'^\s*:\s*', '', text)
                text = re.sub(r'^\s*from the following text:?\s*', '', text)
                text = re.sub(r'^\s*and key concepts from the following text:?\s*', '', text)
            
            return text
        except Exception as e:
            logger.warning(f"Error extracting keywords text, using full prompt: {str(e)}")
            return prompt
    
    def _extract_text_and_question(self, prompt):
        """Extract the text and question from the Q&A prompt."""
        try:
            # Handle various prompt formats
            if "Context text:" in prompt:
                parts = prompt.split("Context text:")[1]
                text = parts.split("Question:")[0].strip()
                question = parts.split("Question:")[1].split("Answer:")[0].strip()
            elif "Text:" in prompt:
                parts = prompt.split("Text:")[1]
                if "\n\nQuestion:" in parts:
                    text = parts.split("\n\nQuestion:")[0].strip()
                    question = parts.split("\n\nQuestion:")[1].split("\n\nAnswer:")[0].strip()
                else:
                    text = parts.split("Question:")[0].strip()
                    question = parts.split("Question:")[1].split("Answer:")[0].strip()
            else:
                # Default fallback
                parts = prompt.split("context:", 1) if "context:" in prompt.lower() else prompt.split("text:", 1)
                text = parts[1].split("question:")[0].strip() if len(parts) > 1 else ""
                question_parts = prompt.split("question:", 1) if "question:" in prompt.lower() else ["", ""]
                question = question_parts[1].split("answer:")[0].strip() if len(question_parts) > 1 else ""
            
            if not text or not question:
                raise ValueError("Could not extract both text and question from prompt")
                
            return text, question
        except Exception as e:
            logger.warning(f"Error extracting QA components, using defaults: {str(e)}")
            # If extraction fails, make a best guess
            parts = prompt.split(" ")
            midpoint = len(parts) // 2
            text = " ".join(parts[:midpoint])
            question = " ".join(parts[midpoint:])
            return text, question
    
    def _post_process_summary(self, summary, original_text):
        """
        Improve the quality of the generated summary.
        """
        try:
            # Clean up the summary
            summary = summary.strip()
            
            # Remove any repeat of "Summary:" in the output
            summary = re.sub(r'^(Summary:?\s*)', '', summary, flags=re.IGNORECASE)
            
            # Check if summary is too short or not meaningful
            if len(summary.split()) < 10 or len(summary) < 50:
                # If summary is too short, extract key sentences from original text
                sentences = re.split(r'(?<=[.!?])\s+', original_text)
                
                # Score sentences by importance (using simple heuristics)
                scores = []
                word_freq = Counter(re.findall(r'\b\w+\b', original_text.lower()))
                
                for sentence in sentences:
                    if len(sentence.split()) < 4:  # Skip very short sentences
                        scores.append(0)
                        continue
                        
                    # Score based on word frequency and position
                    words = re.findall(r'\b\w+\b', sentence.lower())
                    
                    # Position score (earlier sentences often more important)
                    pos_score = 1.0 - (sentences.index(sentence) / max(1, len(sentences)))
                    
                    # Word frequency score
                    word_score = sum(word_freq.get(word, 0) for word in words) / max(1, len(words))
                    
                    # Length score (favor medium-length sentences)
                    length_score = min(1.0, len(words) / 20.0) if len(words) <= 20 else 20.0 / len(words)
                    
                    # Final score
                    final_score = 0.3 * pos_score + 0.5 * word_score + 0.2 * length_score
                    scores.append(final_score)
                
                # Sort sentences by score and take top ones
                ranked_sentences = [(sentences[i], scores[i]) for i in range(len(sentences))]
                ranked_sentences.sort(key=lambda x: x[1], reverse=True)
                
                # Take top 3-5 sentences
                num_sentences = min(5, max(3, len(sentences) // 10))
                top_sentences = [s[0] for s in ranked_sentences[:num_sentences]]
                
                # Reorder sentences to maintain original flow
                ordered_sentences = [s for s in sentences if s in top_sentences]
                
                # Join sentences
                summary = " ".join(ordered_sentences)
            
            return summary
        except Exception as e:
            logger.error(f"Error in post-processing summary: {str(e)}")
            return summary  # Return original summary if processing fails

    def _analyze_sentiment_advanced(self, text, result):
        """
        Enhanced sentiment analysis with advanced classification and confidence scoring.
        
        Args:
            text (str): The input text
            result (str): The raw model output
            
        Returns:
            dict: Detailed sentiment analysis with classification, scores, aspects, and confidence
        """
        try:
            # Process the raw result to extract sentiment
            result_lower = result.lower()
            
            # Initialize comprehensive scores
            sentiment_scores = {
                "positive": 0.0,
                "neutral": 0.0,
                "negative": 0.0
            }
            
            # Intensifiers that amplify sentiment
            intensifiers = {
                "very": 1.5, "extremely": 2.0, "incredibly": 2.0, "absolutely": 2.0,
                "completely": 1.8, "totally": 1.8, "thoroughly": 1.7, "entirely": 1.7,
                "highly": 1.6, "especially": 1.5, "particularly": 1.5, "remarkably": 1.6,
                "quite": 1.3, "rather": 1.2, "somewhat": 0.7, "slightly": 0.5,
                "a bit": 0.6, "a little": 0.6, "fairly": 1.1, "pretty": 1.3,
                "really": 1.5, "truly": 1.7, "positively": 1.5, "negatively": 1.5
            }
            
            # Domain-specific sentiment terms (can be customized per domain)
            domain_lexicon = {
                # Technology domain
                "user-friendly": 1.5, "intuitive": 1.5, "responsive": 1.5, "fast": 1.5,
                "slow": -1.5, "buggy": -1.5, "glitchy": -1.5, "crash": -1.5,
                
                # Customer service domain
                "helpful": 1.5, "responsive": 1.5, "prompt": 1.5, "courteous": 1.5,
                "rude": -1.8, "unhelpful": -1.5, "unresponsive": -1.5, "dismissive": -1.8,
                
                # Product quality domain
                "durable": 1.5, "reliable": 1.6, "sturdy": 1.4, "well-made": 1.6,
                "flimsy": -1.5, "unreliable": -1.7, "breaks": -1.5, "defective": -1.8
            }
            
            # Determine primary sentiment from model output
            if "positive" in result_lower:
                primary_sentiment = "Positive"
                sentiment_scores["positive"] += 0.6
                
                # Check for qualifiers that might reduce confidence
                if any(term in result_lower for term in ["somewhat", "slightly", "a bit", "mostly"]):
                    sentiment_scores["positive"] -= 0.2
                    sentiment_scores["neutral"] += 0.2
            
            elif "negative" in result_lower:
                primary_sentiment = "Negative"
                sentiment_scores["negative"] += 0.6
                
                # Check for qualifiers
                if any(term in result_lower for term in ["somewhat", "slightly", "a bit", "mostly"]):
                    sentiment_scores["negative"] -= 0.2
                    sentiment_scores["neutral"] += 0.2
                    
            else:
                primary_sentiment = "Neutral"
                sentiment_scores["neutral"] += 0.6
            
            # Analyze the text directly to confirm sentiment
            text_lower = text.lower()
            
            # Tokenize text into sentences for more granular analysis
            sentences = re.split(r'(?<=[.!?])\s+', text_lower)
            sentence_sentiments = []
            
            # Extract aspects (nouns that have sentiment associated with them)
            aspects = {}
            
            # Process each sentence for sentiment
            for sentence in sentences:
                # Skip very short sentences
                if len(sentence.split()) < 3:
                    continue
                    
                sentence_pos_score = 0
                sentence_neg_score = 0
                sentence_aspects = {}
                
                words = sentence.split()
                
                # Process words for sentiment
                for i, word in enumerate(words):
                    # Check for sentiment terms
                    sentiment_value = 0
                    
                    # Check positive lexicon
                    if word in self.positive_lexicon:
                        sentiment_value = self.positive_lexicon[word]
                        
                        # Check for preceding intensifiers
                        if i > 0 and words[i-1] in intensifiers:
                            sentiment_value *= intensifiers[words[i-1]]
                            
                        sentence_pos_score += sentiment_value
                        
                        # Try to associate with nearby nouns as aspects
                        self._associate_aspect(words, i, sentiment_value, sentence_aspects)
                        
                    # Check negative lexicon
                    elif word in self.negative_lexicon:
                        sentiment_value = -self.negative_lexicon[word]
                        
                        # Check for preceding intensifiers
                        if i > 0 and words[i-1] in intensifiers:
                            sentiment_value *= intensifiers[words[i-1]]
                            
                        sentence_neg_score += sentiment_value
                        
                        # Try to associate with nearby nouns as aspects
                        self._associate_aspect(words, i, sentiment_value, sentence_aspects)
                        
                    # Check domain-specific lexicon
                    elif word in domain_lexicon:
                        sentiment_value = domain_lexicon[word]
                        
                        # Check for preceding intensifiers
                        if i > 0 and words[i-1] in intensifiers:
                            sentiment_value *= intensifiers[words[i-1]]
                            
                        if sentiment_value > 0:
                            sentence_pos_score += sentiment_value
                        else:
                            sentence_neg_score += -sentiment_value
                            
                        # Try to associate with nearby nouns as aspects
                        self._associate_aspect(words, i, sentiment_value, sentence_aspects)
                
                # Account for negation in the sentence
                if any(neg in sentence for neg in ["not", "no", "never", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "can't", "couldn't", "shouldn't", "wouldn't"]):
                    # Look for specific negation patterns
                    negated_pos = self._find_negated_terms(sentence, self.positive_lexicon.keys())
                    negated_neg = self._find_negated_terms(sentence, self.negative_lexicon.keys())
                    
                    # Adjust scores based on negated terms
                    for term, weight in negated_pos:
                        sentence_pos_score -= weight  # Reduce positive score
                        sentence_neg_score += weight * 0.7  # Add to negative, but with slightly less weight
                        
                    for term, weight in negated_neg:
                        sentence_neg_score -= weight  # Reduce negative score
                        sentence_pos_score += weight * 0.7  # Add to positive, but with slightly less weight
                
                # Calculate sentence sentiment
                total_score = sentence_pos_score - sentence_neg_score
                
                if total_score > 0.5:
                    sentence_sentiment = "Positive"
                elif total_score < -0.5:
                    sentence_sentiment = "Negative"
                else:
                    sentence_sentiment = "Neutral"
                    
                # Add to sentence sentiments
                sentence_sentiments.append({
                    "text": sentence,
                    "sentiment": sentence_sentiment,
                    "score": total_score
                })
                
                # Merge sentence aspects into overall aspects
                for aspect, value in sentence_aspects.items():
                    if aspect in aspects:
                        aspects[aspect] += value
                    else:
                        aspects[aspect] = value
            
            # Calculate overall sentiment scores from sentence analysis
            sentence_count = len(sentence_sentiments)
            positive_sentences = sum(1 for s in sentence_sentiments if s["sentiment"] == "Positive")
            negative_sentences = sum(1 for s in sentence_sentiments if s["sentiment"] == "Negative")
            neutral_sentences = sentence_count - positive_sentences - negative_sentences
            
            # Combine model-based and text-based sentiment scores (weighted)
            model_weight = 0.4
            text_weight = 0.6
            
            if sentence_count > 0:
                text_positive = positive_sentences / sentence_count
                text_negative = negative_sentences / sentence_count
                text_neutral = neutral_sentences / sentence_count
                
                # Update scores with weighted combination
                sentiment_scores["positive"] = (sentiment_scores["positive"] * model_weight) + (text_positive * text_weight)
                sentiment_scores["negative"] = (sentiment_scores["negative"] * model_weight) + (text_negative * text_weight)
                sentiment_scores["neutral"] = (sentiment_scores["neutral"] * model_weight) + (text_neutral * text_weight)
            
            # Normalize scores to sum to 1.0
            total_score = sum(sentiment_scores.values())
            if total_score > 0:
                for key in sentiment_scores:
                    sentiment_scores[key] /= total_score
            
            # Determine final sentiment based on highest score
            final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            
            # Format the output
            if final_sentiment == "positive":
                sentiment_classification = "Positive"
            elif final_sentiment == "negative":
                sentiment_classification = "Negative"
            else:
                sentiment_classification = "Neutral"
            
            # Calculate confidence level
            confidence = max(sentiment_scores.values())
            
            # Sort aspects by absolute value of sentiment
            sorted_aspects = sorted(
                aspects.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Take top aspects
            top_aspects = sorted_aspects[:min(5, len(sorted_aspects))]
            aspect_results = {aspect: score for aspect, score in top_aspects}
            
            # Get explanation from result if available
            explanation = ""
            if "because" in result_lower or "due to" in result_lower or "as it" in result_lower:
                explanation_match = re.search(r'(because|due to|as it)(.+)', result_lower)
                if explanation_match:
                    explanation = explanation_match.group(2).strip()
            
            # Extract key phrases that influenced sentiment
            key_phrases = self._extract_sentiment_key_phrases(sentence_sentiments)
            
            return {
                "sentiment": sentiment_classification,
                "scores": sentiment_scores,
                "confidence": confidence,
                "aspects": aspect_results,
                "explanation": explanation,
                "key_phrases": key_phrases
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "sentiment": "Neutral",
                "scores": {"neutral": 1.0, "positive": 0.0, "negative": 0.0},
                "confidence": 0.6,
                "aspects": {},
                "explanation": f"Error during analysis: {str(e)}",
                "key_phrases": []
            }
    
    def _find_negated_terms(self, text, term_list):
        """
        Finds terms that are negated in the text and returns them with their weights.
        
        Args:
            text (str): The input text
            term_list (list): List of terms to check for negation
            
        Returns:
            list: List of (term, weight) tuples for negated terms
        """
        negated_terms = []
        negators = ["not", "no", "never", "don't", "doesn't", "isn't", "aren't", 
                    "wasn't", "weren't", "haven't", "hasn't", "hadn't", "can't", 
                    "couldn't", "shouldn't", "wouldn't"]
        
        # Check each term
        for term in term_list:
            for negator in negators:
                # Look for patterns like "not good", "isn't excellent", etc.
                pattern = r'\b' + re.escape(negator) + r'\b[^.!?]*?\b' + re.escape(term) + r'\b'
                if re.search(pattern, text):
                    # Get the term's weight (assumed to be in a lexicon)
                    weight = 0
                    if term in self.positive_lexicon:
                        weight = self.positive_lexicon[term]
                    elif term in self.negative_lexicon:
                        weight = self.negative_lexicon[term]
                    else:
                        weight = 1.0  # Default weight
                    
                    negated_terms.append((term, weight))
                    
        return negated_terms

    def _associate_aspect(self, words, sentiment_word_index, sentiment_value, aspects_dict):
        """
        Associates sentiment with nearby nouns (potential aspects).
        
        Args:
            words (list): List of words in the sentence
            sentiment_word_index (int): Index of the sentiment word
            sentiment_value (float): The sentiment value
            aspects_dict (dict): Dictionary to update with aspects
        """
        # Simple POS tagging heuristic: nouns often follow determiners and adjectives
        potential_noun_indicators = ["the", "a", "an", "this", "that", "these", "those", "my", "your", "their"]
        
        # Look for nouns before the sentiment word (up to 3 words before)
        start_idx = max(0, sentiment_word_index - 3)
        for i in range(start_idx, sentiment_word_index):
            # Check if this could be a noun
            if i > 0 and words[i-1] in potential_noun_indicators:
                aspects_dict[words[i]] = aspects_dict.get(words[i], 0) + sentiment_value
        
        # Look for nouns after the sentiment word (up to 3 words after)
        end_idx = min(len(words), sentiment_word_index + 4)
        for i in range(sentiment_word_index + 1, end_idx):
            # Check if this could be a noun
            if words[i] not in potential_noun_indicators and len(words[i]) > 2:
                aspects_dict[words[i]] = aspects_dict.get(words[i], 0) + sentiment_value
    
    def _extract_sentiment_key_phrases(self, sentence_sentiments):
        """
        Extracts key phrases that influenced the sentiment the most.
        
        Args:
            sentence_sentiments (list): List of dictionaries with sentence sentiment data
            
        Returns:
            list: List of key phrases with their sentiment
        """
        # Sort sentences by absolute score value (most impactful first)
        sorted_sentences = sorted(
            sentence_sentiments,
            key=lambda x: abs(x["score"]),
            reverse=True
        )
        
        # Take top sentences and extract key phrases
        key_phrases = []
        for sentence_data in sorted_sentences[:3]:  # Top 3 most impactful sentences
            sentence = sentence_data["text"]
            sentiment = sentence_data["sentiment"]
            
            # Try to extract a shorter key phrase if sentence is long
            if len(sentence.split()) > 10:
                # Look for sentiment-bearing phrases
                key_phrase = self._extract_core_phrase(sentence)
                if key_phrase:
                    key_phrases.append({
                        "phrase": key_phrase,
                        "sentiment": sentiment
                    })
            else:
                # Use the whole sentence if it's short
                key_phrases.append({
                    "phrase": sentence,
                    "sentiment": sentiment
                })
        
        return key_phrases
    
    def _extract_core_phrase(self, sentence):
        """
        Extracts the core phrase that likely contains the sentiment.
        
        Args:
            sentence (str): The input sentence
            
        Returns:
            str: The extracted core phrase
        """
        # Look for sentiment-bearing words
        sentiment_words = set(list(self.positive_lexicon.keys()) + list(self.negative_lexicon.keys()))
        
        words = sentence.split()
        for i, word in enumerate(words):
            if word in sentiment_words:
                # Found a sentiment word, extract surrounding context
                start = max(0, i - 5)
                end = min(len(words), i + 6)
                return " ".join(words[start:end])
        
        # If no sentiment word found, return the first part of the sentence
        return " ".join(words[:min(10, len(words))])
    
    def _extract_keywords_advanced(self, text, result):
        """
        Advanced keyword extraction with improved relevance scoring, topic modeling,
        and semantic grouping for better insight generation.
        
        Args:
            text (str): The input text
            result (str): The raw model output
            
        Returns:
            list: Enhanced keyword data with weights, categories, and metadata
        """
        try:
            # Try to extract keywords from the model result first
            keywords = []
            
            # Check if result contains comma-separated keywords
            if "," in result:
                # Split by commas and clean
                keywords = [k.strip() for k in result.split(",") if k.strip()]
            
            # Get clean text for extraction
            clean_text = ' '.join(re.findall(r'\b\w+\b', text.lower()))
            words = clean_text.split()
            
            # Remove stopwords with an expanded set
            stopwords = self._get_enhanced_stopwords()
            filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
            
            # Get word frequencies with context-based weighting
            word_freq = Counter(filtered_words)
            
            # Extract n-grams (2-3 word phrases)
            bigrams = []
            for i in range(len(words) - 1):
                if (words[i] not in stopwords and words[i+1] not in stopwords and
                    len(words[i]) > 2 and len(words[i+1]) > 2):
                    bigram = f"{words[i]} {words[i+1]}"
                    bigrams.append(bigram)
            
            trigrams = []
            for i in range(len(words) - 2):
                if (words[i] not in stopwords and 
                    words[i+1] not in stopwords and 
                    words[i+2] not in stopwords and
                    len(words[i]) > 2 and 
                    len(words[i+1]) > 2 and 
                    len(words[i+2]) > 2):
                    trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                    trigrams.append(trigram)
            
            bigram_freq = Counter(bigrams)
            trigram_freq = Counter(trigrams)
            
            # If we don't have enough keywords from the result, extract from text
            if len(keywords) < 15:
                # Calculate TF-IDF scores for single words
                # For simplicity, we'll use a variant that focuses on term frequency and term specificity
                candidates = []
                
                # Get total word count for tf calculation
                total_words = len(filtered_words)
                
                # Calculate for unigrams (single words)
                for word, count in word_freq.most_common(30):
                    # Term frequency
                    tf = count / max(1, total_words)
                    
                    # Specificity score (longer words often more meaningful)
                    specificity = min(1.0, len(word) / 10)
                    
                    # Position score (words appearing in the beginning or end often more important)
                    word_positions = [i for i, w in enumerate(words) if w == word]
                    position_scores = []
                    for pos in word_positions:
                        relative_pos = pos / max(1, len(words))
                        # Higher score for words near the beginning or end
                        if relative_pos < 0.2 or relative_pos > 0.8:
                            position_scores.append(0.8)
                        else:
                            position_scores.append(0.5)
                    position_score = sum(position_scores) / max(1, len(position_scores))
                    
                    # Calculate final score
                    final_score = tf * (0.6 + 0.2 * specificity + 0.2 * position_score)
                    
                    candidates.append((word, final_score, "concept"))
                
                # Add bigrams with higher weight
                for bigram, count in bigram_freq.most_common(20):
                    # Bigrams get a boost
                    tf = count / max(1, len(bigrams))
                    score = tf * 1.5  # Boost bigrams
                    candidates.append((bigram, score, "phrase"))
                
                # Add trigrams with even higher weight
                for trigram, count in trigram_freq.most_common(10):
                    # Trigrams get an even bigger boost
                    tf = count / max(1, len(trigrams))
                    score = tf * 2.0  # Boost trigrams
                    candidates.append((trigram, score, "phrase"))
                
                # Sort by score and take top candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                extracted_keywords = [(c[0], c[2]) for c in candidates[:30]]
                
                # Add any extracted keywords that aren't already in our list
                for kw, category in extracted_keywords:
                    if kw not in [k for k, _ in keywords]:
                        keywords.append((kw, category))
            else:
                # Assign categories to existing keywords
                keywords = [(kw, self._determine_keyword_category(kw, text)) for kw in keywords]
            
            # Ensure proper casing for keywords
            proper_case_keywords = []
            for keyword, category in keywords:
                # Try to find the keyword with proper casing in the original text
                keyword_lower = keyword.lower()
                pattern = re.compile(r'\b' + re.escape(keyword_lower) + r'\b', re.IGNORECASE)
                matches = pattern.findall(text)
                
                if matches:
                    # Use the most common case
                    case_counter = Counter(matches)
                    proper_case_keywords.append((case_counter.most_common(1)[0][0], category))
                else:
                    proper_case_keywords.append((keyword, category))
            
            # Group keywords by semantic similarity to identify themes
            themes = self._identify_themes(proper_case_keywords)
            
            # Assign weights and additional metadata to keywords
            keyword_data = []
            for i, (keyword, category) in enumerate(proper_case_keywords[:25]):  # Limit to top 25
                # Base weight - decreases slightly with position
                base_weight = 10.3 - (i * 0.05)
                
                # Adjust weight based on term frequency
                frequency = text.lower().count(keyword.lower())
                freq_factor = min(0.3, frequency * 0.05)
                
                # Adjust weight based on keyword length
                length_factor = min(0.2, len(keyword) * 0.01)
                
                # Adjust weight based on position in text (earlier mentions might be more important)
                first_pos = text.lower().find(keyword.lower())
                pos_factor = 0.1 * (1.0 - min(1.0, first_pos / len(text))) if first_pos >= 0 else 0
                
                # Capitalization bonus for proper nouns
                cap_bonus = 0.1 if keyword[0].isupper() else 0
                
                # Theme bonus for keywords in a known theme
                theme_bonus = 0.15 if any(keyword in theme_words for theme, theme_words in themes) else 0
                
                # Final weight with controlled randomness
                final_weight = base_weight + freq_factor + length_factor + pos_factor + cap_bonus + theme_bonus
                final_weight += random.uniform(-0.1, 0.1)  # Add slight randomness for visual variation
                
                # Ensure weight is in reasonable range
                final_weight = min(10.5, max(9.5, final_weight))
                
                # Identify theme for this keyword
                theme = "General"
                for theme_name, theme_words in themes:
                    if keyword in theme_words:
                        theme = theme_name
                        break
                
                # Add rich keyword data
                keyword_data.append({
                    "keyword": keyword,
                    "weight": final_weight,
                    "category": category,
                    "theme": theme,
                    "occurrences": frequency,
                    "is_entity": keyword[0].isupper()  # Simple entity detection
                })
            
            # Sort by weight
            keyword_data.sort(key=lambda x: x["weight"], reverse=True)
            
            # Return in compatible format (list of tuples) while maintaining rich data
            return [(item["keyword"], item["weight"], item.get("category", ""), item.get("theme", "")) 
                    for item in keyword_data]
        except Exception as e:
            logger.error(f"Error in keyword extraction: {str(e)}")
            return [("Error", 10.0, "error", "")]
    
    def _get_enhanced_stopwords(self):
        """
        Returns an enhanced set of English stopwords.
        """
        # Start with the basic stopwords
        basic_stopwords = self._get_stopwords()
        
        # Add more domain-general words that don't add semantic value
        additional_stopwords = {
            "according", "actually", "additionally", "affect", "affected", "affecting",
            "affects", "afterwards", "almost", "already", "although", "altogether",
            "approximately", "aside", "away", "became", "become", "becomes", "becoming",
            "beforehand", "begin", "beginning", "beginnings", "begins", "beside",
            "besides", "beyond", "came", "cannot", "certain", "certainly", "come",
            "comes", "contain", "containing", "contains", "corresponding", "despite",
            "does", "doing", "done", "else", "elsewhere", "enough", "especially",
            "etc", "ever", "example", "examples", "far", "fifth", "find", "four",
            "found", "furthermore", "gave", "getting", "give", "given", "gives",
            "giving", "goes", "gone", "gotten", "happens", "hardly", "hence",
            "hereby", "herein", "hereupon", "hers", "herself", "himself", "hither",
            "hopefully", "how", "howbeit", "however", "hundred", "immediate",
            "immediately", "importance", "important", "inc", "include", "included",
            "includes", "including", "indeed", "indicate", "indicated", "indicates",
            "inner", "insofar", "instead", "interest", "interested", "interesting",
            "interests", "inward", "keep", "keeps", "kept", "know", "known", "knows",
            "lately", "latter", "latterly", "least", "less", "lest", "let", "lets",
            "liked", "likely", "little", "look", "looking", "looks", "ltd", "mainly",
            "maybe", "mean", "means", "meantime", "meanwhile", "merely", "mill",
            "mine", "moreover", "mostly", "move", "namely", "near", "nearly",
            "necessary", "need", "needed", "needing", "needs", "neither", "never",
            "nevertheless", "nine", "nobody", "non", "none", "nonetheless", "noone",
            "normally", "nothing", "notice", "now", "nowhere", "obviously", "often",
            "oh", "okay", "old", "ones", "onto", "opposite", "otherwise", "ought",
            "outside", "overall", "particular", "particularly", "past", "placed",
            "please", "plus", "possible", "presumably", "probably", "provides",
            "que", "quite", "qv", "rather", "readily", "really", "reasonably",
            "regarding", "regardless", "regards", "relatively", "respectively",
            "right", "said", "saw", "say", "saying", "says", "second", "secondly",
            "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self",
            "selves", "sensible", "sent", "serious", "seriously", "seven", "several",
            "shall", "shes", "show", "showed", "shown", "shows", "side", "sides",
            "significant", "similar", "similarly", "six", "slightly", "somebody",
            "somehow", "someone", "something", "sometime", "sometimes", "somewhat",
            "somewhere", "soon", "sorry", "specifically", "specified", "specify",
            "specifying", "sub", "sup", "tell", "tends", "th", "thank", "thanks",
            "thanx", "thats", "thence", "thereafter", "thereby", "therefore", "therein",
            "theres", "thereupon", "they", "think", "third", "thither", "thorough",
            "thoroughly", "thought", "three", "throughout", "thru", "thus", "till",
            "toward", "towards", "truly", "trying", "twice", "un", "under", "unfortunately",
            "unless", "unlikely", "upon", "use", "used", "useful", "uses", "using",
            "usually", "value", "various", "via", "viz", "vs", "want", "wants", "way",
            "welcome", "went", "whatever", "whence", "whenever", "whereafter", "whereas",
            "whereby", "wherein", "whereupon", "wherever", "whether", "whither", "whoever",
            "whole", "whom", "whomever", "whose", "willing", "wish", "within", "without",
            "wonder", "wont", "words", "world", "wouldnt", "www", "yes", "yet", "zero"
        }
        
        # Combine both sets
        return basic_stopwords.union(additional_stopwords)

    def _determine_keyword_category(self, keyword, text):
        """
        Determines the category of a keyword.
        
        Args:
            keyword (str): The keyword to categorize
            text (str): The original text
            
        Returns:
            str: Category of the keyword
        """
        # Check if it appears to be a named entity (proper noun)
        if keyword[0].isupper() and not keyword.isupper():
            return "entity"
        
        # Check if it's a multi-word phrase
        if " " in keyword:
            return "phrase"
        
        # Check if it's a technical term
        technical_indicators = [
            "algorithm", "function", "method", "system", "process", "data",
            "analysis", "research", "study", "experiment", "result", "conclusion",
            "theory", "concept", "framework", "model", "approach", "technique",
            "technology", "implementation", "application", "device", "machine",
            "software", "hardware", "code", "programming", "development",
            "engineering", "science", "scientific", "technical", "protocol"
        ]
        
        if any(ti in text.lower() for ti in technical_indicators):
            return "technical"
        
        # Default category
        return "concept"

    def _identify_themes(self, keywords):
        """
        Identifies potential themes by grouping semantically related keywords.
        
        Args:
            keywords (list): List of (keyword, category) tuples
            
        Returns:
            list: List of (theme_name, [keywords]) tuples
        """
        themes = []
        
        # Simple approach: group by common words/prefixes
        grouped_keywords = {}
        
        for keyword, _ in keywords:
            words = keyword.lower().split()
            
            # Add to groups based on significant words
            for word in words:
                if len(word) > 3 and word not in self._get_stopwords():
                    if word not in grouped_keywords:
                        grouped_keywords[word] = []
                    grouped_keywords[word].append(keyword)
        
        # Keep only groups with multiple keywords
        for key, group in grouped_keywords.items():
            if len(group) >= 2:
                themes.append((key.capitalize(), group))
        
        # Sort themes by number of keywords
        themes.sort(key=lambda x: len(x[1]), reverse=True)
        
        return themes[:5]  # Return top 5 themes
        
    def _get_stopwords(self):
        """
        Returns an enhanced set of English stopwords.
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
            "your", "yours", "yourself", "yourselves", "also", "like", "just", "get", "got", "getting", "make", 
            "made", "many", "much", "well", "may", "might", "shall", "should", "will", "would", "can", "could", 
            "thing", "things", "something", "anything", "nothing", "everything", "someone", "anyone", 
            "everybody", "one", "two", "three", "first", "second", "third", "new", "old", "time", "year", 
            "day", "today", "tomorrow", "yesterday", "now", "then", "always", "never", "yes", "no", "ok", 
            "okay", "right", "wrong", "good", "bad", "sure", "come", "go", "know", "think", "see", "look", 
            "want", "need", "try", "put", "take"
        }
