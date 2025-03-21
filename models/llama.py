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
        Enhanced sentiment analysis with better classification and confidence scores.
        """
        try:
            # Process the raw result to extract sentiment
            result_lower = result.lower()
            
            # Initialize scores
            sentiment_scores = {
                "positive": 0.0,
                "neutral": 0.0,
                "negative": 0.0
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
            pos_terms = ["good", "great", "excellent", "wonderful", "happy", "positive", 
                         "love", "enjoy", "beneficial", "outstanding", "impressive", "best"]
            
            neg_terms = ["bad", "terrible", "awful", "poor", "negative", "horrible", 
                         "hate", "dislike", "disappointing", "worst", "problem", "difficult"]
            
            # Count sentiment terms
            text_lower = text.lower()
            pos_count = sum(1 for term in pos_terms if term in text_lower)
            neg_count = sum(1 for term in neg_terms if term in text_lower)
            
            # Adjust scores based on term counts
            total_count = pos_count + neg_count
            if total_count > 0:
                # Factor in term counts, but don't override completely
                term_positive = pos_count / total_count if total_count > 0 else 0.5
                
                # Only adjust by a portion to respect the model's output
                sentiment_scores["positive"] += term_positive * 0.3
                sentiment_scores["negative"] += (1 - term_positive) * 0.3
            
            # Account for negation
            negations = ["not", "no", "never", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't"]
            for neg in negations:
                for pos in pos_terms:
                    pattern = f"{neg} {pos}"
                    if pattern in text_lower:
                        # Reduce positive, increase negative
                        sentiment_scores["positive"] -= 0.1
                        sentiment_scores["negative"] += 0.1
                
                for neg_term in neg_terms:
                    pattern = f"{neg} {neg_term}"
                    if pattern in text_lower:
                        # Reduce negative, increase positive
                        sentiment_scores["negative"] -= 0.1
                        sentiment_scores["positive"] += 0.1
            
            # Normalize scores to sum to 1.0
            total_score = sum(sentiment_scores.values())
            if total_score != 0:
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
            
            # Get explanation from result if available
            explanation = ""
            if "because" in result_lower or "due to" in result_lower or "as it" in result_lower:
                explanation_match = re.search(r'(because|due to|as it)(.+)', result_lower)
                if explanation_match:
                    explanation = explanation_match.group(2).strip()
            
            return {
                "sentiment": sentiment_classification,
                "scores": sentiment_scores,
                "explanation": explanation
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "sentiment": "Neutral",
                "scores": {"neutral": 1.0, "positive": 0.0, "negative": 0.0},
                "explanation": f"Error during analysis: {str(e)}"
            }
    
    def _extract_keywords_advanced(self, text, result):
        """
        Enhanced keyword extraction with better relevance scoring.
        """
        try:
            # Try to extract keywords from the model result first
            keywords = []
            
            # Check if result contains comma-separated keywords
            if "," in result:
                # Split by commas and clean
                keywords = [k.strip() for k in result.split(",") if k.strip()]
            
            # If we don't have enough keywords, extract from text
            if len(keywords) < 10:
                # Preprocess text
                clean_text = ' '.join(re.findall(r'\b\w+\b', text.lower()))
                words = clean_text.split()
                
                # Remove stopwords
                stopwords = self._get_stopwords()
                filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
                
                # Get word frequencies
                word_freq = Counter(filtered_words)
                
                # Extract bigrams (two-word phrases)
                bigrams = []
                for i in range(len(words) - 1):
                    if (words[i] not in stopwords and words[i+1] not in stopwords and
                        len(words[i]) > 2 and len(words[i+1]) > 2):
                        bigram = f"{words[i]} {words[i+1]}"
                        bigrams.append(bigram)
                
                bigram_freq = Counter(bigrams)
                
                # Combine unigrams and bigrams with weights
                # Bigrams generally get higher weights
                candidates = []
                for word, count in word_freq.most_common(20):
                    # Base score is frequency
                    score = count
                    
                    # Adjust by word length (longer words often more meaningful)
                    length_factor = min(2.0, len(word) / 4)
                    
                    # Final word score
                    candidates.append((word, score * length_factor))
                
                for bigram, count in bigram_freq.most_common(15):
                    # Bigrams get a boost
                    candidates.append((bigram, count * 2.0))
                
                # Sort by score and take top candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                extracted_keywords = [c[0] for c in candidates[:15]]
                
                # Add any extracted keywords that aren't already in our list
                for kw in extracted_keywords:
                    if kw not in keywords:
                        keywords.append(kw)
            
            # Ensure proper casing for keywords
            proper_case_keywords = []
            for keyword in keywords:
                # Try to find the keyword with proper casing in the original text
                keyword_lower = keyword.lower()
                pattern = re.compile(r'\b' + re.escape(keyword_lower) + r'\b', re.IGNORECASE)
                matches = pattern.findall(text)
                
                if matches:
                    # Use the most common case
                    case_counter = Counter(matches)
                    proper_case_keywords.append(case_counter.most_common(1)[0][0])
                else:
                    proper_case_keywords.append(keyword)
            
            # Assign weights based on relevance
            keyword_weights = []
            for i, keyword in enumerate(proper_case_keywords[:15]):  # Limit to top 15
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
                
                # Final weight with some randomness
                final_weight = base_weight + freq_factor + length_factor + pos_factor
                final_weight += random.uniform(-0.1, 0.1)  # Add slight randomness
                
                # Ensure weight is in reasonable range
                final_weight = min(10.5, max(9.5, final_weight))
                
                keyword_weights.append((keyword, final_weight))
            
            # Sort by weight
            keyword_weights.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_weights
        except Exception as e:
            logger.error(f"Error in keyword extraction: {str(e)}")
            return [("Error", 10.0)]
    
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
