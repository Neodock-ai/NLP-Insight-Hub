from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random
import re
from collections import Counter
import logging
import math

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
                "admirable": 1.5, "marvelous": 1.5, 
                
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
    
    def _has_excessive_repetition(self, text):
        """
        Detects excessive repetition of phrases or sentences in the text.
        
        Args:
            text (str): Text to check for repetition
            
        Returns:
            bool: True if excessive repetition is detected
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Check for duplicate sentences
        sentence_count = len(sentences)
        unique_sentences = set(sentences)
        
        if sentence_count > 3 and len(unique_sentences) < sentence_count * 0.7:
            return True
            
        # Check for repeated phrases (3+ words)
        words = text.split()
        
        if len(words) >= 9:  # Need at least 9 words to have repeated 3-word phrases
            phrases = []
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3]).lower()
                phrases.append(phrase)
                
            phrase_counts = Counter(phrases)
            for phrase, count in phrase_counts.items():
                if count > 2 and len(phrase.split()) >= 3:  # Phrase appears more than twice
                    return True
                    
        return False
    
    def _post_process_summary(self, summary, original_text):
        """
        Improve the quality of the generated summary by fixing formatting issues,
        detecting incomplete sentences, and removing excessive repetition.
        
        Args:
            summary (str): The raw summary from the model
            original_text (str): The original text being summarized
            
        Returns:
            str: Enhanced and properly formatted summary
        """
        try:
            # Clean up the summary
            summary = summary.strip()
            
            # Remove any repeat of "Summary:" in the output
            summary = re.sub(r'^(Summary:?\s*)', '', summary, flags=re.IGNORECASE)
            
            # Check if summary is too short, has incomplete sentences, or has excessive repetition
            is_low_quality = len(summary.split()) < 10 or len(summary) < 50
            has_incomplete_sentence = re.search(r'[a-zA-Z][^.!?]*$', summary) is not None
            has_repetition = self._has_excessive_repetition(summary)
            
            if is_low_quality or has_incomplete_sentence or has_repetition:
                # Generate extractive summary from original text
                extractive_summary = self._generate_extractive_summary(original_text)
                
                if is_low_quality:
                    logger.info("Original summary too short or low quality, using extractive summary")
                    summary = extractive_summary
                elif has_incomplete_sentence:
                    logger.info("Original summary has incomplete sentences, using extractive summary")
                    summary = extractive_summary
                elif has_repetition:
                    logger.info("Original summary has excessive repetition, using extractive summary")
                    summary = extractive_summary
                    
            # Ensure proper formatting and structure
            summary = self._improve_summary_formatting(summary)
            
            return summary
        except Exception as e:
            logger.error(f"Error in post-processing summary: {str(e)}")
            return summary  # Return original summary if processing fails
    
    def _generate_extractive_summary(self, text):
        """
        Creates an extractive summary by selecting important sentences from the text.
        Uses an enhanced version of the TextRank algorithm.
        
        Args:
            text (str): Original text to summarize
            
        Returns:
            str: Extractive summary
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= 5:
            return text  # Text is already short enough
            
        # Remove very short sentences (likely headings, etc.)
        sentences = [s for s in sentences if len(s.split()) >= 4]
        
        # Calculate word frequencies for TF scoring
        word_freq = Counter(re.findall(r'\b\w+\b', text.lower()))
        total_words = sum(word_freq.values())
        
        # Calculate importance scores for each sentence using multiple factors
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 4:  # Skip very short sentences
                continue
                
            # Extract words and normalize
            words = re.findall(r'\b\w+\b', sentence.lower())
            
            # Skip if too few content words
            if len(words) == 0:
                sentence_scores.append(0)
                continue
                
            # 1. Position score - sentences at beginning and end often more important
            relative_pos = i / max(1, len(sentences))
            if relative_pos < 0.2:  # First 20% of text
                pos_score = 1.0 - relative_pos
            elif relative_pos > 0.8:  # Last 20% of text
                pos_score = 5 * (relative_pos - 0.8)
            else:  # Middle of text
                pos_score = 0.3
                
            # 2. Word frequency score - sentences with frequent words more important
            # Use TF-IDF inspired approach
            word_importance = 0
            for word in words:
                # Skip very common words
                if word in self._get_stopwords():
                    continue
                    
                # Term frequency in the document
                tf = word_freq.get(word, 0) / max(1, total_words)
                
                # Inverse sentence frequency (rarer words more important)
                isf = math.log(len(sentences) / max(1, sum(1 for s in sentences if word.lower() in s.lower())))
                
                word_importance += tf * isf
                
            # Normalize by sentence length
            freq_score = word_importance / max(1, len(words))
            
            # 3. Proper noun/entity score - sentences with entities often important
            entity_count = sum(1 for w in words if w[0].isupper() and not words.index(w) == 0)
            entity_score = min(1.0, entity_count / 3)  # Cap at 1.0
            
            # 4. Length score - favor medium-length sentences
            length = len(words)
            if length < 5:
                length_score = length / 5  # Short sentences get lower scores
            elif length > 30:
                length_score = 30 / length  # Long sentences get lower scores
            else:
                length_score = 1.0  # Medium sentences get full score
                
            # 5. Title word score - sentences containing words from the title/beginning
            first_sentence_words = set(re.findall(r'\b\w+\b', sentences[0].lower()))
            title_overlap = len(set(words).intersection(first_sentence_words))
            title_score = min(1.0, title_overlap / max(1, len(first_sentence_words)))
            
            # Combine scores with weights
            final_score = (
                0.25 * pos_score +
                0.35 * freq_score +
                0.15 * entity_score +
                0.10 * length_score +
                0.15 * title_score
            )
            
            sentence_scores.append(final_score)
            
        # Create sentence-score pairs
        ranked_sentences = [(sentences[i], sentence_scores[i]) for i in range(len(sentences)) 
                             if i < len(sentence_scores)]
        
        # Sort by score
        ranked_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences (adaptive to text length)
        num_sentences = min(7, max(3, len(sentences) // 15))
        top_sentences = [s[0] for s in ranked_sentences[:num_sentences]]
        
        # Reorder sentences to maintain original flow
        final_sentences = [s for s in sentences if s in top_sentences]
        
        # Ensure we have enough sentences
        if len(final_sentences) < 2 and len(sentences) >= 2:
            final_sentences = [sentences[0]]  # Always include first sentence
            if len(sentences) > 2:
                final_sentences.append(sentences[-1])  # Include last sentence if available
                
        return " ".join(final_sentences)
    
    def _improve_summary_formatting(self, summary):
        """
        Improves summary formatting for better readability.
        
        Args:
            summary (str): The summary text
            
        Returns:
            str: Formatted summary
        """
        # Fix sentence spacing
        summary = re.sub(r'(?<=[.!?])(?=[A-Z])', ' ', summary)
        
        # Remove excessive whitespace
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # Ensure the summary ends with proper punctuation
        if not re.search(r'[.!?]$', summary):
            summary += '.'
            
        # Capitalize first letter if needed
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
            
        return summary
    
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
                "explanation": explanation,
                "confidence": max(sentiment_scores.values())
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "sentiment": "Neutral",
                "scores": {"neutral": 1.0, "positive": 0.0, "negative": 0.0},
                "explanation": f"Error during analysis: {str(e)}",
                "confidence": 0.7
            }
    
    def _extract_keywords_advanced(self, text, result):
        """
        Enhanced keyword extraction with better relevance scoring and categorization.
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
                stopwords = self._get_enhanced_stopwords()
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
