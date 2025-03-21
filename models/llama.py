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
        Enhance the generated summary with advanced post-processing techniques
        for better coherence, completeness, and readability.
        
        Args:
            summary (str): The raw summary from the model
            original_text (str): The original text being summarized
            
        Returns:
            dict: Enhanced summary with metadata
        """
        try:
            # Clean up the summary
            summary = summary.strip()
            
            # Remove any prefixes like "Summary:" in the output
            summary = re.sub(r'^(Summary:?\s*)', '', summary, flags=re.IGNORECASE)
            
            # Check if summary is too short, low quality, or not meaningful
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
            
            # Extract key topics from the summary
            key_topics = self._extract_summary_topics(summary, original_text)
            
            # Calculate readability metrics
            readability_metrics = self._calculate_readability(summary)
            
            # Determine coverage score (how well the summary covers the original text)
            coverage_score = self._calculate_coverage(summary, original_text)
            
            # Create rich summary object with metadata
            enhanced_summary = {
                "text": summary,
                "topics": key_topics,
                "readability": readability_metrics,
                "coverage": coverage_score,
                "word_count": len(summary.split()),
                "compression_ratio": len(original_text.split()) / max(1, len(summary.split()))
            }
            
            return enhanced_summary
        except Exception as e:
            logger.error(f"Error in post-processing summary: {str(e)}")
            # Return basic summary if processing fails
            return {
                "text": summary,
                "topics": [],
                "readability": {"score": 0},
                "coverage": 0,
                "word_count": len(summary.split()),
                "compression_ratio": 0
            }
    
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
    
    def _extract_summary_topics(self, summary, original_text):
        """
        Extracts key topics from the summary for better insights.
        
        Args:
            summary (str): The summary text
            original_text (str): The original text
            
        Returns:
            list: Key topics
        """
        # Extract potential topic words (nouns, often capitalized)
        summary_words = summary.split()
        original_words = original_text.split()
        
        # Filter for potential topic words (longer words, proper nouns, etc.)
        potential_topics = []
        
        for word in summary_words:
            # Skip small words and stopwords
            if len(word) <= 3 or word.lower() in self._get_stopwords():
                continue
                
            # Clean the word (remove punctuation)
            clean_word = re.sub(r'[^\w]', '', word)
            if not clean_word:
                continue
                
            # Potential topic indicators:
            is_proper_noun = clean_word[0].isupper() and not clean_word.isupper()
            is_frequent = summary.lower().count(clean_word.lower()) > 1
            is_in_original = original_text.lower().count(clean_word.lower()) > 2
            
            if is_proper_noun or is_frequent or is_in_original:
                potential_topics.append(clean_word)
                
        # Count frequency
        topic_freq = Counter(potential_topics)
        
        # Select top topics
        top_topics = [topic for topic, _ in topic_freq.most_common(5)]
        
        return top_topics
    
    def _calculate_readability(self, text):
        """
        Calculates readability metrics for the summary.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Readability metrics
        """
        # Simple Flesch Reading Ease score approximation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        words = text.split()
        syllables = self._count_syllables(text)
        
        # Avoid division by zero
        if len(sentences) == 0 or len(words) == 0:
            return {"score": 0, "grade_level": "Unknown", "complexity": "Unknown"}
            
        # Calculate averages
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / max(1, len(words))
        
        # Flesch Reading Ease score (higher is easier to read)
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Determine grade level and complexity
        if score >= 90:
            grade_level = "5th grade"
            complexity = "Very Easy"
        elif score >= 80:
            grade_level = "6th grade"
            complexity = "Easy"
        elif score >= 70:
            grade_level = "7th grade"
            complexity = "Fairly Easy"
        elif score >= 60:
            grade_level = "8th-9th grade"
            complexity = "Standard"
        elif score >= 50:
            grade_level = "10th-12th grade"
            complexity = "Fairly Difficult"
        elif score >= 30:
            grade_level = "College"
            complexity = "Difficult"
        else:
            grade_level = "College Graduate"
            complexity = "Very Difficult"
            
        return {
            "score": round(score, 1),
            "grade_level": grade_level,
            "complexity": complexity
        }
    
    def _count_syllables(self, text):
        """
        Estimates syllable count in text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            int: Estimated syllable count
        """
        # Simple heuristic for English syllable counting
        text = text.lower()
        text = re.sub(r'[^a-z]', ' ', text)
        words = text.split()
        
        count = 0
        for word in words:
            word_count = 0
            
            # Count vowel groups as syllables
            if len(word) <= 3:  # Short words often have just one syllable
                word_count = 1
            else:
                # Count vowel groups
                vowels = "aeiouy"
                prev_is_vowel = False
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_is_vowel:
                        word_count += 1
                    prev_is_vowel = is_vowel
                
                # Words ending in e often don't count that as a syllable
                if word.endswith('e'):
                    word_count -= 1
                    
                # Words ending with le usually add a syllable
                if len(word) > 2 and word.endswith('le') and word[-3] not in vowels:
                    word_count += 1
                    
                # Words ending with es or ed often don't add a syllable
                if word.endswith(('es', 'ed')) and len(word) > 2 and word[-3] not in vowels:
                    word_count -= 1
                    
                # Every word has at least one syllable
                word_count = max(1, word_count)
                
            count += word_count
            
        return count
    
    def _calculate_coverage(self, summary, original_text):
        """
        Calculates how well the summary covers the main topics of the original text.
        
        Args:
            summary (str): The summary text
            original_text (str): The original text
            
        Returns:
            float: Coverage score (0-1)
        """
        # Get important words from original text
        original_words = re.findall(r'\b\w{4,}\b', original_text.lower())
        word_freq = Counter(original_words)
        
        # Filter out stopwords
        important_words = [(word, count) for word, count in word_freq.most_common(50)
                            if word not in self._get_stopwords()]
        
        # Calculate coverage
        summary_lower = summary.lower()
        covered_count = 0
        
        for word, _ in important_words:
            if word in summary_lower:
                covered_count += 1
                
        # Calculate coverage percentage
        coverage = covered_count / max(1, len(important_words))
        
        return round(coverage, 2)
    
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
