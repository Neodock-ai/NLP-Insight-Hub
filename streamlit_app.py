import streamlit as st
import logging
import os
from pathlib import Path
import importlib.util
import time
import re
import math
from plotly.subplots import make_subplots

# First check if dependencies are installed
try:
    from pipeline import data_ingestion, pre_processing, inference, post_processing
    from utils.logger import get_logger
    from utils.visualizations import create_sentiment_chart, create_keyword_cloud
except ImportError:
    st.error("Required modules not found. Installing dependencies...")
    import subprocess
    try:
        # Only run setup.sh if it exists
        if Path("setup.sh").exists():
            subprocess.run(["bash", "setup.sh"], check=True)
            st.success("Dependencies installed successfully! Please restart the app.")
            st.stop()
        else:
            st.error("setup.sh file not found. Please make sure all required files are in place.")
            st.stop()
    except subprocess.SubprocessError as e:
        st.error(f"Failed to install dependencies: {str(e)}")
        st.stop()

# Configure logger for the app
logger = get_logger("NLPInsightHub")

# ========== IMPROVED TEXT PROCESSING HELPER FUNCTIONS ==========

def improve_summary_text(text):
    """
    Improve text before summarization by focusing on important parts and removing noise.
    """
    # Remove multiple newlines and whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # For long texts, sample from beginning, middle, and end (these usually have important info)
    if len(text) > 5000:
        first_part = text[:2000]
        middle_part = text[len(text)//2 - 1000:len(text)//2 + 1000]
        last_part = text[-2000:]
        text = first_part + "\n\n...\n\n" + middle_part + "\n\n...\n\n" + last_part
    
    return text

def enhance_sentiment_prompt(text):
    """
    Enhance the prompt for sentiment analysis to get better results.
    """
    # Create a more directive prompt for better sentiment analysis
    enhanced_prompt = f"""Analyze the sentiment in this text. 
Classify it as Positive, Negative, or Neutral.
Be decisive in your classification.
Identify key aspects that influence sentiment and provide confidence levels.
Highlight specific phrases that express sentiment.
Text: {text}
Sentiment analysis:"""
    
    return enhanced_prompt

def clean_summary_output(summary):
    """
    Clean up the summary output to make it more coherent.
    """
    # Handle enhanced summary format (dictionary with text key)
    if isinstance(summary, dict) and "text" in summary:
        summary_text = summary["text"]
    else:
        summary_text = str(summary)
    
    # Remove any "Summary:" prefix that might appear in the output
    summary_text = re.sub(r'^(Summary:?\s*)', '', summary_text, flags=re.IGNORECASE)
    
    # Fix sentence spacing issues
    summary_text = re.sub(r'\.([A-Z])', r'. \1', summary_text)
    
    # Remove duplicate sentences that might appear
    sentences = re.split(r'(?<=[.!?])\s+', summary_text)
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        normalized = sentence.lower().strip()
        if normalized not in seen and len(normalized) > 10:
            seen.add(normalized)
            unique_sentences.append(sentence)
    
    # Join unique sentences back together
    improved_summary = ' '.join(unique_sentences)
    
    # If original was an enhanced dictionary, update the text but keep metadata
    if isinstance(summary, dict) and "text" in summary:
        enhanced_summary = summary.copy()
        enhanced_summary["text"] = improved_summary
        return enhanced_summary
    else:
        return improved_summary

def format_improved_summary(summary):
    """
    Format the summary with better styling.
    """
    # Handle different input types
    if isinstance(summary, dict) and "text" in summary:
        summary_text = summary["text"]
        topics = summary.get("topics", [])
        readability = summary.get("readability", {})
        coverage = summary.get("coverage", 0)
        word_count = summary.get("word_count", 0)
        compression_ratio = summary.get("compression_ratio", 0)
    else:
        summary_text = str(summary)
        topics = []
        readability = {}
        coverage = 0
        word_count = len(summary_text.split())
        compression_ratio = 0
    
    # Create base formatted text
    formatted_text = f"""## Text Summary

{summary_text}

"""
    
    # Add topics if available
    if topics:
        topic_text = ", ".join(topics)
        formatted_text += f"**Key Topics:** {topic_text}\n\n"
    
    # Add readability metrics if available
    if readability and "score" in readability:
        score = readability.get("score", 0)
        grade_level = readability.get("grade_level", "Unknown")
        complexity = readability.get("complexity", "Unknown")
        
        # Add visual indicator for readability
        if score >= 70:
            indicator = "✓ Easy to read"
        elif score >= 50:
            indicator = "○ Average readability"
        else:
            indicator = "⚠ Complex reading"
            
        formatted_text += f"**Readability:** {indicator} ({grade_level} level)\n\n"
    
    # Add coverage and stats if available
    if coverage > 0:
        coverage_pct = int(coverage * 100)
        
        # Format coverage with visual indicator
        if coverage_pct >= 80:
            coverage_text = f"✓ High coverage ({coverage_pct}%)"
        elif coverage_pct >= 60:
            coverage_text = f"○ Good coverage ({coverage_pct}%)"
        else:
            coverage_text = f"⚠ Limited coverage ({coverage_pct}%)"
            
        formatted_text += f"**Coverage:** {coverage_text}\n\n"
    
    formatted_text += "*This summary was generated by AI and captures the key points from the text.*"
    
    # Return rich summary dictionary
    return {
        "text": formatted_text,
        "raw_summary": summary_text,
        "topics": topics,
        "readability": readability,
        "coverage": coverage,
        "word_count": word_count,
        "compression_ratio": compression_ratio
    }

def parse_sentiment_result(sentiment_result):
    """
    Parse sentiment results to handle different output formats.
    """
    if isinstance(sentiment_result, dict):
        # Handle dictionary output (new format)
        if 'sentiment' in sentiment_result:
            sentiment = sentiment_result['sentiment'].lower()
        elif 'scores' in sentiment_result:
            # Get the highest scoring sentiment
            scores = sentiment_result['scores']
            sentiment = max(scores.items(), key=lambda x: x[1])[0].lower()
        else:
            # Default fallback
            sentiment = "neutral"
    else:
        # Handle string output (old format)
        sentiment_text = str(sentiment_result).lower()
        
        if "positive" in sentiment_text:
            sentiment = "positive"
        elif "negative" in sentiment_text:
            sentiment = "negative"
        else:
            sentiment = "neutral"
    
    return sentiment

def format_improved_sentiment(sentiment_result):
    """
    Format sentiment analysis with better styling and comprehensive insights.
    """
    # Handle different input types
    if isinstance(sentiment_result, dict):
        # Use the enhanced sentiment data
        if 'sentiment' in sentiment_result:
            sentiment = sentiment_result['sentiment'].lower()
            scores = sentiment_result.get('scores', {})
            confidence = sentiment_result.get('confidence', 0.0)
            aspects = sentiment_result.get('aspects', {})
            explanation = sentiment_result.get('explanation', '')
            key_phrases = sentiment_result.get('key_phrases', [])
        else:
            # Extract from scores if available
            scores = sentiment_result.get('scores', {})
            if scores:
                sentiment = max(scores.items(), key=lambda x: x[1])[0].lower()
            else:
                sentiment = "neutral"
            confidence = 0.0
            aspects = {}
            explanation = ''
            key_phrases = []
    else:
        # Process string input
        sentiment_text = str(sentiment_result).lower()
        
        if "positive" in sentiment_text:
            sentiment = "positive"
            scores = {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
        elif "negative" in sentiment_text:
            sentiment = "negative"
            scores = {"positive": 0.1, "neutral": 0.2, "negative": 0.7}
        else:
            sentiment = "neutral"
            scores = {"positive": 0.3, "neutral": 0.5, "negative": 0.2}
        
        confidence = max(scores.values())
        aspects = {}
        key_phrases = []
        
        # Try to extract explanation
        explanation = ""
        match = re.search(r'(positive|negative|neutral)[:\s]+(.+)', sentiment_text, re.IGNORECASE)
        if match:
            explanation = match.group(2).strip()
    
    # Determine emoji and color based on sentiment
    if sentiment == "positive":
        emoji = "😃"
        sentiment_text = "Positive"
        color = "green"
    elif sentiment == "negative":
        emoji = "😞"
        sentiment_text = "Negative"
        color = "red"
    else:
        emoji = "😐"
        sentiment_text = "Neutral"
        color = "gray"
    
    # Calculate confidence description
    if isinstance(confidence, (int, float)):
        if confidence > 0.8:
            confidence_text = "High confidence"
        elif confidence > 0.6:
            confidence_text = "Moderate confidence"
        else:
            confidence_text = "Low confidence"
    else:
        confidence_text = ""
    
    # Create formatted text with enhanced details
    formatted_text = f"""## Sentiment Analysis

**Overall sentiment:** <span style="color:{color}">**{sentiment_text}**</span> {emoji}"""

    if confidence_text:
        formatted_text += f" ({confidence_text})\n\n"
    else:
        formatted_text += "\n\n"
    
    # Add explanation if available
    if explanation:
        formatted_text += f"**Analysis:** {explanation}\n\n"
    
    # Add key sentiment phrases if available
    if key_phrases:
        formatted_text += "**Key sentiment phrases:**\n"
        for i, phrase_data in enumerate(key_phrases[:3]):  # Show top 3 phrases
            phrase = phrase_data.get("phrase", "")
            phrase_sentiment = phrase_data.get("sentiment", "")
            
            if phrase_sentiment.lower() == "positive":
                phrase_emoji = "✓"
                phrase_color = "green"
            elif phrase_sentiment.lower() == "negative":
                phrase_emoji = "✗"
                phrase_color = "red"
            else:
                phrase_emoji = "•"
                phrase_color = "gray"
                
            formatted_text += f"{phrase_emoji} <span style=\"color:{phrase_color}\">\"{phrase}\"</span>\n"
        formatted_text += "\n"
    
    # Add aspect-based sentiment if available
    if aspects:
        formatted_text += "**Aspect analysis:**\n"
        
        # Sort aspects by absolute sentiment value
        sorted_aspects = sorted(aspects.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for i, (aspect, value) in enumerate(sorted_aspects[:5]):  # Show top 5 aspects
            if value > 0:
                aspect_emoji = "👍"
                aspect_text = f"{aspect.capitalize()}: Positive"
            elif value < 0:
                aspect_emoji = "👎"
                aspect_text = f"{aspect.capitalize()}: Negative"
            else:
                aspect_emoji = "•"
                aspect_text = f"{aspect.capitalize()}: Neutral"
                
            formatted_text += f"{aspect_emoji} {aspect_text}\n"
        formatted_text += "\n"
    
    formatted_text += "*Note: This is an automated sentiment analysis that evaluates the emotional tone of the text.*"
    
    # Return rich sentiment dictionary
    return {
        "text": formatted_text,
        "raw_sentiment": sentiment,
        "sentiment": sentiment_text,
        "emoji": emoji,
        "confidence": confidence,
        "aspects": aspects,
        "key_phrases": key_phrases,
        "scores": scores,
        "explanation": explanation
    }

def improve_qa_answer(answer):
    """
    Improve Q&A answers for better clarity.
    """
    # Remove any "Answer:" prefix that might appear in the output
    answer = re.sub(r'^(Answer:?\s*)', '', answer, flags=re.IGNORECASE)
    
    # Format the answer
    formatted_answer = f"""### Answer

{answer}

*Note: This answer is generated based on the provided text and may not be comprehensive.*
"""
    
    return formatted_answer

# ========== ENHANCED PROCESSING FUNCTION ==========

# Cache expensive operations
@st.cache_data
def process_text(text, model, tasks, summarization_length=3):
    """Process text with the selected model and return results for all tasks"""
    results = {}
    clean_text = pre_processing.clean_text(text)
    
    if "Summarization" in tasks:
        try:
            # Improve the text before summarization
            summary_text = improve_summary_text(clean_text)
            
            # Get the summary with length parameter
            summary = inference.get_summary(summary_text, model=model, length=summarization_length)
            
            # Clean and improve the summary
            improved_summary = clean_summary_output(summary)
            
            # Format the summary
            results["Summary"] = format_improved_summary(improved_summary)
            
            logger.info(f"Summarization completed with {model}")
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            results["Summary"] = {
                "text": f"Error generating summary: {str(e)}",
                "raw_summary": "",
                "topics": [],
                "readability": {},
                "coverage": 0,
                "word_count": 0,
                "compression_ratio": 0
            }
    
    if "Sentiment Analysis" in tasks:
        try:
            # Create a better prompt for sentiment analysis
            sentiment_prompt = enhance_sentiment_prompt(clean_text)
            
            # Get the sentiment with enhanced model
            sentiment = inference.get_sentiment(clean_text, model=model)
            
            # Process and format the sentiment results
            formatted_sentiment = format_improved_sentiment(sentiment)
            
            # Prepare results with all the rich data
            results["Sentiment Analysis"] = {
                "text": formatted_sentiment["text"],
                "raw_sentiment": formatted_sentiment["raw_sentiment"],
                "sentiment": formatted_sentiment["sentiment"],
                "emoji": formatted_sentiment["emoji"],
                "confidence": formatted_sentiment.get("confidence", 0),
                "aspects": formatted_sentiment.get("aspects", {}),
                "key_phrases": formatted_sentiment.get("key_phrases", []),
                "scores": formatted_sentiment.get("scores", {}),
                "explanation": formatted_sentiment.get("explanation", ""),
                "data": sentiment  # Keep original data for visualization
            }
            
            logger.info(f"Sentiment analysis completed with {model}")
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            results["Sentiment Analysis"] = {"text": f"Error analyzing sentiment: {str(e)}"}
    
    if "Keyword Extraction" in tasks:
        try:
            keywords = inference.get_keywords(clean_text, model=model)
            
            # Format keywords with enhanced function
            formatted_keywords = post_processing.format_keywords(keywords)
            
            # Extract themes and categories from formatted result if available
            themes = {}
            categories = {"entity": [], "phrase": [], "concept": [], "technical": []}
            
            if isinstance(formatted_keywords, dict):
                themes = formatted_keywords.get("themes", {})
                categories = formatted_keywords.get("categories", categories)
                formatted_text = formatted_keywords.get("text", "")
            else:
                formatted_text = str(formatted_keywords)
            
            results["Keyword Extraction"] = {
                "text": formatted_text,
                "data": keywords,  # Raw data for visualization
                "themes": themes,
                "categories": categories
            }
            
            logger.info(f"Keyword extraction completed with {model}")
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            results["Keyword Extraction"] = {"text": f"Error extracting keywords: {str(e)}"}
    
    return clean_text, results

def main():
    st.set_page_config(
        page_title="NLP Insight Hub",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("NLP Insight Hub")
    st.write("### An Industry-Level AI-Powered NLP Pipeline for Business Insights")
    
    # Add session state for persistent settings
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'clean_text' not in st.session_state:
        st.session_state.clean_text = ""
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'raw_text' not in st.session_state:
        st.session_state.raw_text = ""
    
    # Sidebar configuration for model and tasks
    st.sidebar.header("Configuration")
    
    # Model selection with descriptions
    model_options = {
        "Llama": "Meta's powerful large language model",
        "Falcon": "TII's efficient open-source LLM",
        "Mistral": "Lightweight and high-performance model",
        "DeepSeek": "Advanced model for complex language tasks"
    }
    model_choice = st.sidebar.selectbox(
        "Select Model", 
        list(model_options.keys()),
        help="Choose the AI model that will process your text"
    )
    st.sidebar.caption(model_options[model_choice])
    
    # Task selection with descriptions
    task_options = {
        "Summarization": "Generate concise summaries of your text",
        "Sentiment Analysis": "Determine the emotional tone of the text",
        "Keyword Extraction": "Identify important topics and terms",
        "Q&A": "Ask questions about the content"
    }
    
    task_choices = st.sidebar.multiselect(
        "Select Tasks",
        list(task_options.keys()),
        default=["Summarization", "Sentiment Analysis"],
        help="Choose which analyses to perform on your text"
    )
    
    # Add advanced settings in expander
    with st.sidebar.expander("Advanced Settings"):
        summarization_length = st.slider(
            "Summary Length", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="Controls the length of generated summaries (1=very brief, 5=detailed)"
        )
        
        visualization_enabled = st.checkbox(
            "Enable Visualizations", 
            value=True,
            help="Show charts and visualizations for analysis results"
        )
    
    # Input section
    st.sidebar.write("Upload a text file or paste text below:")
    
    tab1, tab2 = st.tabs(["📄 Upload File", "✏️ Enter Text"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload your text file", type=["txt", "pdf", "docx"])
    
    with tab2:
        text_input = st.text_area("Paste text here:", height=200)
    
    # Process button
    col1, col2 = st.columns([1, 4])
    with col1:
        process_button = st.button("Process Text", type="primary")
    with col2:
        if process_button:
            st.session_state.start_time = time.time()
    
    # Progress tracking
    if process_button and (uploaded_file or text_input):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Data Ingestion
            status_text.text("Reading input...")
            progress_bar.progress(10)
            
            if uploaded_file:
                file_extension = Path(uploaded_file.name).suffix
                if file_extension == '.txt':
                    raw_text = data_ingestion.read_file(uploaded_file)
                elif file_extension == '.pdf':
                    raw_text = data_ingestion.read_pdf(uploaded_file)
                elif file_extension == '.docx':
                    raw_text = data_ingestion.read_docx(uploaded_file)
            else:
                if not text_input.strip():
                    st.warning("Please enter some text or upload a file.")
                    st.stop()
                raw_text = text_input

            # Store the raw text for display
            st.session_state.raw_text = raw_text

            status_text.text("Processing text...")
            progress_bar.progress(30)
            
            # Process the text and get results
            st.session_state.clean_text, st.session_state.results = process_text(
                raw_text, 
                model_choice, 
                task_choices,
                summarization_length
            )
            
            progress_bar.progress(90)
            status_text.text("Finalizing results...")
            
            # Set processed flag to true
            st.session_state.processed = True
            
            # Finish
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            # Calculate and display processing time
            processing_time = time.time() - st.session_state.start_time
            st.success(f"✅ Text processed successfully in {processing_time:.2f} seconds")
            
            # Auto-remove progress indicators after 2 seconds
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()
            
            # Force page refresh to show results
            st.rerun()

        except Exception as ex:
            progress_bar.empty()
            status_text.empty()
            st.error("An error occurred while processing your text.")
            st.error(f"Error details: {str(ex)}")
            logger.exception("Error in NLP processing", exc_info=ex)
            
            # Provide troubleshooting information
            st.info(
                "Troubleshooting tips: \n"
                "1. Check if the text format is compatible \n"
                "2. Try with a smaller text sample \n" 
                "3. Restart the application"
            )
    
    # Show the results if data has been processed
    if st.session_state.processed:
        # Original and cleaned text expanders
        with st.expander("Original and Cleaned Text", expanded=False):
            tabs = st.tabs(["Original", "Cleaned"])
            with tabs[0]:
                if 'uploaded_file' in locals() and uploaded_file:
                    st.write(f"File: {uploaded_file.name}")
                st.write(st.session_state.raw_text)
            with tabs[1]:
                st.write(st.session_state.clean_text)
                
        # Display results
        if st.session_state.results:
            st.subheader("Analysis Results")
            
            # Create tabs for different results
            result_tabs = st.tabs([task for task in st.session_state.results.keys() if task != "Q&A"])
            
            for i, (key, value) in enumerate([(k, v) for k, v in st.session_state.results.items() if k != "Q&A"]):
                with result_tabs[i]:
                    # Different display depending on the result type
                    if key == "Summary":
                        # Display formatted summary
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(value["text"] if isinstance(value, dict) and "text" in value else value)
                        
                        with col2:
                            # Display summary metrics in sidebar
                            if isinstance(value, dict):
                                # Show readability gauge
                                if "readability" in value and "score" in value["readability"]:
                                    readability = value["readability"]
                                    readability_score = readability.get("score", 0)
                                    
                                    # Create a simple gauge for readability
                                    if readability_score > 0:
                                        st.markdown("### Readability")
                                        
                                        # Calculate color
                                        if readability_score >= 70:
                                            color = "green"
                                        elif readability_score >= 50:
                                            color = "orange"
                                        else:
                                            color = "red"
                                            
                                        # Display score with colored bar
                                        st.progress(min(1.0, readability_score / 100), text=f"{readability_score:.0f}/100")
                                        st.caption(f"Level: {readability.get('grade_level', 'Unknown')}")
                                
                                # Show coverage gauge if available
                                if "coverage" in value and value["coverage"] > 0:
                                    st.markdown("### Content Coverage")
                                    coverage = value["coverage"]
                                    st.progress(coverage, text=f"{int(coverage * 100)}%")
                                    
                                    if coverage >= 0.8:
                                        st.caption("Excellent coverage")
                                    elif coverage >= 0.6:
                                        st.caption("Good coverage")
                                    else:
                                        st.caption("Partial coverage")
                        
                        # Add metrics display
                        if isinstance(value, dict) and value:
                            # Create expandable metrics section
                            with st.expander("Summary Metrics", expanded=False):
                                # Create 3-column layout for metrics
                                metric_cols = st.columns(3)
                                
                                # Summary metrics
                                word_count = value.get("word_count", 0)
                                compression = value.get("compression_ratio", 0)
                                readability = value.get("readability", {}).get("score", 0)
                                
                                with metric_cols[0]:
                                    st.metric("Word Count", word_count)
                                
                                with metric_cols[1]:
                                    compression_pct = int((1 - (1 / max(1.01, compression))) * 100) if compression > 0 else 0
                                    st.metric("Compression", f"{compression_pct}%")
                                
                                with metric_cols[2]:
                                    st.metric("Readability", f"{int(readability)}/100")
                        
                    elif key == "Sentiment Analysis" and isinstance(value, dict):
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            # Display the formatted sentiment analysis
                            if "text" in value and isinstance(value["text"], str):
                                st.markdown(value["text"])
                            else:
                                # Extract sentiment value to determine display
                                raw_sentiment = value.get("raw_sentiment", "").strip().lower()
                                
                                st.markdown("## Sentiment Analysis")
                                
                                # Determine sentiment category and emoji
                                if "positive" in raw_sentiment:
                                    sentiment_text = "Positive"
                                    emoji = "😃"
                                elif "negative" in raw_sentiment:
                                    sentiment_text = "Negative"
                                    emoji = "😞"
                                else:
                                    sentiment_text = "Neutral"
                                    emoji = "😐"
                                    
                                st.markdown(f"**Overall sentiment:** **{sentiment_text}** {emoji}")
                                    
                                # Display additional sentiment details if available
                                if "aspects" in value and value["aspects"]:
                                    st.markdown("#### Aspect Analysis")
                                    for aspect, score in value["aspects"].items():
                                        sentiment_icon = "👍" if score > 0 else ("👎" if score < 0 else "⚖️")
                                        st.markdown(f"{sentiment_icon} **{aspect.capitalize()}**: {abs(score):.2f}")
                                
                                # Display key phrases if available
                                if "key_phrases" in value and value["key_phrases"]:
                                    st.markdown("#### Key Sentiment Phrases")
                                    for phrase in value["key_phrases"]:
                                        phrase_text = phrase.get("phrase", "")
                                        phrase_sentiment = phrase.get("sentiment", "")
                                        
                                        if phrase_sentiment.lower() == "positive":
                                            st.success(f"\"{phrase_text}\"")
                                        elif phrase_sentiment.lower() == "negative":
                                            st.error(f"\"{phrase_text}\"")
                                        else:
                                            st.info(f"\"{phrase_text}\"")
                                
                                st.markdown("*Note: This is an automated sentiment analysis and may not capture nuanced emotions.*")
                        
                        with col2:
                            if visualization_enabled and "data" in value:
                                sentiment_chart = create_sentiment_chart(value["data"])
                                st.plotly_chart(sentiment_chart, use_container_width=True)
                        
                        # Add metrics display
                        if isinstance(value, dict) and value:
                            # Create expandable metrics section
                            with st.expander("Sentiment Metrics", expanded=False):
                                # Create 3-column layout for metrics
                                metric_cols = st.columns(3)
                                
                                # Sentiment metrics
                                scores = value.get("scores", {})
                                confidence = value.get("confidence", 0)
                                aspect_count = len(value.get("aspects", {}))
                                
                                with metric_cols[0]:
                                    pos_score = scores.get("positive", 0)
                                    st.metric("Positive", f"{int(pos_score * 100)}%")
                                
                                with metric_cols[1]:
                                    neg_score = scores.get("negative", 0)
                                    st.metric("Negative", f"{int(neg_score * 100)}%")
                                
                                with metric_cols[2]:
                                    st.metric("Confidence", f"{int(confidence * 100)}%")
                    
                    elif key == "Keyword Extraction" and "data" in value:
                        col1, col2 = st.columns([2, 3])
                        with col1:
                            # Display the formatted keywords
                            if "text" in value and isinstance(value["text"], str):
                                st.markdown(value["text"])
                            else:
                                st.markdown("## Key Topics & Concepts")
                                st.markdown("Keywords extracted from the text:")
                                
                                # Display themes if available
                                if "themes" in value and value["themes"]:
                                    st.markdown("### Topic Clusters")
                                    for theme, keywords in value["themes"].items():
                                        st.markdown(f"**{theme}**: {', '.join(keywords[:5])}")
                                        if len(keywords) > 5:
                                            st.caption(f"and {len(keywords)-5} more")
                                
                                # Display categories if available
                                if "categories" in value and value["categories"]:
                                    categories = value["categories"]
                                    
                                    if categories.get("entity", []):
                                        st.markdown("**Entities:** " + ", ".join(categories["entity"][:5]))
                                    
                                    if categories.get("technical", []):
                                        st.markdown("**Technical Terms:** " + ", ".join(categories["technical"][:5]))
                                    
                                    if categories.get("phrase", []):
                                        st.markdown("**Key Phrases:** " + ", ".join(categories["phrase"][:5]))
                        
                        with col2:
                            if visualization_enabled and "data" in value:
                                keyword_cloud = create_keyword_cloud(value["data"])
                                st.plotly_chart(keyword_cloud, use_container_width=True)
                        
                        # Add metrics display
                        if isinstance(value, dict) and value:
                            # Create expandable metrics section
                            with st.expander("Keyword Metrics", expanded=False):
                                # Create 3-column layout for metrics
                                metric_cols = st.columns(3)
                                
                                # Keyword metrics
                                keyword_count = len(value.get("data", []))
                                theme_count = len(value.get("themes", {}))
                                entity_count = len(value.get("categories", {}).get("entity", []))
                                
                                with metric_cols[0]:
                                    st.metric("Keywords", keyword_count)
                                
                                with metric_cols[1]:
                                    st.metric("Topic Clusters", theme_count)
                                
                                with metric_cols[2]:
                                    st.metric("Named Entities", entity_count)
                    else:
                        # Fallback for other result types
                        st.markdown(value["text"] if isinstance(value, dict) and "text" in value else value)
            
            # Q&A section (separate from tabs)
            if "Q&A" in task_choices:
                st.subheader("Ask Questions About Your Text")
                
                question = st.text_input("Enter your question about the text:")
                ask_button = st.button("Ask Question")
                
                if question and ask_button:
                    with st.spinner("Processing your question..."):
                        try:
                            answer = inference.get_qa(st.session_state.clean_text, question, model=model_choice)
                            
                            # Apply improved formatting
                            formatted_answer = improve_qa_answer(answer)
                            
                            # Display answer in a nice format
                            st.markdown(formatted_answer)
                            
                            # Add the Q&A to results history
                            if "Q&A History" not in st.session_state:
                                st.session_state["Q&A History"] = []
                                
                            st.session_state["Q&A History"].append({
                                "question": question,
                                "answer": formatted_answer
                            })
                            
                            logger.info("Q&A processing completed.")
                        except Exception as e:
                            logger.error(f"Q&A processing failed: {str(e)}")
                            st.error(f"Error processing question: {str(e)}")
                
                # Display Q&A history if available
                if "Q&A History" in st.session_state and st.session_state["Q&A History"]:
                    with st.expander("Q&A History", expanded=False):
                        for i, qa in enumerate(st.session_state["Q&A History"]):
                            st.markdown(f"**Q{i+1}: {qa['question']}**")
                            st.markdown(qa['answer'])
                            st.divider()
        
        # Export options
        with st.expander("Export Results", expanded=False):
            export_format = st.selectbox(
                "Select export format:",
                ["PDF", "JSON", "CSV", "TXT"]
            )
            
            if st.button("Export Results"):
                st.info("Export functionality would generate a downloadable file with the results.")
                # This would connect to actual export functionality

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        # Create a log file on error for easier debugging
        logging.basicConfig(filename='app_error.log', level=logging.ERROR)
        logging.error("Unhandled exception", exc_info=True)
