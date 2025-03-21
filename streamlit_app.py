import streamlit as st
import logging
import os
from pathlib import Path
import importlib.util
import time

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

# Cache expensive operations
@st.cache_data
def process_text(text, model, tasks, advanced_settings=None):
    """
    Process text with the selected model and return results for all tasks with improved processing.
    
    Args:
        text (str): The text to process
        model (str): The model name to use
        tasks (list): List of tasks to perform
        advanced_settings (dict): Optional advanced settings
        
    Returns:
        tuple: (cleaned_text, results_dict)
    """
    if advanced_settings is None:
        advanced_settings = {}
    
    summary_length = advanced_settings.get('summary_length', 3)
    
    logger.info(f"Processing text with {model} model for tasks: {', '.join(tasks)}")
    results = {}
    
    # Apply appropriate preprocessing based on the tasks
    if "Summarization" in tasks:
        # Special preprocessing for summarization
        summary_text = pre_processing.preprocess_for_summarization(text)
        clean_text = pre_processing.clean_text(text)  # Keep standard clean text for other tasks
    else:
        # Standard preprocessing for other tasks
        clean_text = pre_processing.clean_text(text)
        summary_text = clean_text  # Not used if summarization not selected
    
    # Process each requested task
    if "Summarization" in tasks:
        try:
            # Get optimized summary prompt
            from prompt_engineering import prompt_templates, prompt_optimizer
            prompt = prompt_templates.get_summary_prompt(summary_text)
            optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
            
            # Generate summary with length parameter
            summary = inference.get_summary(summary_text, model=model, length=summary_length)
            
            # Format summary with improved formatting
            results["Summary"] = post_processing.format_summary(summary)
            logger.info("Summarization completed successfully")
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            results["Summary"] = f"Error generating summary: {str(e)}"
    
    if "Sentiment Analysis" in tasks:
        try:
            # Special preprocessing for sentiment analysis
            sentiment_text = pre_processing.preprocess_for_sentiment(text)
            
            # Get sentiment analysis
            sentiment = inference.get_sentiment(sentiment_text, model=model)
            
            # Format sentiment results with improved formatting
            formatted_sentiment = post_processing.format_sentiment(sentiment)
            
            # Structure result for compatibility with visualization
            results["Sentiment Analysis"] = {
                "text": formatted_sentiment.get("text", ""),
                "raw_sentiment": formatted_sentiment.get("raw_sentiment", ""),
                "data": sentiment  # Keep original data for visualization
            }
            logger.info("Sentiment analysis completed successfully")
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            results["Sentiment Analysis"] = {"text": f"Error analyzing sentiment: {str(e)}"}
    
    if "Keyword Extraction" in tasks:
        try:
            # Special preprocessing for keyword extraction
            keyword_text = pre_processing.preprocess_for_keywords(text)
            
            # Extract keywords
            keywords = inference.get_keywords(keyword_text, model=model)
            
            # Format keywords with improved formatting
            formatted_keywords = post_processing.format_keywords(keywords)
            
            # Structure result for compatibility with visualization
            results["Keyword Extraction"] = {
                "text": formatted_keywords.get("text", ""),
                "data": keywords  # Keep original data for visualization
            }
            logger.info("Keyword extraction completed successfully")
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            results["Keyword Extraction"] = {"text": f"Error extracting keywords: {str(e)}"}
    
    return clean_text, results

def answer_question(text, question, model):
    """
    Improved question answering function.
    
    Args:
        text (str): The context text
        question (str): The question to answer
        model (str): The model to use
        
    Returns:
        str: Formatted answer
    """
    try:
        # Apply optimized prompting
        from prompt_engineering import prompt_templates, prompt_optimizer
        prompt = prompt_templates.get_qa_prompt(text, question)
        optimized_prompt = prompt_optimizer.optimize_prompt(prompt)
        
        # Generate answer
        answer = inference.get_qa(text, question, model=model)
        
        # Format answer
        formatted_answer = post_processing.format_qa(answer)
        
        return formatted_answer
    except Exception as e:
        logger.error(f"Error in question answering: {str(e)}")
        return f"Error answering question: {str(e)}"

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
    if 'summary_length' not in st.session_state:
        st.session_state.summary_length = 3
    if 'visualization_enabled' not in st.session_state:
        st.session_state.visualization_enabled = True
    
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
        summary_length = st.slider(
            "Summary Length", 
            min_value=1, 
            max_value=5, 
            value=st.session_state.get('summary_length', 3),
            help="Controls the length of generated summaries (1=very brief, 5=detailed)"
        )
        st.session_state.summary_length = summary_length
        
        visualization_enabled = st.checkbox(
            "Enable Visualizations", 
            value=st.session_state.get('visualization_enabled', True),
            help="Show charts and visualizations for analysis results"
        )
        st.session_state.visualization_enabled = visualization_enabled
    
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

            status_text.text("Processing text...")
            progress_bar.progress(30)
            
            # Process the text and get results with advanced settings
            advanced_settings = {
                'summary_length': summary_length,
                'visualization_enabled': visualization_enabled
            }
            
            st.session_state.clean_text, st.session_state.results = process_text(
                raw_text, 
                model_choice, 
                task_choices,
                advanced_settings
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
            
            # Store raw text for display
            st.session_state.raw_text = raw_text
            
            # Force page refresh to show results
            st.experimental_rerun()

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
                st.write(st.session_state.get('raw_text', "Original text not available"))
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
                        st.markdown(value)
                    elif key == "Sentiment Analysis" and isinstance(value, dict):
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            # Use text key from formatted sentiment if available
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
                                    st.markdown(f"**Overall sentiment:** **{sentiment_text}** {emoji}")
                                elif "negative" in raw_sentiment:
                                    sentiment_text = "Negative"
                                    emoji = "😞"
                                    st.markdown(f"**Overall sentiment:** **{sentiment_text}** {emoji}")
                                else:
                                    sentiment_text = "Neutral"
                                    emoji = "😐"
                                    st.markdown(f"**Overall sentiment:** **{sentiment_text}** {emoji}")
                                    
                                st.markdown("*Note: This is an automated sentiment analysis and may not capture nuanced emotions.*")
                        
                        with col2:
                            if visualization_enabled and "data" in value:
                                sentiment_chart = create_sentiment_chart(value["data"])
                                st.plotly_chart(sentiment_chart, use_container_width=True)
                    
                    elif key == "Keyword Extraction" and isinstance(value, dict):
                        col1, col2 = st.columns([2, 3])
                        with col1:
                            # Use text key from formatted keywords if available
                            if "text" in value and isinstance(value["text"], str):
                                st.markdown(value["text"])
                            else:
                                st.markdown("## Key Topics & Concepts")
                                st.markdown("Keywords extracted from the text:")
                        
                        with col2:
                            if visualization_enabled and "data" in value:
                                keyword_cloud = create_keyword_cloud(value["data"])
                                st.plotly_chart(keyword_cloud, use_container_width=True)
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
                            # Use improved question answering function
                            formatted_answer = answer_question(
                                st.session_state.clean_text, 
                                question, 
                                model=model_choice
                            )
                            
                            # Display answer in a nice format
                            st.markdown("### Answer")
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
