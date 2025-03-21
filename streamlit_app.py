import streamlit as st
import logging
import os
from pathlib import Path
import importlib.util

# First check if dependencies are installed
try:
    from pipeline import data_ingestion, pre_processing, inference, post_processing
    from utils.logger import get_logger
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

def main():
    st.set_page_config(
        page_title="NLP Insight Hub",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("NLP Insight Hub")
    st.write("### An Industry-Level AI-Powered NLP Pipeline for Business Insights")
    
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
    
    # Input section
    st.sidebar.write("Upload a text file or paste text below:")
    
    tab1, tab2 = st.tabs(["📄 Upload File", "✏️ Enter Text"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload your text file", type=["txt"])
    
    with tab2:
        text_input = st.text_area("Paste text here:", height=200)
    
    # Process button
    process_button = st.button("Process Text", type="primary")
    
    if (uploaded_file or text_input) and process_button:
        try:
            # Data Ingestion
            if uploaded_file:
                raw_text = data_ingestion.read_file(uploaded_file)
            else:
                if not text_input.strip():
                    st.warning("Please enter some text or upload a file.")
                    st.stop()
                raw_text = text_input

            with st.expander("Original Text", expanded=False):
                st.write(raw_text)
            logger.info("Data ingestion successful.")

            # Pre-Processing
            with st.spinner("Cleaning text..."):
                clean_text = pre_processing.clean_text(raw_text)
            
            with st.expander("Cleaned Text", expanded=False):
                st.write(clean_text)
            logger.info("Text preprocessing successful.")

            # Check if text is too short
            if len(clean_text.split()) < 10:
                st.warning("The text appears to be very short. Results may not be accurate.")

            # Results section
            results_container = st.container()
            results = {}

            # Summarization
            if "Summarization" in task_choices:
                with st.spinner("Generating summary..."):
                    try:
                        summary = inference.get_summary(clean_text, model=model_choice)
                        formatted_summary = post_processing.format_summary(summary)
                        results["Summary"] = formatted_summary
                        logger.info("Summarization completed.")
                    except Exception as e:
                        logger.error(f"Summarization failed: {str(e)}")
                        results["Summary"] = f"Error generating summary: {str(e)}"

            # Sentiment Analysis
            if "Sentiment Analysis" in task_choices:
                with st.spinner("Performing sentiment analysis..."):
                    try:
                        sentiment = inference.get_sentiment(clean_text, model=model_choice)
                        formatted_sentiment = post_processing.format_sentiment(sentiment)
                        results["Sentiment Analysis"] = formatted_sentiment
                        logger.info("Sentiment analysis completed.")
                    except Exception as e:
                        logger.error(f"Sentiment analysis failed: {str(e)}")
                        results["Sentiment Analysis"] = f"Error analyzing sentiment: {str(e)}"

            # Keyword Extraction
            if "Keyword Extraction" in task_choices:
                with st.spinner("Extracting keywords..."):
                    try:
                        keywords = inference.get_keywords(clean_text, model=model_choice)
                        formatted_keywords = post_processing.format_keywords(keywords)
                        results["Keyword Extraction"] = formatted_keywords
                        logger.info("Keyword extraction completed.")
                    except Exception as e:
                        logger.error(f"Keyword extraction failed: {str(e)}")
                        results["Keyword Extraction"] = f"Error extracting keywords: {str(e)}"

            # Q&A Functionality
            if "Q&A" in task_choices:
                qa_container = st.container()
                
                with qa_container:
                    question = st.text_input("Enter your question about the text:")
                    ask_button = st.button("Ask")
                    
                    if question and ask_button:
                        with st.spinner("Processing your question..."):
                            try:
                                answer = inference.get_qa(clean_text, question, model=model_choice)
                                formatted_answer = post_processing.format_qa(answer)
                                results["Q&A"] = formatted_answer
                                logger.info("Q&A processing completed.")
                            except Exception as e:
                                logger.error(f"Q&A processing failed: {str(e)}")
                                results["Q&A"] = f"Error processing question: {str(e)}"

            # Display results
            with results_container:
                if results:
                    st.subheader("Analysis Results")
                    
                    for key, value in results.items():
                        with st.expander(key, expanded=True):
                            st.markdown(value)

        except Exception as ex:
            st.error("An error occurred while processing your text.")
            st.error(f"Error details: {str(ex)}")
            logger.exception("Error in NLP processing", exc_info=ex)
            
            # Provide troubleshooting information
            st.info("Troubleshooting tips: \n"
                   "1. Check if the text format is compatible \n"
                   "2. Try with a smaller text sample \n" 
                   "3. Restart the application")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        # Create a log file on error for easier debugging
        logging.basicConfig(filename='app_error.log', level=logging.ERROR)
        logging.error("Unhandled exception", exc_info=True)
