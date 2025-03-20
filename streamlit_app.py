import subprocess
subprocess.run(["bash", "setup.sh"], check=True)
import streamlit as st
import logging
from pipeline import data_ingestion, pre_processing, inference, post_processing
from utils.logger import get_logger

# Configure logger for the app
logger = get_logger("NLPInsightHub")

def main():
    st.title("NLP Insight Hub")
    st.write("### An Industry-Level AI-Powered NLP Pipeline for Business Insights")
    
    # Sidebar configuration for model and tasks
    st.sidebar.header("Configuration")
    model_choice = st.sidebar.selectbox("Select Model", ["Llama", "Falcon", "Mistral", "DeepSeek"])
    task_choices = st.sidebar.multiselect(
        "Select Tasks",
        ["Summarization", "Sentiment Analysis", "Keyword Extraction", "Q&A"],
        default=["Summarization", "Sentiment Analysis"]
    )
    st.sidebar.write("Upload a text file or paste text below:")

    # Input: file uploader and text area
    uploaded_file = st.file_uploader("Upload your text file", type=["txt"])
    text_input = st.text_area("Or paste text here:")

    if uploaded_file or text_input:
        try:
            # Data Ingestion
            if uploaded_file:
                raw_text = data_ingestion.read_file(uploaded_file)
            else:
                raw_text = text_input

            st.subheader("Original Text")
            st.write(raw_text)
            logger.info("Data ingestion successful.")

            # Pre-Processing
            clean_text = pre_processing.clean_text(raw_text)
            st.subheader("Cleaned Text")
            st.write(clean_text)
            logger.info("Text preprocessing successful.")

            # Inference & Post-Processing
            results = {}

            # Summarization
            if "Summarization" in task_choices:
                with st.spinner("Generating summary..."):
                    summary = inference.get_summary(clean_text, model=model_choice)
                    formatted_summary = post_processing.format_summary(summary)
                    results["Summary"] = formatted_summary
                    logger.info("Summarization completed.")

            # Sentiment Analysis
            if "Sentiment Analysis" in task_choices:
                with st.spinner("Performing sentiment analysis..."):
                    sentiment = inference.get_sentiment(clean_text, model=model_choice)
                    formatted_sentiment = post_processing.format_sentiment(sentiment)
                    results["Sentiment Analysis"] = formatted_sentiment
                    logger.info("Sentiment analysis completed.")

            # Keyword Extraction
            if "Keyword Extraction" in task_choices:
                with st.spinner("Extracting keywords..."):
                    keywords = inference.get_keywords(clean_text, model=model_choice)
                    formatted_keywords = post_processing.format_keywords(keywords)
                    results["Keyword Extraction"] = formatted_keywords
                    logger.info("Keyword extraction completed.")

            # Q&A Functionality
            if "Q&A" in task_choices:
                question = st.text_input("Enter your question about the text:")
                if question:
                    with st.spinner("Processing your question..."):
                        answer = inference.get_qa(clean_text, question, model=model_choice)
                        formatted_answer = post_processing.format_qa(answer)
                        results["Q&A"] = formatted_answer
                        logger.info("Q&A processing completed.")

            # Display results
            st.subheader("Results")
            for key, value in results.items():
                st.markdown(f"### {key}")
                st.write(value)
                st.markdown("---")

        except Exception as ex:
            st.error("An error occurred while processing your text. Please try again or contact support.")
            logger.exception("Error in NLP processing", exc_info=ex)

if __name__ == "__main__":
    main()
