# models/openai_gpt.py

import openai
import re

class OpenAIGPTModel:
    """
    A model class that uses OpenAI's GPT (via openai.ChatCompletion)
    to perform summarization, sentiment, keyword extraction, etc.
    with a chunk-based approach.
    """

    def __init__(self):
        # By default, we set no API key. We'll have a setter for it.
        self.api_key = None
        # You can pick a default model name (e.g. 'gpt-3.5-turbo' or 'gpt-4'):
        self.default_model = "gpt-3.5-turbo"
        self.max_chars_per_chunk = 3000  # adjust as needed

    def set_api_key(self, api_key: str):
        """
        Save the user-supplied API key locally and in openai.api_key.
        """
        self.api_key = api_key
        openai.api_key = api_key

    ########################################################################
    # Utility for chunking large text
    ########################################################################
    def _chunk_text(self, text):
        """
        Splits text into smaller chunks of ~max_chars_per_chunk characters.
        """
        chunks = []
        current = []
        current_len = 0

        paragraphs = text.split("\n\n")
        for p in paragraphs:
            if current_len + len(p) > self.max_chars_per_chunk:
                chunks.append("\n\n".join(current))
                current = [p]
                current_len = len(p)
            else:
                current.append(p)
                current_len += len(p)

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    ########################################################################
    # Summarization
    ########################################################################
    def _gpt_summarize_chunk(self, chunk, model_name=None):
        if not model_name:
            model_name = self.default_model

        prompt = f"""You are an expert text summarizer. Summarize the following text
in a concise but thorough way, highlighting key points and important details.

Text:
{chunk}

Summary:
"""
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp['choices'][0]['message']['content'].strip()

    def generate_text(self, prompt):
        """
        This method is used by your pipeline for *any* task. 
        We'll interpret the 'prompt' to see if it's summarization, sentiment, etc.
        But in your existing code, we have separate functions, so we might skip using `generate_text` directly.
        """
        # For clarity, we'll just return the prompt. 
        # (We won't rely on this method for the chunk-based approach.)
        return "This method isn't used for chunk-based tasks in openai_gpt.py."

    def summarize(self, text, model_name=None):
        # 1) Chunk the text
        chunks = self._chunk_text(text)
        partial_summaries = []

        # 2) Summarize each chunk
        for c in chunks:
            partial_summaries.append(self._gpt_summarize_chunk(c, model_name))

        # 3) Summarize the partial summaries into a final summary
        combined_text = "\n\n".join(partial_summaries)
        final_summary = self._gpt_summarize_chunk(combined_text, model_name)
        return final_summary

    ########################################################################
    # Sentiment
    ########################################################################
    def _gpt_sentiment_chunk(self, chunk, model_name=None):
        if not model_name:
            model_name = self.default_model

        prompt = f"""Analyze the sentiment of the following text.
Decide if it is primarily Positive, Negative, or Neutral, with a short explanation.

Text:
{chunk}

Sentiment:
"""
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp['choices'][0]['message']['content'].strip()

    def sentiment(self, text, model_name=None):
        chunks = self._chunk_text(text)
        sentiments = []

        for c in chunks:
            chunk_sent = self._gpt_sentiment_chunk(c, model_name)
            sentiments.append(chunk_sent)

        # Very simplistic final classification
        pos = sum("positive" in s.lower() for s in sentiments)
        neg = sum("negative" in s.lower() for s in sentiments)
        neu = sum("neutral" in s.lower() for s in sentiments)

        if pos > neg and pos > neu:
            overall = "Positive"
        elif neg > pos and neg > neu:
            overall = "Negative"
        else:
            overall = "Neutral"

        return {
            "sentiment": overall,
            "details": sentiments
        }

    ########################################################################
    # Keywords
    ########################################################################
    def _gpt_keywords_chunk(self, chunk, model_name=None):
        if not model_name:
            model_name = self.default_model

        prompt = f"""Extract the main keywords, concepts, or entities 
from the following text. Return them as a comma-separated list (no extra text).

Text:
{chunk}

Keywords:
"""
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp['choices'][0]['message']['content'].strip()

    def keywords(self, text, model_name=None):
        chunks = self._chunk_text(text)
        all_keywords = []

        for c in chunks:
            kw_string = self._gpt_keywords_chunk(c, model_name)
            chunk_kws = [k.strip() for k in kw_string.split(",") if k.strip()]
            all_keywords.extend(chunk_kws)

        # Deduplicate
        unique_kws = []
        seen = set()
        for kw in all_keywords:
            low = kw.lower()
            if low not in seen:
                seen.add(low)
                unique_kws.append(kw)
        return unique_kws

    ########################################################################
    # Q&A
    ########################################################################
    def _gpt_qa_chunk(self, chunk, question, model_name=None):
        if not model_name:
            model_name = self.default_model

        prompt = f"""You are a Q&A assistant. Based on the text below, answer the question.
If the text doesn't contain the answer, say so briefly.

Text:
{chunk}

Question:
{question}

Answer:
"""
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp['choices'][0]['message']['content'].strip()

    def qa(self, text, question, model_name=None):
        """
        A simplistic chunk-based approach:
        1) We'll run the question on each chunk
        2) We'll store all chunk answers
        3) Optionally do a final merge 
        """
        chunks = self._chunk_text(text)
        answers = []

        for c in chunks:
            ans = self._gpt_qa_chunk(c, question, model_name)
            answers.append(ans)

        # Combine or pick the "best" answer. 
        # For now, just returns the longest or non-empty.
        best = ""
        for a in answers:
            if len(a) > len(best) and "no answer" not in a.lower() and "doesn't contain" not in a.lower():
                best = a

        if not best.strip():
            best = "No answer found in the text."

        return best
