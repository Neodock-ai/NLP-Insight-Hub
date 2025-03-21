import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re

def create_sentiment_chart(sentiment_data):
    """
    Creates a sentiment visualization chart.
    
    Args:
        sentiment_data: Can be a string like 'Positive', 'Negative', 'Neutral' 
                       or a dict with sentiment scores
    
    Returns:
        A Plotly figure object
    """
    # Handle string input
    if isinstance(sentiment_data, str):
        sentiment = sentiment_data.lower().strip()
        if "positive" in sentiment:
            values = [0.8, 0.1, 0.1]  # Positive, Neutral, Negative
            dominant = "Positive"
        elif "negative" in sentiment:
            values = [0.1, 0.1, 0.8]  # Positive, Neutral, Negative
            dominant = "Negative"
        else:
            values = [0.1, 0.8, 0.1]  # Positive, Neutral, Negative
            dominant = "Neutral"
    
    # Handle dictionary input
    elif isinstance(sentiment_data, dict) and 'positive' in sentiment_data:
        values = [
            sentiment_data.get('positive', 0),
            sentiment_data.get('neutral', 0),
            sentiment_data.get('negative', 0)
        ]
        # Determine dominant sentiment
        max_val = max(values)
        if values[0] == max_val:
            dominant = "Positive"
        elif values[2] == max_val:
            dominant = "Negative"
        else:
            dominant = "Neutral"
    
    # Default fallback
    else:
        values = [0.33, 0.34, 0.33]  # Equal distribution
        dominant = "Mixed"
    
    # Create a gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = values[0] - values[2],  # Positive minus negative
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [-1, 1], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.3], 'color': "firebrick"},
                {'range': [-0.3, 0.3], 'color': "lightgray"},
                {'range': [0.3, 1], 'color': "forestgreen"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': values[0] - values[2]
            }
        },
        title = {'text': f"Sentiment: {dominant}"}
    ))
    
    return fig

def create_keyword_cloud(keywords_data):
    """
    Creates a visualization of keywords.
    
    Args:
        keywords_data: String of comma-separated keywords or a list of keywords,
                      optionally with weights
    
    Returns:
        A Plotly figure object
    """
    # Process the keywords data
    if isinstance(keywords_data, str):
        # Split by commas and clean up
        keywords = [k.strip() for k in keywords_data.split(',') if k.strip()]
        # Assign default weights
        keyword_dict = {k: 10 for k in keywords}
    
    elif isinstance(keywords_data, list):
        # If it's a list of strings
        if all(isinstance(k, str) for k in keywords_data):
            keyword_dict = {k: 10 for k in keywords_data}
        # If it's a list of tuples (keyword, weight)
        elif all(isinstance(k, tuple) and len(k) == 2 for k in keywords_data):
            keyword_dict = {k[0]: k[1] for k in keywords_data}
        else:
            # Default empty
            keyword_dict = {}
    
    elif isinstance(keywords_data, dict):
        # Already in the right format
        keyword_dict = keywords_data
    
    else:
        # Default empty
        keyword_dict = {}
    
    # Ensure we have at least some data
    if not keyword_dict:
        keyword_dict = {"No keywords": 5}
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'keyword': list(keyword_dict.keys()),
        'weight': list(keyword_dict.values())
    })
    
    # Create a horizontal bar chart
    fig = px.bar(
        df.sort_values('weight', ascending=True),
        y='keyword',
        x='weight',
        orientation='h',
        title='Key Topics',
        color='weight',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis_title="",
        xaxis_title="Relevance",
        showlegend=False
    )
    
    return fig
