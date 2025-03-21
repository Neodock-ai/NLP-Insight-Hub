import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re
import numpy as np

def create_sentiment_chart(sentiment_data):
    """
    Creates a sentiment visualization chart.
    
    Args:
        sentiment_data: Can be a string like 'Positive', 'Negative', 'Neutral' 
                       or a dict with sentiment scores
    
    Returns:
        A Plotly figure object
    """
    # Handle dictionary input with detailed scores
    if isinstance(sentiment_data, dict) and 'scores' in sentiment_data:
        scores = sentiment_data['scores']
        values = [
            scores.get('positive', 0),
            scores.get('neutral', 0),
            scores.get('negative', 0)
        ]
        
        # Determine dominant sentiment
        max_val = max(values)
        if values[0] == max_val:
            dominant = "Positive"
        elif values[2] == max_val:
            dominant = "Negative"
        else:
            dominant = "Neutral"
            
    # Handle dictionary input with simple scores    
    elif isinstance(sentiment_data, dict) and 'sentiment' in sentiment_data:
        dominant = sentiment_data['sentiment']
        
        # Generate realistic scores based on the sentiment
        if dominant.lower() == "positive":
            values = [0.7, 0.2, 0.1]  # positive, neutral, negative
        elif dominant.lower() == "negative":
            values = [0.15, 0.25, 0.6]
        else:
            values = [0.25, 0.5, 0.25]
    
    # Handle string input
    elif isinstance(sentiment_data, str):
        sentiment = sentiment_data.lower().strip()
        if "positive" in sentiment:
            values = [0.7, 0.2, 0.1]  # Positive, Neutral, Negative
            dominant = "Positive"
        elif "negative" in sentiment:
            values = [0.15, 0.25, 0.6]  # Positive, Neutral, Negative
            dominant = "Negative"
        else:
            values = [0.25, 0.5, 0.25]  # Positive, Neutral, Negative
            dominant = "Neutral"
    
    # Default fallback
    else:
        values = [0.33, 0.34, 0.33]  # Equal distribution
        dominant = "Mixed"
    
    # Calculate sentiment score from -1 to 1
    sentiment_score = values[0] - values[2]
    
    # Create a gauge chart
    fig = go.Figure()
    
    # Add the gauge
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Sentiment: {dominant}"},
        gauge = {
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.5], 'color': "firebrick"},
                {'range': [-0.5, -0.2], 'color': "tomato"},
                {'range': [-0.2, 0.2], 'color': "lightgray"},
                {'range': [0.2, 0.5], 'color': "palegreen"},
                {'range': [0.5, 1], 'color': "forestgreen"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_score
            }
        },
        number = {'font': {'color': 'white'}, 'suffix': ''}
    ))
    
    # Add a pie chart to show the distribution
    fig.add_trace(go.Pie(
        values=values,
        labels=['Positive', 'Neutral', 'Negative'],
        domain={'x': [0.7, 1], 'y': [0.7, 1]},
        marker={'colors': ['forestgreen', 'lightgray', 'firebrick']},
        textinfo='none',
        hole=.3,
        showlegend=False
    ))
    
    # Improve layout
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white', 'family': 'Arial'},
        height=300,
        margin=dict(l=20, r=30, t=50, b=20),
    )
    
    return fig

def create_keyword_cloud(keywords_data):
    """
    Creates a visualization of keywords.
    
    Args:
        keywords_data: String of comma-separated keywords, a list of keywords,
                      or a list of tuples (keyword, weight)
    
    Returns:
        A Plotly figure object
    """
    # Process the keywords data
    if isinstance(keywords_data, str):
        # Split by commas and clean up
        keywords = [k.strip() for k in keywords_data.split(',') if k.strip()]
        # Assign varying weights
        keyword_dict = {}
        for i, k in enumerate(keywords):
            # Create varying weights between 9.6 and 10.4
            base = 10.0
            position_factor = (len(keywords) - i) / len(keywords) * 0.4
            random_factor = np.random.uniform(-0.2, 0.2)
            keyword_dict[k] = base + position_factor + random_factor
    
    elif isinstance(keywords_data, list):
        # If it's a list of strings
        if all(isinstance(k, str) for k in keywords_data):
            keyword_dict = {}
            for i, k in enumerate(keywords_data):
                # Create varying weights
                base = 10.0
                position_factor = (len(keywords_data) - i) / len(keywords_data) * 0.4
                random_factor = np.random.uniform(-0.2, 0.2)
                keyword_dict[k] = base + position_factor + random_factor
                
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
        keyword_dict = {
            "No keywords": 10.0,
            "found": 9.8,
            "in": 9.7,
            "text": 9.6,
            "sample": 9.5
        }
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'keyword': list(keyword_dict.keys()),
        'weight': list(keyword_dict.values())
    })
    
    # Ensure weight range has sufficient variation (at least 0.5 point spread)
    weight_range = df['weight'].max() - df['weight'].min()
    if weight_range < 0.5 and len(df) > 1:
        # Rescale weights to ensure variation
        min_weight, max_weight = df['weight'].min(), df['weight'].max()
        target_min, target_max = 9.5, 10.4
        
        df['weight'] = df['weight'].apply(
            lambda w: target_min + ((w - min_weight) / (max_weight - min_weight)) * (target_max - target_min)
            if max_weight > min_weight else w
        )
    
    # Sort by weight
    df = df.sort_values('weight', ascending=True)
    
    # Create a horizontal bar chart with gradient color scheme
    fig = px.bar(
        df.tail(15),  # Show only top 15 keywords
        y='keyword',
        x='weight',
        orientation='h',
        title='Key Topics',
        color='weight',
        color_continuous_scale=[
            [0, 'rgb(70, 100, 170)'],
            [0.5, 'rgb(45, 130, 200)'],
            [1, 'rgb(30, 170, 190)']
        ],
        text='weight'
    )
    
    # Format the weight values in the text to 1 decimal place
    fig.update_traces(
        texttemplate='%{text:.1f}',
        textposition='outside',
        cliponaxis=False,
        hovertemplate='<b>%{y}</b><br>Weight: %{x:.2f}<extra></extra>'
    )
    
    # Improve formatting and style
    fig.update_layout(
        yaxis_title="",
        xaxis_title="Relevance",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white', 'family': 'Arial'},
        coloraxis_showscale=False,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)',
            range=[df['weight'].min() - 0.1, df['weight'].max() + 0.3]  # Add padding
        ),
        yaxis=dict(
            showgrid=False,
            categoryorder='total ascending'
        ),
        margin=dict(l=10, r=30, t=50, b=50),
        height=max(300, min(500, 100 + 30 * len(df)))  # Dynamic height based on item count
    )
    
    return fig
