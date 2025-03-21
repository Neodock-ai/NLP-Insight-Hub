import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re
import numpy as np

def create_sentiment_chart(sentiment_data):
    """
    Creates a comprehensive sentiment visualization dashboard with multiple charts.
    
    Args:
        sentiment_data: Can be a string, basic dict, or enhanced dict with aspects,
                       confidence scores, and key phrases
    
    Returns:
        A Plotly figure object
    """
    # Extract sentiment data based on input type
    if isinstance(sentiment_data, dict):
        # Enhanced dictionary format
        if 'sentiment' in sentiment_data:
            sentiment = sentiment_data['sentiment']
            scores = sentiment_data.get('scores', {})
            confidence = sentiment_data.get('confidence', 0.8)
            aspects = sentiment_data.get('aspects', {})
            key_phrases = sentiment_data.get('key_phrases', [])
        else:
            # Basic dictionary
            scores = sentiment_data
            sentiment = max(scores, key=scores.get) if scores else "Neutral"
            confidence = max(scores.values()) if scores else 0.8
            aspects = {}
            key_phrases = []
    elif isinstance(sentiment_data, str):
        # String input
        sentiment = sentiment_data.strip().lower()
        
        # Convert string to sentiment classification
        if "positive" in sentiment:
            sentiment = "Positive"
            scores = {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
        elif "negative" in sentiment:
            sentiment = "Negative"
            scores = {"positive": 0.1, "neutral": 0.2, "negative": 0.7}
        else:
            sentiment = "Neutral"
            scores = {"positive": 0.25, "neutral": 0.5, "negative": 0.25}
            
        confidence = max(scores.values())
        aspects = {}
        key_phrases = []
    else:
        # Default fallback
        sentiment = "Neutral"
        scores = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
        confidence = 0.34
        aspects = {}
        key_phrases = []
    
    # Standardize sentiment to title case
    sentiment = sentiment.title()
    
    # Calculate sentiment score from -1 to 1 for the gauge
    pos_score = scores.get("positive", 0)
    neg_score = scores.get("negative", 0)
    sentiment_score = pos_score - neg_score
    
    # Create the dashboard using subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "pie"}],
            [{"type": "bar", "colspan": 2}, None]
        ],
        column_widths=[0.7, 0.3],
        row_heights=[0.6, 0.4],
        subplot_titles=["Overall Sentiment", "Sentiment Distribution", "Aspect Analysis"]
    )
    
    # 1. Main Gauge Chart (top left)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score,
            title={"text": f"Sentiment: {sentiment}", "font": {"size": 24}},
            gauge={
                "axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": "darkblue"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [-1, -0.5], "color": "firebrick"},
                    {"range": [-0.5, -0.2], "color": "tomato"},
                    {"range": [-0.2, 0.2], "color": "lightgray"},
                    {"range": [0.2, 0.5], "color": "palegreen"},
                    {"range": [0.5, 1], "color": "forestgreen"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": sentiment_score
                }
            },
            number={
                "font": {"color": "white", "size": 20},
                "suffix": "",
                "valueformat": ".2f"
            },
            delta={
                "reference": 0,
                "position": "bottom",
                "valueformat": ".2f"
            }
        ),
        row=1, col=1
    )
    
    # 2. Pie Chart (top right)
    # Ensure we have standardized score keys
    standard_scores = {
        "Positive": scores.get("positive", 0),
        "Neutral": scores.get("neutral", 0),
        "Negative": scores.get("negative", 0)
    }
    
    fig.add_trace(
        go.Pie(
            labels=list(standard_scores.keys()),
            values=list(standard_scores.values()),
            textinfo="label+percent",
            hoverinfo="label+percent",
            marker={
                "colors": ["forestgreen", "lightgray", "firebrick"],
                "line": {"color": "rgba(0,0,0,0.3)", "width": 1}
            },
            hole=0.4,
            textposition="inside",
            insidetextorientation="radial"
        ),
        row=1, col=2
    )
    
    # 3. Aspect Analysis Bar Chart (bottom)
    if aspects:
        # Sort aspects by absolute value
        sorted_aspects = sorted(
            aspects.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Take top 6 aspects
        top_aspects = sorted_aspects[:min(6, len(sorted_aspects))]
        
        # Create data for bar chart
        aspect_names = [aspect[0].capitalize() for aspect in top_aspects]
        aspect_scores = [aspect[1] for aspect in top_aspects]
        
        # Determine colors based on sentiment
        bar_colors = [
            "forestgreen" if score > 0 else "firebrick"
            for score in aspect_scores
        ]
        
        fig.add_trace(
            go.Bar(
                x=aspect_names,
                y=aspect_scores,
                marker_color=bar_colors,
                text=[f"{abs(score):.2f}" for score in aspect_scores],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Sentiment: %{y:.2f}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Update y-axis title for aspect chart
        fig.update_yaxes(
            title_text="Sentiment Score",
            range=[-1, 1],
            zeroline=True,
            zerolinecolor="white",
            zerolinewidth=1,
            row=2, col=1
        )
    else:
        # Display message when no aspects available
        fig.add_annotation(
            text="No aspect-level sentiment data available",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="white"),
            row=2, col=1
        )
    
    # Add confidence annotation
    confidence_text = f"Confidence: {confidence:.0%}"
    fig.add_annotation(
        text=confidence_text,
        xref="x domain",
        yref="y domain",
        x=0.12,
        y=-0.15,
        showarrow=False,
        font=dict(size=14, color="white"),
        row=1, col=1
    )
    
    # Add key phrases if available
    if key_phrases:
        phrase_text = "<b>Key Sentiment Phrases:</b><br>"
        for i, phrase_data in enumerate(key_phrases[:3]):
            phrase = phrase_data.get("phrase", "")
            phrase_sentiment = phrase_data.get("sentiment", "")
            
            if phrase_sentiment.lower() == "positive":
                emoji = "✓ "
                color = "green"
            elif phrase_sentiment.lower() == "negative":
                emoji = "✗ "
                color = "red"
            else:
                emoji = "• "
                color = "white"
                
            phrase_text += f"{emoji}<span style='color:{color}'>\"{phrase}\"</span><br>"
            
        fig.add_annotation(
            text=phrase_text,
            xref="paper",
            yref="paper",
            x=0.99,
            y=0.02,
            showarrow=False,
            font=dict(size=12, color="white"),
            align="right",
            xanchor="right",
            yanchor="bottom"
        )
    
    # Update overall layout
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white', 'family': 'Arial'},
        margin=dict(l=20, r=20, t=80, b=80),
        height=500,
        title={
            "text": "Sentiment Analysis Dashboard",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24}
        }
    )
    
    # Add explanatory annotation
    fig.add_annotation(
        text=f"Scale: -1 (Very Negative) to +1 (Very Positive)",
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.01,
        showarrow=False,
        font=dict(size=10, color="lightgray"),
        align="left"
    )
    
    return fig

def create_keyword_cloud(keywords_data):
    """
    Creates an advanced visualization of keywords with theme grouping and
    category-based coloring for enhanced insights.
    
    Args:
        keywords_data: String of comma-separated keywords, a list of keywords,
                      or a list of tuples (keyword, weight, category, theme)
    
    Returns:
        A Plotly figure object
    """
    # Process the keywords data
    keyword_dict = {}
    themes = {}
    categories = {"entity": [], "phrase": [], "concept": [], "technical": []}
    
    if isinstance(keywords_data, list):
        if keywords_data and isinstance(keywords_data[0], tuple):
            if len(keywords_data[0]) >= 4:  # Enhanced format with category and theme
                for kw, weight, category, theme in keywords_data:
                    keyword_dict[kw] = weight
                    if theme:
                        if theme not in themes:
                            themes[theme] = []
                        themes[theme].append((kw, weight))
                    if category in categories:
                        categories[category].append(kw)
            elif len(keywords_data[0]) >= 3:  # With category
                for kw, weight, category, *rest in keywords_data:
                    keyword_dict[kw] = weight
                    if category in categories:
                        categories[category].append(kw)
            else:  # Basic (keyword, weight) tuples
                keyword_dict = {kw: weight for kw, weight in keywords_data}
        else:
            # Simple list of keywords, assign decreasing weights
            keyword_dict = {kw: 10.0 - (0.1 * i) for i, kw in enumerate(keywords_data)}
    elif isinstance(keywords_data, str):
        # Split by commas and clean up
        kw_list = [k.strip() for k in keywords_data.split(',') if k.strip()]
        # Assign varying weights
        keyword_dict = {}
        for i, k in enumerate(kw_list):
            # Create varying weights between 9.6 and 10.4
            base = 10.0
            position_factor = (len(kw_list) - i) / len(kw_list) * 0.4
            random_factor = np.random.uniform(-0.2, 0.2)
            keyword_dict[k] = base + position_factor + random_factor
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
    
    # Add category information if available
    if any(len(cat) > 0 for cat in categories.values()):
        df['category'] = 'concept'  # Default category
        
        for cat, words in categories.items():
            for word in words:
                if word in df['keyword'].values:
                    df.loc[df['keyword'] == word, 'category'] = cat
    else:
        df['category'] = 'concept'  # Default category
    
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
    
    # Visualize keywords with advanced features
    fig = go.Figure()
    
    # Define colors by category
    category_colors = {
        'entity': 'rgb(60, 145, 230)',      # Blue for entities
        'phrase': 'rgb(80, 170, 120)',      # Green for phrases
        'technical': 'rgb(180, 90, 180)',   # Purple for technical terms
        'concept': 'rgb(240, 150, 50)'      # Orange for concepts
    }
    
    # Add different traces by category
    for category in df['category'].unique():
        category_df = df[df['category'] == category].tail(15)  # Top 15 per category
        
        if len(category_df) > 0:
            fig.add_trace(go.Bar(
                y=category_df['keyword'],
                x=category_df['weight'],
                orientation='h',
                name=category.capitalize(),
                marker=dict(
                    color=category_colors.get(category, 'rgb(100, 100, 100)'),
                    opacity=0.8
                ),
                text=category_df['weight'],
                texttemplate='%{text:.1f}',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Weight: %{x:.2f}<br>Type: ' + category.capitalize() + '<extra></extra>'
            ))
    
    # If we have theme data, add theme annotations
    if themes:
        annotations = []
        for i, (theme, theme_keywords) in enumerate(themes.items()):
            if theme and theme != "General" and len(theme_keywords) >= 2:
                # Sort theme keywords by weight
                sorted_theme_keywords = sorted(theme_keywords, key=lambda x: x[1], reverse=True)
                
                # Format theme label with keywords
                theme_keywords_text = ', '.join([kw for kw, _ in sorted_theme_keywords[:3]])
                if len(sorted_theme_keywords) > 3:
                    theme_keywords_text += f" +{len(sorted_theme_keywords) - 3} more"
                    
                # Add theme annotation
                annotations.append(dict(
                    x=0,
                    y=-0.15 - (i * 0.08),
                    xref="paper",
                    yref="paper",
                    text=f"<b>{theme} theme:</b> {theme_keywords_text}",
                    showarrow=False,
                    font=dict(color='white', size=12)
                ))
        
        # Add the annotations to the figure
        fig.update_layout(annotations=annotations)
    
    # Create legend title for category explanation
    fig.update_layout(
        title="Key Topics & Concepts",
        yaxis_title="",
        xaxis_title="Relevance Score",
        legend_title="Keyword Type",
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white', 'family': 'Arial'},
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)',
            range=[df['weight'].min() - 0.1, df['weight'].max() + 0.3]
        ),
        yaxis=dict(
            showgrid=False,
            categoryorder='total ascending'
        ),
        # Add extra space at bottom for theme annotations
        margin=dict(l=10, r=30, t=50, b=100 + 20 * min(len(themes), 4)),
        # Adjust height based on number of keywords
        height=max(350, min(600, 100 + 30 * len(df.tail(15)))),
        # Position legend at top to save space
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig
