# app/components/charts.py
# FeedbackIQ — Reusable Plotly Chart Functions

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from styles.theme import CHART_COLORS, PLOTLY_TEMPLATE


def sentiment_donut(pos_count: int, neg_count: int, title: str = "") -> go.Figure:
    """Donut chart showing positive vs negative split."""
    fig = go.Figure(go.Pie(
        labels=['Positive', 'Negative'],
        values=[pos_count, neg_count],
        hole=0.65,
        marker=dict(
            colors=[CHART_COLORS['positive'], CHART_COLORS['negative']],
            line=dict(color='white', width=2)
        ),
        textinfo='percent',
        textfont=dict(size=13),
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>'
    ))

    total = pos_count + neg_count
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:11px;color:#6B7280'>Reviews</span>",
        x=0.5, y=0.5,
        font=dict(size=16, color='#111827'),
        showarrow=False
    )

    layout = dict(
        template=PLOTLY_TEMPLATE,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        margin=dict(t=30, b=30, l=10, r=10),
        height=320,
    )

    if title:
        layout['title'] = dict(text=title, font=dict(size=14, color='#111827'))

    fig.update_layout(**layout)
    return fig


def rating_bar(rating_counts: pd.Series) -> go.Figure:
    """Horizontal bar chart for rating distribution."""
    colors = CHART_COLORS['scale']
    fig = go.Figure()

    for i, (rating, count) in enumerate(rating_counts.items()):
        fig.add_trace(go.Bar(
            y=[f"{'⭐' * int(rating)} {int(rating)} Star"],
            x=[count],
            orientation='h',
            marker_color=colors[int(rating) - 1],
            text=f"{count:,}",
            textposition='outside',
            hovertemplate=f"<b>{int(rating)} Stars</b><br>Reviews: {count:,}<extra></extra>",
            name=f"{int(rating)} Star"
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        height=280,
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(showgrid=True, gridcolor='#F3F4F6', title=''),
        yaxis=dict(showgrid=False, title=''),
        bargap=0.3,
        plot_bgcolor='white'
    )
    return fig


def sentiment_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart showing sentiment volume over time."""
    year_data = df.dropna(subset=['year'])
    year_sentiment = year_data.groupby(
        ['year', 'predicted_sentiment']
    ).size().reset_index(name='count')

    fig = go.Figure()

    for sentiment, color in [
        ('Positive', CHART_COLORS['positive']),
        ('Negative', CHART_COLORS['negative'])
    ]:
        data = year_sentiment[year_sentiment['predicted_sentiment'] == sentiment]
        fig.add_trace(go.Scatter(
            x=data['year'],
            y=data['count'],
            name=sentiment,
            mode='lines+markers',
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
            hovertemplate=f"<b>{sentiment}</b><br>Year: %{{x}}<br>Reviews: %{{y:,}}<extra></extra>"
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=320,
        margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5),
        xaxis=dict(showgrid=False, title=''),
        yaxis=dict(showgrid=True, gridcolor='#F3F4F6', title='Reviews'),
        hovermode='x unified',
        plot_bgcolor='white'
    )
    return fig


def confidence_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of prediction confidence scores."""
    fig = go.Figure()

    for sentiment, color in [
        ('Positive', CHART_COLORS['positive']),
        ('Negative', CHART_COLORS['negative'])
    ]:
        subset = df[df['predicted_sentiment'] == sentiment]['confidence']
        fig.add_trace(go.Histogram(
            x=subset,
            name=sentiment,
            marker_color=color,
            opacity=0.75,
            nbinsx=25,
            hovertemplate=f"<b>{sentiment}</b><br>Confidence: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>"
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode='overlay',
        height=300,
        margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
        xaxis=dict(title='Confidence Score', showgrid=False),
        yaxis=dict(title='Count', showgrid=True, gridcolor='#F3F4F6'),
        plot_bgcolor='white'
    )
    return fig


def sentiment_by_rating(df: pd.DataFrame) -> go.Figure:
    """Grouped bar showing sentiment split per star rating."""
    rating_sentiment = df.groupby(
        ['Score', 'predicted_sentiment']
    ).size().reset_index(name='count')

    fig = go.Figure()
    for sentiment, color in [
        ('Positive', CHART_COLORS['positive']),
        ('Negative', CHART_COLORS['negative'])
    ]:
        data = rating_sentiment[rating_sentiment['predicted_sentiment'] == sentiment]
        fig.add_trace(go.Bar(
            x=data['Score'].astype(str) + ' ⭐',
            y=data['count'],
            name=sentiment,
            marker_color=color,
            hovertemplate=f"<b>{sentiment}</b><br>Rating: %{{x}}<br>Count: %{{y:,}}<extra></extra>"
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode='group',
        height=320,
        margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
        xaxis=dict(title='Star Rating', showgrid=False),
        yaxis=dict(title='Reviews', showgrid=True, gridcolor='#F3F4F6'),
        plot_bgcolor='white',
        bargap=0.2
    )
    return fig


def model_comparison_bar(baseline: dict, distilbert: dict) -> go.Figure:
    """Grouped bar comparing baseline vs DistilBERT metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    b_vals = [
        baseline['accuracy'],
        baseline['precision'],
        baseline['recall'],
        baseline['f1']
    ]
    d_vals = [
        distilbert['accuracy'],
        distilbert['precision'],
        distilbert['recall'],
        distilbert['f1']
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='TF-IDF + Logistic Regression',
        x=metrics,
        y=b_vals,
        marker_color='#94A3B8',
        text=[f'{v:.4f}' for v in b_vals],
        textposition='outside',
        hovertemplate='<b>Baseline</b><br>%{x}: %{y:.4f}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='Fine-tuned DistilBERT',
        x=metrics,
        y=d_vals,
        marker_color=CHART_COLORS['primary'],
        text=[f'{v:.4f}' for v in d_vals],
        textposition='outside',
        hovertemplate='<b>DistilBERT</b><br>%{x}: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode='group',
        height=380,
        margin=dict(t=40, b=20, l=10, r=10),
        yaxis=dict(
            range=[0.85, 1.0],
            title='Score',
            showgrid=True,
            gridcolor='#F3F4F6'
        ),
        xaxis=dict(showgrid=False),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5
        ),
        plot_bgcolor='white'
    )
    return fig


def training_curves(history: dict) -> go.Figure:
    """Line chart showing training and validation loss/accuracy."""
    fig = go.Figure()

    epochs = history['epoch']

    # Loss curves
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['train_loss'],
        name='Train Loss',
        mode='lines+markers',
        line=dict(color='#2563EB', width=2),
        marker=dict(size=7)
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['val_loss'],
        name='Val Loss',
        mode='lines+markers',
        line=dict(color='#DC2626', width=2, dash='dash'),
        marker=dict(size=7)
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=320,
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(title='Epoch', tickvals=epochs, showgrid=False),
        yaxis=dict(title='Loss', showgrid=True, gridcolor='#F3F4F6'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified',
        plot_bgcolor='white'
    )
    return fig


def accuracy_curves(history: dict) -> go.Figure:
    """Line chart showing training and validation accuracy."""
    fig = go.Figure()
    epochs = history['epoch']

    fig.add_trace(go.Scatter(
        x=epochs,
        y=[v * 100 for v in history['train_acc']],
        name='Train Accuracy',
        mode='lines+markers',
        line=dict(color='#16A34A', width=2),
        marker=dict(size=7)
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[v * 100 for v in history['val_acc']],
        name='Val Accuracy',
        mode='lines+markers',
        line=dict(color='#D97706', width=2, dash='dash'),
        marker=dict(size=7)
    ))
    fig.add_hline(
        y=92.12,
        line_dash='dot',
        line_color='#94A3B8',
        annotation_text='Baseline (92.12%)',
        annotation_position='right'
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=320,
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(title='Epoch', tickvals=epochs, showgrid=False),
        yaxis=dict(
            title='Accuracy (%)',
            showgrid=True,
            gridcolor='#F3F4F6',
            range=[88, 100]
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified',
        plot_bgcolor='white'
    )
    return fig