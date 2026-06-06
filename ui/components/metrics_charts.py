"""
Metrics Charts — PHEME Rumor Detection UI
==========================================
Reusable Plotly chart components for metrics visualization.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_metrics_bar_chart(df, title="Model Performance Comparison"):
    """Create grouped bar chart of Accuracy, Precision, Recall, F1 for all models."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    models = df['Model'].tolist() if 'Model' in df.columns else df.index.tolist()
    
    fig = go.Figure()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=df[metric].values,
                marker_color=colors[i % len(colors)],
                text=[f"{v:.4f}" for v in df[metric].values],
                textposition='outside',
                textfont=dict(size=9)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_recall_fn_chart(df):
    """Create dual-axis chart of Recall and False Negatives."""
    models = df['Model'].tolist() if 'Model' in df.columns else df.index.tolist()
    
    fig = go.Figure()
    
    if 'Recall' in df.columns:
        fig.add_trace(go.Scatter(
            x=models,
            y=df['Recall'].values,
            name='Recall',
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=10),
            yaxis='y'
        ))
    
    if 'FN' in df.columns:
        fig.add_trace(go.Bar(
            x=models,
            y=df['FN'].values,
            name='False Negatives',
            marker_color='rgba(128, 128, 128, 0.4)',
            yaxis='y2'
        ))
    
    fig.update_layout(
        title="Recall vs False Negatives",
        xaxis_title="Model",
        yaxis=dict(title="Recall", color='red', range=[0.5, 1.0]),
        yaxis2=dict(title="False Negatives", overlaying='y', side='right'),
        template='plotly_white',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_feature_dimension_chart():
    """Create horizontal bar chart showing feature dimensions for V1 and V2."""
    categories = ['TF-IDF', 'MiniLM', 'Graph', 'Propagation']
    v1_values = [5000, 384, 14, 0]
    v2_values = [0, 384, 14, 4]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories,
        x=v1_values,
        name='V1 (Post-Level)',
        orientation='h',
        marker_color='#2E86AB',
        text=[str(v) if v > 0 else '' for v in v1_values],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        y=categories,
        x=v2_values,
        name='V2 (Thread-Level)',
        orientation='h',
        marker_color='#6A994E',
        text=[str(v) if v > 0 else '' for v in v2_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Dimension Comparison",
        xaxis_title="Dimension Count",
        barmode='group',
        template='plotly_white',
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_pie_chart(prediction, confidence):
    """Create a pie chart for prediction confidence."""
    label = "Rumor" if prediction == 1 else "Non-Rumor"
    other_prob = 1.0 - confidence
    
    fig = go.Figure(data=[go.Pie(
        labels=[label, 'Other'],
        values=[confidence, other_prob],
        marker_colors=['#C73E1D' if label == 'Rumor' else '#2E86AB', '#E0E0E0'],
        textinfo='label+percent',
        hole=0.4
    )])
    fig.update_layout(
        title=f"Prediction: {label}",
        height=300,
        template='plotly_white',
        showlegend=False
    )
    return fig


def create_confusion_matrix(cm, labels=['Non-Rumor', 'Rumor']):
    """Create confusion matrix heatmap."""
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=labels[j],
                y=labels[i],
                text=f"{cm[i,j]:,}<br>({cm_norm[i,j]:.1%})",
                font=dict(color='white' if cm_norm[i,j] > 0.5 else 'black', size=14),
                showarrow=False
            ))
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=False,
        zmin=0,
        zmax=1
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        xaxis=dict(side='bottom'),
        height=350,
        template='plotly_white',
        annotations=annotations
    )
    return fig