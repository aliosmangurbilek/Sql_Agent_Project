"""
Chart and visualization components
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


def create_performance_chart() -> Optional[go.Figure]:
    """Create performance chart from query history"""
    if st.session_state.query_history:
        df = pd.DataFrame(st.session_state.query_history)
        
        # Time-based response time chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df['duration'],
            mode='lines+markers',
            name='Yanıt Süresi',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#667eea')
        ))
        
        fig.update_layout(
            title="📈 Sorgu Performansı",
            xaxis_title="Sorgu Sırası",
            yaxis_title="Süre (saniye)",
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    return None


def create_category_chart(category_data):
    """Create category distribution chart"""
    if category_data:
        df_cat = pd.DataFrame(category_data)
        fig_cat = px.bar(
            df_cat.head(10), 
            x='count', 
            y='category',
            orientation='h',
            title="Top 10 Film Kategorisi",
            color='count',
            color_continuous_scale='viridis'
        )
        fig_cat.update_layout(height=400)
        return fig_cat
    return None


def create_rental_pie_chart(rental_data):
    """Create rental rate distribution pie chart"""
    if rental_data:
        df_rental = pd.DataFrame(rental_data)
        fig_rental = px.pie(
            df_rental,
            values='count',
            names='price_range',
            title="Kira Oranı Dağılımı"
        )
        fig_rental.update_layout(height=400)
        return fig_rental
    return None


def create_actor_chart(actor_data):
    """Create actor productivity chart"""
    if actor_data:
        df_actor = pd.DataFrame(actor_data)
        fig_actor = px.bar(
            df_actor,
            x='film_count',
            y='actor_name',
            orientation='h',
            title="En Produktif Aktörler",
            color='film_count',
            color_continuous_scale='plasma'
        )
        fig_actor.update_layout(height=500)
        return fig_actor
    return None


def create_auto_visualization(df: pd.DataFrame, user_question: str, max_results: int = 50):
    """Create automatic visualization based on data"""
    if len(df.columns) >= 2 and len(df) <= max_results:
        try:
            # Determine chart type
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) >= 1:
                # Bar chart
                if len(df.columns) == 2:
                    fig = px.bar(
                        df, 
                        x=df.columns[0], 
                        y=df.columns[1],
                        title=f"📊 {user_question[:50]}...",
                        color=df.columns[1],
                        color_continuous_scale="viridis"
                    )
                else:
                    # Pie chart for categorical data
                    fig = px.pie(
                        df.head(10), 
                        values=numeric_cols[0], 
                        names=df.columns[0],
                        title=f"🥧 {user_question[:50]}..."
                    )
                
                fig.update_layout(height=400)
                return fig
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
    return None
