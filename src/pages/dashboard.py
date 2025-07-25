"""
Dashboard page with visualizations and metrics
"""

import streamlit as st
import pandas as pd
from ..utils.async_helpers import run_async
from ..components.charts import create_category_chart, create_rental_pie_chart, create_actor_chart
from schema_tools import ask_db


def render_dashboard():
    """Render the main dashboard"""
    st.markdown("### 📊 Pagila Veritabanı Dashboard")
    
    # Dashboard metrics
    _render_main_metrics()
    
    # Charts section
    col_left, col_right = st.columns(2)
    
    with col_left:
        _render_category_chart()
    
    with col_right:
        _render_rental_chart()
    
    # Actor analysis
    _render_actor_analysis()


def _render_main_metrics():
    """Render main dashboard metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        with st.spinner("Dashboard yükleniyor..."):
            # Film count
            film_count = run_async(ask_db("SELECT COUNT(*) as count FROM film"))[0]['count']
            
            # Average rental rate
            avg_rental = run_async(ask_db("SELECT ROUND(AVG(rental_rate), 2) as avg FROM film"))[0]['avg']
            
            # Longest film
            max_length = run_async(ask_db("SELECT MAX(length) as max_len FROM film"))[0]['max_len']
            
            # Active customers
            active_customers = run_async(ask_db("SELECT COUNT(*) as count FROM customer WHERE active = 1"))[0]['count']
            
            with col1:
                st.metric("🎬 Toplam Film", f"{film_count:,}")
            with col2:
                st.metric("💰 Ort. Kira", f"${avg_rental}")
            with col3:
                st.metric("⏱️ En Uzun Film", f"{max_length} dk")
            with col4:
                st.metric("👥 Aktif Müşteri", f"{active_customers:,}")
                
    except Exception as e:
        st.error(f"Metrik yüklenirken hata: {str(e)}")


def _render_category_chart():
    """Render category distribution chart"""
    st.markdown("#### 📊 Film Kategorileri")
    
    try:
        category_data = run_async(ask_db("""
            SELECT c.name as category, COUNT(*) as count 
            FROM film f
            JOIN film_category fc ON f.film_id = fc.film_id
            JOIN category c ON fc.category_id = c.category_id
            GROUP BY c.name
            ORDER BY count DESC
        """))
        
        fig_cat = create_category_chart(category_data)
        if fig_cat:
            st.plotly_chart(fig_cat, use_container_width=True)
            
    except Exception as e:
        st.error(f"Kategori grafiği yüklenirken hata: {str(e)}")


def _render_rental_chart():
    """Render rental rate distribution chart"""
    st.markdown("#### 💰 Kira Oranı Dağılımı")
    
    try:
        rental_data = run_async(ask_db("""
            SELECT 
                CASE 
                    WHEN rental_rate < 1 THEN '< $1'
                    WHEN rental_rate < 2 THEN '$1-2'
                    WHEN rental_rate < 3 THEN '$2-3'
                    WHEN rental_rate < 4 THEN '$3-4'
                    ELSE '$4+'
                END as price_range,
                COUNT(*) as count
            FROM film
            GROUP BY price_range
            ORDER BY min(rental_rate)
        """))
        
        fig_rental = create_rental_pie_chart(rental_data)
        if fig_rental:
            st.plotly_chart(fig_rental, use_container_width=True)
            
    except Exception as e:
        st.error(f"Kira grafiği yüklenirken hata: {str(e)}")


def _render_actor_analysis():
    """Render actor analysis section"""
    st.markdown("#### 🎭 En Çok Film Çeviren Aktörler")
    
    try:
        actor_data = run_async(ask_db("""
            SELECT 
                a.first_name || ' ' || a.last_name as actor_name,
                COUNT(*) as film_count
            FROM actor a
            JOIN film_actor fa ON a.actor_id = fa.actor_id
            GROUP BY a.actor_id, a.first_name, a.last_name
            ORDER BY film_count DESC
            LIMIT 15
        """))
        
        fig_actor = create_actor_chart(actor_data)
        if fig_actor:
            st.plotly_chart(fig_actor, use_container_width=True)
            
    except Exception as e:
        st.error(f"Aktör analizi yüklenirken hata: {str(e)}")
