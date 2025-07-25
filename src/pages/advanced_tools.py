"""
Advanced tools and analysis page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import psycopg2
from ..utils.async_helpers import run_async
from ..utils.debug import show_error
from schema_tools import ask_db


def render_advanced_tools():
    """Render advanced tools page"""
    st.markdown("### 🔧 Gelişmiş Araçlar ve Analiz")
    
    # Sub tabs
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🔍 Özel Sorgu", "📈 Veri Analizi", "⚙️ Sistem Araçları"])
    
    with sub_tab1:
        _render_custom_query_editor()
    
    with sub_tab2:
        _render_data_analysis()
    
    with sub_tab3:
        _render_system_tools()


def _render_custom_query_editor():
    """Render custom SQL query editor"""
    st.markdown("#### 🔍 Özel SQL Sorgu Editörü")
    
    # SQL editor
    sql_query = st.text_area(
        "SQL Sorgusu:",
        height=150,
        placeholder="SELECT * FROM film WHERE rental_rate > 2.99 LIMIT 10;",
        help="Dikkat: Sadece SELECT sorguları çalıştırılabilir."
    )
    
    col_sql1, col_sql2 = st.columns([1, 3])
    
    with col_sql1:
        if st.button("🚀 SQL Çalıştır", type="primary"):
            if sql_query.strip():
                if sql_query.strip().upper().startswith('SELECT'):
                    try:
                        with st.spinner("SQL çalıştırılıyor..."):
                            # Execute SQL directly (security: only SELECT)
                            conn = psycopg2.connect(os.environ["DATABASE_URL"])
                            df = pd.read_sql(sql_query, conn)
                            conn.close()
                            
                            st.success(f"✅ {len(df)} kayıt bulundu")
                            st.dataframe(df, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"❌ SQL Hatası: {str(e)}")
                else:
                    st.error("⚠️ Güvenlik nedeniyle sadece SELECT sorguları çalıştırılabilir.")
            else:
                st.warning("⚠️ Lütfen bir SQL sorgusu girin.")


def _render_data_analysis():
    """Render advanced data analysis"""
    st.markdown("#### 📈 Gelişmiş Veri Analizi")
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Analiz Türü:",
        [
            "Film Uzunluğu Analizi",
            "Kira Geliri Analizi", 
            "Kategori Performansı",
            "Aktör-Film İlişkisi",
            "Müşteri Davranış Analizi"
        ]
    )
    
    if st.button("📊 Analizi Başlat"):
        with st.spinner("Analiz yapılıyor..."):
            try:
                if analysis_type == "Film Uzunluğu Analizi":
                    _perform_length_analysis()
                elif analysis_type == "Kategori Performansı":
                    _perform_category_performance_analysis()
                else:
                    st.info("Bu analiz türü henüz geliştirilme aşamasında...")
                    
            except Exception as e:
                st.error(f"Analiz hatası: {str(e)}")


def _perform_length_analysis():
    """Perform film length analysis"""
    length_data = run_async(ask_db("""
        SELECT 
            length,
            rental_rate,
            CASE 
                WHEN length < 90 THEN 'Kısa'
                WHEN length < 120 THEN 'Orta'
                ELSE 'Uzun'
            END as length_category
        FROM film
    """))
    
    if length_data:
        df_length = pd.DataFrame(length_data)
        
        # Scatter plot
        fig = px.scatter(
            df_length,
            x='length',
            y='rental_rate',
            color='length_category',
            title="Film Uzunluğu vs Kira Oranı",
            labels={'length': 'Film Uzunluğu (dk)', 'rental_rate': 'Kira Oranı ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.write("**📊 İstatistiksel Özet:**")
        summary_stats = df_length.groupby('length_category').agg({
            'length': ['mean', 'count'],
            'rental_rate': 'mean'
        }).round(2)
        st.dataframe(summary_stats, use_container_width=True)


def _perform_category_performance_analysis():
    """Perform category performance analysis"""
    perf_data = run_async(ask_db("""
        SELECT 
            c.name as category,
            COUNT(*) as film_count,
            AVG(f.rental_rate) as avg_rental_rate,
            AVG(f.length) as avg_length
        FROM film f
        JOIN film_category fc ON f.film_id = fc.film_id
        JOIN category c ON fc.category_id = c.category_id
        GROUP BY c.name
        ORDER BY film_count DESC
    """))
    
    if perf_data:
        df_perf = pd.DataFrame(perf_data)
        
        # Multi-axis chart
        fig = go.Figure()
        
        # Film count
        fig.add_trace(go.Bar(
            name='Film Sayısı',
            x=df_perf['category'],
            y=df_perf['film_count'],
            yaxis='y',
            offsetgroup=1
        ))
        
        # Average rental rate
        fig.add_trace(go.Scatter(
            name='Ort. Kira Oranı',
            x=df_perf['category'],
            y=df_perf['avg_rental_rate'],
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title='Kategori Performans Analizi',
            xaxis=dict(title='Kategori'),
            yaxis=dict(title='Film Sayısı', side='left'),
            yaxis2=dict(title='Ortalama Kira Oranı ($)', side='right', overlaying='y'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def _render_system_tools():
    """Render system tools"""
    st.markdown("#### ⚙️ Sistem Araçları")
    
    # Performance test
    if st.button("🚀 Performans Testi Çalıştır"):
        with st.spinner("Performans testi yapılıyor..."):
            test_queries = [
                "SELECT COUNT(*) FROM film",
                "SELECT COUNT(*) FROM actor",
                "SELECT COUNT(*) FROM customer"
            ]
            
            results = []
            for query in test_queries:
                start = time.time()
                try:
                    result = run_async(ask_db(query))
                    duration = time.time() - start
                    results.append({
                        'Sorgu': query,
                        'Durum': 'Başarılı' if result else 'Başarısız',
                        'Süre (s)': f"{duration:.3f}"
                    })
                except Exception as e:
                    duration = time.time() - start
                    results.append({
                        'Sorgu': query,
                        'Durum': 'Hata',
                        'Süre (s)': f"{duration:.3f}"
                    })
            
            df_test = pd.DataFrame(results)
            st.dataframe(df_test, use_container_width=True)
    
    # System information
    with st.expander("ℹ️ Detaylı Sistem Bilgileri"):
        st.markdown(f"""
        **🗄️ Veritabanı Yapılandırması:**
        - Host: localhost:5432
        - Database: pagila
        - User: postgres
        
        **🤖 AI Model Bilgileri:**
        - Provider: {os.getenv('AI_PROVIDER', 'ollama').title()}
        - Model: {os.getenv('OLLAMA_MODEL', 'N/A') if os.getenv('AI_PROVIDER') == 'ollama' else os.getenv('OPENAI_MODEL', 'N/A')}
        - Host: {os.getenv('OLLAMA_HOST', 'N/A') if os.getenv('AI_PROVIDER') == 'ollama' else 'External API'}
        
        **📊 Session İstatistikleri:**
        - Toplam Sorgu: {st.session_state.get('performance_stats', {}).get('total_queries', 0)}
        - Ortalama Yanıt Süresi: {st.session_state.get('performance_stats', {}).get('avg_response_time', 0):.2f}s
        - Başarı Oranı: {st.session_state.get('performance_stats', {}).get('success_rate', 100):.1f}%
        """)
