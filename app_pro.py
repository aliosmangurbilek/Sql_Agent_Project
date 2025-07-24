import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import time
from typing import Dict, List, Any
from schema_tools import ask_db, generate_final_answer

# Sayfa yapılandırması
st.set_page_config(
    page_title="🎬 Pagila AI Assistant Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gelişmiş CSS stilleri
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    .query-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .result-card {
        background: linear-gradient(145deg, #d4edda, #c3e6cb);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #a3d9a4;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(40,167,69,0.2);
    }
    
    .error-card {
        background: linear-gradient(145deg, #f8d7da, #f1b0b7);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #f1b0b7;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(220,53,69,0.2);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e9ecef;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #28a745;
        box-shadow: 0 0 10px rgba(40, 167, 69, 0.5);
    }
    
    .status-offline {
        background-color: #dc3545;
        box-shadow: 0 0 10px rgba(220, 53, 69, 0.5);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .query-history {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #6c757d;
        font-size: 0.9rem;
    }
    
    .performance-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state başlatma
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'performance_stats' not in st.session_state:
    st.session_state.performance_stats = {'total_queries': 0, 'avg_response_time': 0, 'success_rate': 100}

def setup_environment():
    """Çevre değişkenlerini ayarla"""
    os.environ["DATABASE_URL"] = "postgresql://postgres:2336@localhost:5432/pagila"
    os.environ["CHAT_MODEL"] = "mistral:7b-instruct"
    os.environ["EMBED_MODEL"] = "mxbai-embed-large"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

def run_async(coro):
    """Async fonksiyonu senkron ortamda çalıştır"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def add_to_history(query: str, result_type: str, duration: float, success: bool):
    """Sorgu geçmişine ekle"""
    st.session_state.query_history.insert(0, {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'query': query[:50] + '...' if len(query) > 50 else query,
        'type': result_type,
        'duration': duration,
        'success': success
    })
    
    # Son 10 sorguyu tut
    if len(st.session_state.query_history) > 10:
        st.session_state.query_history = st.session_state.query_history[:10]
    
    # Performans istatistiklerini güncelle
    st.session_state.performance_stats['total_queries'] += 1
    
    # Ortalama yanıt süresini güncelle
    total_time = st.session_state.performance_stats['avg_response_time'] * (st.session_state.performance_stats['total_queries'] - 1)
    st.session_state.performance_stats['avg_response_time'] = (total_time + duration) / st.session_state.performance_stats['total_queries']
    
    # Başarı oranını güncelle
    successful_queries = sum(1 for h in st.session_state.query_history if h['success'])
    st.session_state.performance_stats['success_rate'] = (successful_queries / len(st.session_state.query_history)) * 100

def create_performance_chart():
    """Performans grafiği oluştur"""
    if st.session_state.query_history:
        df = pd.DataFrame(st.session_state.query_history)
        
        # Zaman bazlı yanıt süresi grafiği
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

# Ana başlık
st.markdown("""
<div class="main-header">
    <h1 class="header-title">🎬 Pagila AI Assistant Pro</h1>
    <p class="header-subtitle">Gelişmiş AI destekli veritabanı sorgulama sistemi</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h3>🔧 Sistem Kontrol Paneli</h3></div>', unsafe_allow_html=True)
    
    setup_environment()
    
    # Sistem durumu
    with st.spinner("Sistem kontrol ediliyor..."):
        try:
            test_result = run_async(ask_db("SELECT COUNT(*) as total_films FROM film LIMIT 1"))
            if test_result:
                db_status = "online"
                total_films = test_result[0]['total_films']
            else:
                db_status = "offline"
                total_films = "N/A"
        except:
            db_status = "offline"
            total_films = "N/A"
    
    # Durum göstergesi
    status_class = "status-online" if db_status == "online" else "status-offline"
    status_text = "🟢 Çevrimiçi" if db_status == "online" else "🔴 Çevrimdışı"
    
    st.markdown(f"""
    <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; margin: 1rem 0;">
        <h4>📊 Sistem Durumu</h4>
        <p><span class="status-indicator {status_class}"></span><strong>Veritabanı:</strong> {status_text}</p>
        <p>📽️ <strong>Toplam Film:</strong> {total_films}</p>
        <p>🤖 <strong>AI Model:</strong> Mistral 7B</p>
        <p>⚡ <strong>Platform:</strong> Ollama</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performans metrikleri
    if st.session_state.performance_stats['total_queries'] > 0:
        st.markdown("### 📈 Performans Metrikleri")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Sorgu", st.session_state.performance_stats['total_queries'])
        with col2:
            st.metric("Başarı Oranı", f"{st.session_state.performance_stats['success_rate']:.1f}%")
        
        st.metric("Ortalama Süre", f"{st.session_state.performance_stats['avg_response_time']:.2f}s")
        
        # Performans grafiği
        perf_chart = create_performance_chart()
        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Sorgu geçmişi
    if st.session_state.query_history:
        st.markdown("### 📜 Son Sorgular")
        for i, query in enumerate(st.session_state.query_history[:5]):
            success_icon = "✅" if query['success'] else "❌"
            st.markdown(f"""
            <div class="query-history">
                <small>{query['timestamp']}</small><br>
                {success_icon} {query['query']}<br>
                <span class="performance-badge">{query['duration']:.2f}s</span>
                <span class="performance-badge">{query['type']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Temizleme butonu
    if st.button("🗑️ Geçmişi Temizle", type="secondary"):
        st.session_state.query_history = []
        st.session_state.performance_stats = {'total_queries': 0, 'avg_response_time': 0, 'success_rate': 100}
        st.rerun()

# Ana içerik
tab1, tab2, tab3 = st.tabs(["🔍 Sorgu Arayüzü", "📊 Dashboard", "🔧 Gelişmiş Araçlar"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🗣️ AI ile Konuşun")
        
        # Gelişmiş sorgu seçenekleri
        query_options = st.expander("⚙️ Gelişmiş Seçenekler", expanded=False)
        with query_options:
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                response_format = st.selectbox(
                    "📋 Yanıt Formatı:",
                    ["Otomatik", "Tablo", "Grafik", "Metin"]
                )
                max_results = st.slider("📊 Maksimum Sonuç:", 10, 1000, 100)
            
            with col_opt2:
                include_sql = st.checkbox("🔍 SQL sorgusunu göster", value=False)
                auto_visualize = st.checkbox("📈 Otomatik görselleştir", value=True)
        
        # Sorgu türü
        query_type = st.radio(
            "🎯 Sorgu Modu:",
            ["🚀 Hızlı SQL", "🤖 AI Analizi", "📝 Hibrit Mod"],
            horizontal=True,
            help="Hızlı: Direkt SQL | AI: Kapsamlı analiz | Hibrit: Her ikisi"
        )
        
        # Önceden tanımlı şablonlar
        templates = {
            "": "Kendi sorunuzu yazın...",
            "Film İstatistikleri": "En popüler film kategorileri ve sayıları nelerdir?",
            "Aktör Analizi": "En çok film çeviren ilk 10 aktör kimdir?", 
            "Gelir Analizi": "Hangi film kategorileri en yüksek gelir getiriyor?",
            "Müşteri İncelemesi": "En aktif müşteriler hangi şehirlerden?",
            "Trend Analizi": "Aylık kira geliri trendi nasıl?",
            "Karşılaştırma": "Drama ve komedi filmlerinin ortalama süreleri",
            "Özel Sorgu": "DRAGON kelimesi geçen filmler hangileri?"
        }
        
        selected_template = st.selectbox("📋 Şablon Seç:", list(templates.keys()))
        
        # Ana sorgu girişi
        user_question = st.text_area(
            "💬 Sorunuzu buraya yazın:",
            value=templates.get(selected_template, ""),
            height=120,
            placeholder="Örnek: 'En uzun film hangisi?' veya 'Drama kategorisinde kaç film var?'"
        )
        
        # Aksiyon butonları
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        
        with col_btn1:
            if st.button("🚀 Çalıştır", type="primary", use_container_width=True):
                if user_question.strip():
                    start_time = time.time()
                    
                    with st.spinner("🤖 AI çalışıyor..."):
                        try:
                            if query_type in ["🚀 Hızlı SQL", "📝 Hibrit Mod"]:
                                # SQL sorgusu
                                result = run_async(ask_db(user_question))
                                
                                end_time = time.time()
                                duration = end_time - start_time
                                
                                if result:
                                    st.markdown(f"""
                                    <div class="result-card">
                                        <h4>✅ Sorgu Başarılı</h4>
                                        <p><strong>📊 Sonuç:</strong> {len(result)} kayıt bulundu</p>
                                        <p><strong>⏱️ Süre:</strong> {duration:.2f} saniye</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Sonuçları DataFrame'e çevir
                                    df = pd.DataFrame(result)
                                    
                                    # Otomatik görselleştirme
                                    if auto_visualize and len(df.columns) >= 2 and len(df) <= 50:
                                        # Grafik türünü belirle
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
                                                # Pie chart kategorik veriler için
                                                fig = px.pie(
                                                    df.head(10), 
                                                    values=numeric_cols[0], 
                                                    names=df.columns[0],
                                                    title=f"🥧 {user_question[:50]}..."
                                                )
                                            
                                            fig.update_layout(height=400)
                                            st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Veri tablosu
                                    st.dataframe(
                                        df.head(max_results), 
                                        use_container_width=True, 
                                        height=min(400, len(df) * 35 + 50)
                                    )
                                    
                                    # İndirme seçenekleri
                                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                                    
                                    with col_dl1:
                                        csv = df.to_csv(index=False)
                                        st.download_button(
                                            "📥 CSV İndir",
                                            csv,
                                            f"pagila_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                            "text/csv",
                                            use_container_width=True
                                        )
                                    
                                    with col_dl2:
                                        json_data = df.to_json(orient='records', indent=2)
                                        st.download_button(
                                            "📋 JSON İndir",
                                            json_data,
                                            f"pagila_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                            "application/json",
                                            use_container_width=True
                                        )
                                    
                                    add_to_history(user_question, "SQL", duration, True)
                                    
                                else:
                                    st.warning("⚠️ Sonuç bulunamadı")
                                    add_to_history(user_question, "SQL", duration, False)
                            
                            # AI analizi (hibrit modda da çalışır)
                            if query_type in ["🤖 AI Analizi", "📝 Hibrit Mod"]:
                                if query_type == "📝 Hibrit Mod":
                                    st.markdown("---")
                                    st.markdown("### 🤖 AI Detaylı Analizi")
                                
                                start_ai = time.time()
                                answer = run_async(generate_final_answer(user_question))
                                end_ai = time.time()
                                ai_duration = end_ai - start_ai
                                
                                st.markdown(f"""
                                <div class="result-card">
                                    <h4>🤖 AI Analizi Tamamlandı</h4>
                                    <p><strong>⏱️ AI Süre:</strong> {ai_duration:.2f} saniye</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(answer)
                                add_to_history(user_question, "AI", ai_duration, True)
                        
                        except Exception as e:
                            duration = time.time() - start_time
                            st.markdown(f"""
                            <div class="error-card">
                                <h4>❌ Hata Oluştu</h4>
                                <p><strong>Hata:</strong> {str(e)}</p>
                                <p><strong>Süre:</strong> {duration:.2f} saniye</p>
                            </div>
                            """, unsafe_allow_html=True)
                            add_to_history(user_question, "Hata", duration, False)
                else:
                    st.warning("⚠️ Lütfen bir soru girin.")
        
        with col_btn2:
            if st.button("🔄 Yenile", use_container_width=True):
                st.rerun()
        
        with col_btn3:
            if st.button("💡 Örnek", use_container_width=True):
                st.session_state['show_examples'] = True
        
        with col_btn4:
            if st.button("📋 Şablonlar", use_container_width=True):
                st.session_state['show_templates'] = True

    with col2:
        # Hızlı istatistikler
        st.markdown("### 📊 Anlık İstatistikler")
        
        try:
            with st.spinner("📈 Yükleniyor..."):
                quick_stats = {
                    "🎬 Film": "SELECT COUNT(*) as count FROM film",
                    "🎭 Aktör": "SELECT COUNT(*) as count FROM actor",
                    "👥 Müşteri": "SELECT COUNT(*) as count FROM customer",
                    "🏷️ Kategori": "SELECT COUNT(*) as count FROM category",
                    "🏪 Mağaza": "SELECT COUNT(*) as count FROM store",
                    "💿 Envanter": "SELECT COUNT(*) as count FROM inventory"
                }
                
                for stat_name, query in quick_stats.items():
                    try:
                        result = run_async(ask_db(query))
                        if result:
                            count = result[0]['count']
                            st.metric(stat_name, f"{count:,}")
                    except:
                        st.metric(stat_name, "Hata")
                        
        except Exception as e:
            st.error(f"İstatistik hatası: {str(e)[:30]}...")

with tab2:
    st.markdown("### 📊 Pagila Veritabanı Dashboard")
    
    # Dashboard metrikleri
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Temel metrikler
        with st.spinner("Dashboard yükleniyor..."):
            # Film sayısı
            film_count = run_async(ask_db("SELECT COUNT(*) as count FROM film"))[0]['count']
            
            # Ortalama kira ücreti
            avg_rental = run_async(ask_db("SELECT ROUND(AVG(rental_rate), 2) as avg FROM film"))[0]['avg']
            
            # En uzun film
            max_length = run_async(ask_db("SELECT MAX(length) as max_len FROM film"))[0]['max_len']
            
            # Aktif müşteri sayısı
            active_customers = run_async(ask_db("SELECT COUNT(*) as count FROM customer WHERE active = 1"))[0]['count']
            
            with col1:
                st.metric("🎬 Toplam Film", f"{film_count:,}")
            with col2:
                st.metric("💰 Ort. Kira", f"${avg_rental}")
            with col3:
                st.metric("⏱️ En Uzun Film", f"{max_length} dk")
            with col4:
                st.metric("👥 Aktif Müşteri", f"{active_customers:,}")
        
        # Grafik bölümü
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Kategori dağılımı
            st.markdown("#### 📊 Film Kategorileri")
            category_data = run_async(ask_db("""
                SELECT c.name as category, COUNT(*) as count 
                FROM film f
                JOIN film_category fc ON f.film_id = fc.film_id
                JOIN category c ON fc.category_id = c.category_id
                GROUP BY c.name
                ORDER BY count DESC
            """))
            
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
                st.plotly_chart(fig_cat, use_container_width=True)
        
        with col_right:
            # Kira oranı dağılımı
            st.markdown("#### 💰 Kira Oranı Dağılımı")
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
            
            if rental_data:
                df_rental = pd.DataFrame(rental_data)
                fig_rental = px.pie(
                    df_rental,
                    values='count',
                    names='price_range',
                    title="Kira Oranı Dağılımı"
                )
                fig_rental.update_layout(height=400)
                st.plotly_chart(fig_rental, use_container_width=True)
        
        # Aktör analizi
        st.markdown("#### 🎭 En Çok Film Çeviren Aktörler")
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
            st.plotly_chart(fig_actor, use_container_width=True)
    
    except Exception as e:
        st.error(f"Dashboard yüklenirken hata: {str(e)}")

with tab3:
    st.markdown("### 🔧 Gelişmiş Araçlar ve Analiz")
    
    # Alt sekmeler
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🔍 Özel Sorgu", "📈 Veri Analizi", "⚙️ Sistem Araçları"])
    
    with sub_tab1:
        st.markdown("#### 🔍 Özel SQL Sorgu Editörü")
        
        # SQL editörü
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
                                # SQL'i doğrudan çalıştır (güvenlik için sadece SELECT)
                                import psycopg2
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
    
    with sub_tab2:
        st.markdown("#### 📈 Gelişmiş Veri Analizi")
        
        # Analiz türü seçimi
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
                        # Film uzunluğu dağılımı
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
                            
                            # İstatistikler
                            st.write("**📊 İstatistiksel Özet:**")
                            st.write(df_length.groupby('length_category').agg({
                                'length': ['mean', 'count'],
                                'rental_rate': 'mean'
                            }).round(2))
                    
                    elif analysis_type == "Kategori Performansı":
                        # Kategori bazlı analiz
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
                            
                            # Çoklu grafik
                            fig = go.Figure()
                            
                            # Film sayısı
                            fig.add_trace(go.Bar(
                                name='Film Sayısı',
                                x=df_perf['category'],
                                y=df_perf['film_count'],
                                yaxis='y',
                                offsetgroup=1
                            ))
                            
                            # Ortalama kira oranı
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
                    
                    # Diğer analiz türleri...
                    else:
                        st.info("Bu analiz türü henüz geliştirilme aşamasında...")
                        
                except Exception as e:
                    st.error(f"Analiz hatası: {str(e)}")
    
    with sub_tab3:
        st.markdown("#### ⚙️ Sistem Araçları")
        
        # Performans test
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
        
        # Sistem bilgileri
        with st.expander("ℹ️ Detaylı Sistem Bilgileri"):
            st.markdown(f"""
            **🗄️ Veritabanı Yapılandırması:**
            - Host: localhost:5432
            - Database: pagila
            - User: postgres
            
            **🤖 AI Model Bilgileri:**
            - Chat Model: {os.environ.get('CHAT_MODEL', 'N/A')}
            - Embedding Model: {os.environ.get('EMBED_MODEL', 'N/A')}
            - Base URL: {os.environ.get('OLLAMA_BASE_URL', 'N/A')}
            
            **📊 Session İstatistikleri:**
            - Toplam Sorgu: {st.session_state.performance_stats['total_queries']}
            - Ortalama Yanıt Süresi: {st.session_state.performance_stats['avg_response_time']:.2f}s
            - Başarı Oranı: {st.session_state.performance_stats['success_rate']:.1f}%
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(45deg, #f8f9fa, #e9ecef); border-radius: 10px; margin-top: 2rem;">
    <h4 style="margin: 0; color: #495057;">🎬 Pagila AI Assistant Pro</h4>
    <p style="margin: 0.5rem 0;">Streamlit & Ollama ile güçlendirilmiş akıllı veritabanı arayüzü</p>
    <p style="margin: 0; font-size: 0.9rem; color: #6c757d;">© 2025 - Gelişmiş AI destekli veri analiz platformu</p>
</div>
""", unsafe_allow_html=True)
