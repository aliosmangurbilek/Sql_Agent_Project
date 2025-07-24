import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from schema_tools import ask_db, generate_final_answer

# Streamlit sayfa yapılandırması
st.set_page_config(
    page_title="🎬 Pagila Database AI Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS stilleri
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .query-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .result-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .sidebar-header {
        background: #667eea;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Çevre değişkenlerini ayarlama
def setup_environment():
    """Gerekli çevre değişkenlerini ayarla"""
    os.environ["DATABASE_URL"] = "postgresql://postgres:2336@localhost:5432/pagila"
    os.environ["CHAT_MODEL"] = "mistral:7b-instruct"
    os.environ["EMBED_MODEL"] = "mxbai-embed-large"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

# Async fonksiyonları senkron ortamda çalıştırmak için
def run_async(coro):
    """Async fonksiyonu senkron ortamda çalıştır"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Ana başlık
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🎬 Pagila Database AI Assistant</h1>
    <p style="color: white; margin: 0; font-size: 1.2rem;">Doğal dille veritabanı sorgulama sistemi</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h3>🔧 Sistem Durumu</h3></div>', unsafe_allow_html=True)
    
    setup_environment()
    
    # Sistem durumu kontrolü
    with st.spinner("Sistem bağlantıları kontrol ediliyor..."):
        try:
            # Basit bir test sorgusu
            test_result = run_async(ask_db("SELECT COUNT(*) as total_films FROM film"))
            if test_result:
                st.success("✅ Veritabanı Bağlantısı")
                total_films = test_result[0]['total_films']
                st.info(f"📊 Toplam Film: {total_films}")
            else:
                st.error("❌ Veritabanı Bağlantısı")
        except Exception as e:
            st.error(f"❌ Bağlantı Hatası: {str(e)[:50]}...")
    
    st.markdown("---")
    
    # Örnek sorgular
    st.markdown('<div class="sidebar-header"><h3>💡 Örnek Sorgular</h3></div>', unsafe_allow_html=True)
    
    example_queries = [
        "Veritabanında kaç film var?",
        "En uzun film hangisi?",
        "En popüler film kategorileri",
        "Ortalama kira ücreti nedir?",
        "En çok film çeviren aktör kim?",
        "'DRAGON' kelimesi geçen filmler",
        "En yüksek gelir getiren filmler",
        "Hiç kiralanmamış filmler var mı?"
    ]
    
    selected_example = st.selectbox(
        "Bir örnek seçin:",
        [""] + example_queries,
        help="Bu örneklerden birini seçerek hızlıca test edebilirsiniz"
    )

# Ana içerik alanı
col1, col2 = st.columns([2, 1])

with col1:
    # Sorgu girişi
    st.markdown("### 🗣️ Sorunuzu Sorun")
    
    # Sorgu türü seçimi
    query_type = st.radio(
        "Sorgu Türü:",
        ["🔍 Hızlı SQL Sorgusu", "🤖 Detaylı AI Analizi"],
        horizontal=True,
        help="Hızlı sorgu: Direkt SQL çalıştırır. Detaylı analiz: AI destekli kapsamlı cevap verir."
    )
    
    # Metin girişi
    user_question = st.text_area(
        "Sorunuzu buraya yazın:",
        value=selected_example if selected_example else "",
        height=100,
        placeholder="Örnek: 'En popüler film kategorisi hangisi?' veya 'DRAMA türünde kaç film var?'"
    )
    
    # Sorgula butonu
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("🚀 Sorgula", type="primary", use_container_width=True):
            if user_question.strip():
                with st.spinner("🤖 AI sistemi çalışıyor..."):
                    try:
                        start_time = datetime.now()
                        
                        if query_type == "🔍 Hızlı SQL Sorgusu":
                            # Hızlı SQL sorgusu
                            result = run_async(ask_db(user_question))
                            
                            if result:
                                end_time = datetime.now()
                                duration = (end_time - start_time).total_seconds()
                                
                                st.markdown(f'<div class="query-box"><strong>📋 Sorgu:</strong> {user_question}</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="result-box"><strong>✅ Sonuç:</strong> {len(result)} kayıt bulundu ({duration:.2f} saniye)</div>', unsafe_allow_html=True)
                                
                                # Sonuçları tablo olarak göster
                                if result:
                                    df = pd.DataFrame(result)
                                    
                                    # Sayısal verileri görselleştir
                                    if len(df.columns) == 2 and len(df) <= 20:
                                        # Bar chart oluştur
                                        try:
                                            fig = px.bar(
                                                df, 
                                                x=df.columns[0], 
                                                y=df.columns[1],
                                                title=f"📊 {user_question}",
                                                color=df.columns[1],
                                                color_continuous_scale="viridis"
                                            )
                                            fig.update_layout(height=400, showlegend=False)
                                            st.plotly_chart(fig, use_container_width=True)
                                        except:
                                            pass
                                    
                                    # Tablo gösterimi
                                    st.dataframe(df, use_container_width=True, height=300)
                                    
                                    # CSV indirme
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 CSV olarak indir",
                                        data=csv,
                                        file_name=f"pagila_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                            else:
                                st.warning("⚠️ Sorgu sonucu bulunamadı.")
                                
                        else:
                            # Detaylı AI analizi
                            answer = run_async(generate_final_answer(user_question))
                            
                            end_time = datetime.now()
                            duration = (end_time - start_time).total_seconds()
                            
                            st.markdown(f'<div class="query-box"><strong>📋 Sorgu:</strong> {user_question}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="result-box"><strong>🤖 AI Analizi:</strong> ({duration:.2f} saniye)</div>', unsafe_allow_html=True)
                            
                            st.markdown(answer)
                    
                    except Exception as e:
                        st.markdown(f'<div class="error-box"><strong>❌ Hata:</strong> {str(e)}</div>', unsafe_allow_html=True)
                        st.error("Lütfen sorunuzu yeniden formüle ederek tekrar deneyin.")
            else:
                st.warning("⚠️ Lütfen bir soru girin.")
    
    with col_btn2:
        if st.button("🗑️ Temizle", use_container_width=True):
            st.rerun()

with col2:
    # İstatistikler ve bilgiler
    st.markdown("### 📊 Hızlı İstatistikler")
    
    try:
        # Temel istatistikleri al
        with st.spinner("📈 İstatistikler yükleniyor..."):
            stats_queries = {
                "Toplam Film": "SELECT COUNT(*) as count FROM film",
                "Toplam Aktör": "SELECT COUNT(*) as count FROM actor", 
                "Toplam Müşteri": "SELECT COUNT(*) as count FROM customer",
                "Toplam Kategori": "SELECT COUNT(*) as count FROM category"
            }
            
            stats_data = {}
            for stat_name, query in stats_queries.items():
                try:
                    result = run_async(ask_db(query))
                    if result:
                        stats_data[stat_name] = result[0]['count']
                    else:
                        stats_data[stat_name] = "N/A"
                except:
                    stats_data[stat_name] = "Hata"
            
            # Metrikleri göster
            for stat_name, value in stats_data.items():
                st.metric(stat_name, value)
            
            # Dairesel grafik
            if all(isinstance(v, int) for v in stats_data.values()):
                fig = go.Figure(data=[
                    go.Pie(
                        labels=list(stats_data.keys()),
                        values=list(stats_data.values()),
                        hole=0.4,
                        hovertemplate='<b>%{label}</b><br>%{value}<br>%{percent}<extra></extra>'
                    )
                ])
                fig.update_layout(
                    title="📊 Veritabanı Dağılımı",
                    height=300,
                    showlegend=True,
                    margin=dict(t=50, b=50, l=25, r=25)
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"İstatistik yüklenirken hata: {str(e)[:50]}...")
    
    st.markdown("---")
    
    # Sistem bilgileri
    st.markdown("### ℹ️ Sistem Bilgileri")
    
    with st.expander("🔧 Teknik Detaylar"):
        st.markdown("""
        **🗄️ Veritabanı:** PostgreSQL (Pagila)  
        **🤖 AI Model:** Mistral 7B Instruct  
        **🔍 Embedding:** MXBai Embed Large  
        **⚡ Platform:** Ollama (Yerel)  
        **🎨 Arayüz:** Streamlit  
        """)
    
    with st.expander("📚 Kullanım Kılavuzu"):
        st.markdown("""
        **Nasıl Kullanılır:**
        1. Sorunuzu doğal dilde yazın
        2. Sorgu türünü seçin (Hızlı/Detaylı)
        3. "Sorgula" butonuna tıklayın
        4. Sonuçları inceleyin ve indirin
        
        **💡 İpuçları:**
        - Açık ve spesifik sorular sorun
        - Film, aktör, müşteri tabloları mevcuttur
        - Örnek sorgulardan faydalanın
        """)

# Alt bilgi
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎬 <strong>Pagila Database AI Assistant</strong> - Ollama & Streamlit ile güçlendirilmiştir</p>
    <p>📊 Doğal dil işleme ile veritabanı sorgulama deneyimi</p>
</div>
""", unsafe_allow_html=True)
