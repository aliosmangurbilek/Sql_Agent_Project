"""
Query interface page
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from ..utils.async_helpers import run_async
from ..utils.debug import log_debug, show_error, add_to_history
from ..components.charts import create_auto_visualization
from schema_tools import ask_db, generate_final_answer


def render_query_interface():
    """Render the main query interface"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🗣️ AI ile Konuşun")
        
        # Advanced query options
        _render_advanced_options()
        
        # Query type selection
        query_type = st.radio(
            "🎯 Sorgu Modu:",
            ["🚀 Hızlı SQL", "🤖 AI Analizi", "📝 Hibrit Mod"],
            horizontal=True,
            help="Hızlı: Direkt SQL | AI: Kapsamlı analiz | Hibrit: Her ikisi"
        )
        
        # Predefined templates
        user_question = _render_query_templates()
        
        # Action buttons
        _render_action_buttons(user_question, query_type)

    with col2:
        # Quick statistics
        _render_quick_stats()


def _render_advanced_options():
    """Render advanced query options"""
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
        
        return {
            'response_format': response_format,
            'max_results': max_results,
            'include_sql': include_sql,
            'auto_visualize': auto_visualize
        }


def _render_query_templates():
    """Render query templates and input"""
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
    
    # Main query input
    user_question = st.text_area(
        "💬 Sorunuzu buraya yazın:",
        value=templates.get(selected_template, ""),
        height=120,
        placeholder="Örnek: 'En uzun film hangisi?' veya 'Drama kategorisinde kaç film var?'"
    )
    
    return user_question


def _render_action_buttons(user_question, query_type):
    """Render action buttons"""
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("🚀 Çalıştır", type="primary", use_container_width=True):
            if user_question.strip():
                _execute_query(user_question, query_type)
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


def _execute_query(user_question, query_type):
    """Execute the query based on type"""
    start_time = time.time()
    log_debug("Query execution started", {
        "question": user_question[:100],
        "query_type": query_type
    })
    
    with st.spinner("🤖 AI çalışıyor..."):
        try:
            if query_type in ["🚀 Hızlı SQL", "📝 Hibrit Mod"]:
                _execute_sql_query(user_question, start_time)
            
            # AI analysis (also runs in hybrid mode)
            if query_type in ["🤖 AI Analizi", "📝 Hibrit Mod"]:
                _execute_ai_analysis(user_question, query_type)
                
        except Exception as e:
            show_error(e, "Query execution")
            end_time = time.time()
            duration = end_time - start_time
            add_to_history(user_question, query_type, duration, False)


def _execute_sql_query(user_question, start_time):
    """Execute SQL query"""
    log_debug("Starting SQL query execution")
    result = run_async(ask_db(user_question))
    
    end_time = time.time()
    duration = end_time - start_time
    
    log_debug("SQL query completed", {
        "result_count": len(result) if result else 0,
        "duration": duration
    })
    
    if result:
        st.markdown(f"""
        <div class="result-card">
            <h4>✅ Sorgu Başarılı</h4>
            <p><strong>📊 Sonuç:</strong> {len(result)} kayıt bulundu</p>
            <p><strong>⏱️ Süre:</strong> {duration:.2f} saniye</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Convert results to DataFrame
        try:
            df = pd.DataFrame(result)
            log_debug("DataFrame created successfully", {
                "shape": df.shape,
                "columns": list(df.columns)
            })
            
            # Auto visualization
            auto_viz = create_auto_visualization(df, user_question)
            if auto_viz:
                st.plotly_chart(auto_viz, use_container_width=True)
                log_debug("Visualization created successfully")
            
            # Data table
            st.dataframe(df, use_container_width=True, height=min(400, len(df) * 35 + 50))
            log_debug("Data table displayed successfully")
            
            # Download options
            _render_download_options(df)
            
        except Exception as e:
            show_error(e, "DataFrame creation")
            df = pd.DataFrame()
        
        add_to_history(user_question, "SQL", duration, True)
        
    else:
        st.warning("⚠️ Sonuç bulunamadı")
        log_debug("No results found for SQL query")
        add_to_history(user_question, "SQL", duration, False)


def _execute_ai_analysis(user_question, query_type):
    """Execute AI analysis"""
    try:
        if query_type == "📝 Hibrit Mod":
            st.markdown("---")
            st.markdown("### 🤖 AI Detaylı Analizi")
        
        log_debug("Starting AI analysis")
        start_ai = time.time()
        answer = run_async(generate_final_answer(user_question))
        end_ai = time.time()
        ai_duration = end_ai - start_ai
        
        log_debug("AI analysis completed", {
            "ai_duration": ai_duration,
            "answer_length": len(answer) if answer else 0
        })
        
        st.markdown(f"""
        <div class="result-card">
            <h4>🤖 AI Analizi Tamamlandı</h4>
            <p><strong>⏱️ AI Süre:</strong> {ai_duration:.2f} saniye</p>
        </div>
        """, unsafe_allow_html=True)
        
        if answer:
            st.markdown(answer)
            add_to_history(user_question, "AI", ai_duration, True)
        else:
            st.warning("⚠️ AI analizi sonuç üretemedi")
            add_to_history(user_question, "AI", ai_duration, False)
            
    except Exception as e:
        show_error(e, "AI Analysis")


def _render_download_options(df):
    """Render download options for results"""
    try:
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
        log_debug("Download options created successfully")
    except Exception as e:
        show_error(e, "Download options")


def _render_quick_stats():
    """Render quick statistics sidebar"""
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
