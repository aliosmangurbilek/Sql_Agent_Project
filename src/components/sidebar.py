"""
Sidebar component for system control panel
"""

import streamlit as st
import os
import time
import psycopg2
from .provider_manager import render_provider_selector, render_provider_switching
from .charts import create_performance_chart
from ..utils.debug import log_debug, show_error
from ..utils.environment import check_environment


def render_sidebar():
    """Render the main sidebar with system controls"""
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h3>🔧 Sistem Kontrol Paneli</h3></div>', unsafe_allow_html=True)
        
        # Debug Mode Toggle
        _render_debug_section()
        
        # AI Provider Section
        render_provider_selector()
        render_provider_switching()
        
        st.markdown("---")
        
        # System Status
        _render_system_status()
        
        # Performance Metrics
        _render_performance_metrics()
        
        st.markdown("---")
        
        # Query History
        _render_query_history()
        
        # Clear History Button
        if st.button("🗑️ Geçmişi Temizle", type="secondary"):
            st.session_state.query_history = []
            st.session_state.performance_stats = {'total_queries': 0, 'avg_response_time': 0, 'success_rate': 100}
            st.rerun()


def _render_debug_section():
    """Render debug mode section"""
    st.markdown("### 🐛 Debug Modu")
    debug_mode = st.checkbox("Debug modu aktif", 
                           value=st.session_state.get('debug_mode', False),
                           help="Debug bilgileri ve detaylı hata mesajları gösterir")
    st.session_state.debug_mode = debug_mode
    
    if debug_mode:
        st.info("🔍 Debug modu aktif - Detaylı loglar gösterilecek")
        
        # Environment status
        with st.expander("🌍 Environment Durumu", expanded=False):
            env_status = check_environment()
            
            st.write("**Database:**")
            st.code(f"Status: {env_status['database']['status']}")
            st.code(f"URL: {env_status['database']['value']}")
            
            st.write("**AI Provider:**")
            ai_provider = env_status['ai_provider']
            st.code(f"Type: {ai_provider['type']}")
            
            if ai_provider['type'] == 'OpenAI':
                st.code(f"API Key: {ai_provider['api_key']}")
                st.code(f"Model: {ai_provider['model']}")
            elif ai_provider['type'] == 'Google Gemini':
                st.code(f"API Key: {ai_provider['api_key']}")
                st.code(f"Model: {ai_provider['model']}")
            elif ai_provider['type'] == 'OpenRouter':
                st.code(f"API Key: {ai_provider['api_key']}")
                st.code(f"Model: {ai_provider['model']}")
            elif ai_provider['type'] == 'Ollama (Local)':
                st.code(f"Host: {ai_provider['host']}")
                st.code(f"Chat Model: {ai_provider['chat_model']}")
                st.code(f"Embed Model: {ai_provider['embedding_model']}")
            else:
                st.code(f"Status: {ai_provider.get('status', 'Unknown configuration')}")
        
        # Clear debug logs button
        if st.button("🗑️ Debug Logları Temizle"):
            if 'debug_logs' in st.session_state:
                st.session_state.debug_logs = []
            st.success("Debug logları temizlendi")


def _render_system_status():
    """Render system status section"""
    with st.spinner("Sistem kontrol ediliyor..."):
        try:
            log_debug("Database connection test starting")
            
            # Simple database test without AI
            try:
                conn = psycopg2.connect(os.environ.get("DATABASE_URL"))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as total_films FROM film LIMIT 1")
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    db_status = "online"
                    total_films = result[0]
                    log_debug("Database test successful", {"total_films": total_films})
                else:
                    db_status = "offline"
                    total_films = "N/A"
                    log_debug("Database test failed - no results")
            except Exception as e:
                db_status = "offline"
                total_films = "N/A"
                log_debug("Database connection failed", {"error": str(e)})
                
        except Exception as e:
            db_status = "offline"
            total_films = "N/A"
            show_error(e, "Database connection test")
    
    # Status indicator
    status_class = "status-online" if db_status == "online" else "status-offline"
    status_text = "🟢 Çevrimiçi" if db_status == "online" else "🔴 Çevrimdışı"
    
    st.markdown(f"""
    <div style="padding: 1rem; background: grey; border-radius: 8px; margin: 1rem 0;">
        <h4>📊 Sistem Durumu</h4>
        <p><span class="status-indicator {status_class}"></span><strong>Veritabanı:</strong> {status_text}</p>
        <p>📽️ <strong>Toplam Film:</strong> {total_films}</p>
        <p>🤖 <strong>AI Provider:</strong> {os.getenv('AI_PROVIDER', 'ollama').title()}</p>
        <p>⚡ <strong>Platform:</strong> {_get_platform_name()}</p>
    </div>
    """, unsafe_allow_html=True)


def _render_performance_metrics():
    """Render performance metrics section"""
    if st.session_state.performance_stats['total_queries'] > 0:
        st.markdown("### 📈 Performans Metrikleri")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Sorgu", st.session_state.performance_stats['total_queries'])
        with col2:
            st.metric("Başarı Oranı", f"{st.session_state.performance_stats['success_rate']:.1f}%")
        
        st.metric("Ortalama Süre", f"{st.session_state.performance_stats['avg_response_time']:.2f}s")
        
        # Performance chart
        perf_chart = create_performance_chart()
        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)


def _render_query_history():
    """Render query history section"""
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


def _get_platform_name():
    """Get platform name based on current AI provider"""
    provider = os.getenv('AI_PROVIDER', 'ollama').lower()
    platform_map = {
        'openai': 'OpenAI API',
        'gemini': 'Google AI',
        'openrouter': 'OpenRouter',
        'ollama': 'Ollama Local'
    }
    return platform_map.get(provider, 'Unknown')
