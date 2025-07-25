"""
Debug and logging utilities for the application
"""

import streamlit as st
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional


# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_debug(message: str, data: Dict[str, Any] = None):
    """Log debug information both to console and Streamlit"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    
    if data:
        log_msg += f" | Data: {data}"
    
    logger.info(log_msg)
    
    # Show in Streamlit debug expander if debug mode is enabled
    if st.session_state.get('debug_mode', False):
        with st.expander(f"🐛 Debug [{timestamp}]", expanded=False):
            st.code(log_msg)
            if data:
                st.json(data)


def show_error(error: Exception, context: str = ""):
    """Display error information in a user-friendly way"""
    error_type = type(error).__name__
    error_msg = str(error)
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Log the error
    logger.error(f"[{timestamp}] {context}: {error_type}: {error_msg}")
    
    # Show to user
    st.error(f"❌ **Hata ({error_type})**: {error_msg}")
    
    # Show debug info if debug mode is on
    if st.session_state.get('debug_mode', False):
        with st.expander("🔍 Hata Detayları", expanded=False):
            st.code(f"Hata Türü: {error_type}")
            st.code(f"Mesaj: {error_msg}")
            st.code(f"Konum: {context}")
            
            # Show full traceback
            tb_str = traceback.format_exc()
            st.code(tb_str, language="python")


def init_session_state():
    """Initialize session state variables"""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'performance_stats' not in st.session_state:
        st.session_state.performance_stats = {
            'total_queries': 0, 
            'avg_response_time': 0, 
            'success_rate': 100
        }
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False


def add_to_history(query: str, result_type: str, duration: float, success: bool):
    """Add query to history and update performance stats"""
    st.session_state.query_history.insert(0, {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'query': query[:50] + '...' if len(query) > 50 else query,
        'type': result_type,
        'duration': duration,
        'success': success
    })
    
    # Keep only last 10 queries
    if len(st.session_state.query_history) > 10:
        st.session_state.query_history = st.session_state.query_history[:10]
    
    # Update performance statistics
    st.session_state.performance_stats['total_queries'] += 1
    
    # Update average response time
    total_time = st.session_state.performance_stats['avg_response_time'] * (st.session_state.performance_stats['total_queries'] - 1)
    st.session_state.performance_stats['avg_response_time'] = (total_time + duration) / st.session_state.performance_stats['total_queries']
    
    # Update success rate
    successful_queries = sum(1 for h in st.session_state.query_history if h['success'])
    st.session_state.performance_stats['success_rate'] = (successful_queries / len(st.session_state.query_history)) * 100
