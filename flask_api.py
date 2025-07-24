from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import asyncio
import os
import json
import time
from datetime import datetime
from schema_tools import ask_db, generate_final_answer

# Flask uygulamasını oluştur
app = Flask(__name__)
CORS(app)  # CORS desteği

# Çevre değişkenlerini ayarla
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

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 Pagila API Interface</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .content {
            padding: 2rem;
        }
        
        .section {
            margin-bottom: 2rem;
            padding: 1.5rem;
            border-radius: 10px;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
        }
        
        .input-group {
            margin-bottom: 1rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: #333;
        }
        
        input, textarea, select {
            width: 100%;
            padding: 0.8rem;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 0.8rem 2rem;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s;
            margin-right: 0.5rem;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .result {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 8px;
            background: #e8f5e8;
            border-left: 4px solid #28a745;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .error {
            background: #f8d7da;
            border-left-color: #dc3545;
            color: #721c24;
        }
        
        .loading {
            background: #fff3cd;
            border-left-color: #ffc107;
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
        
        .api-docs {
            background: #e9ecef;
            padding: 1rem;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Pagila Database API</h1>
            <p>RESTful API ile Doğal Dil Veritabanı Sorgulama</p>
        </div>
        
        <div class="content">
            <!-- Sistem Durumu -->
            <div class="section">
                <h3>📊 Sistem Durumu</h3>
                <p id="systemStatus">Kontrol ediliyor...</p>
            </div>
            
            <div class="grid">
                <!-- Hızlı SQL Sorgusu -->
                <div class="section">
                    <h3>🚀 Hızlı SQL Sorgusu</h3>
                    <div class="input-group">
                        <label for="sqlQuery">Doğal Dil Sorgusu:</label>
                        <textarea id="sqlQuery" rows="3" placeholder="Örnek: Veritabanında kaç film var?"></textarea>
                    </div>
                    <button class="btn" onclick="runSQLQuery()">🔍 Sorgula</button>
                    <button class="btn" onclick="clearResults('sqlResult')">🗑️ Temizle</button>
                    <div id="sqlResult"></div>
                </div>
                
                <!-- AI Analizi -->
                <div class="section">
                    <h3>🤖 AI Detaylı Analizi</h3>
                    <div class="input-group">
                        <label for="aiQuery">Analiz Sorusu:</label>
                        <textarea id="aiQuery" rows="3" placeholder="Örnek: Pagila veritabanının yapısını açıkla"></textarea>
                    </div>
                    <button class="btn" onclick="runAIAnalysis()">🧠 Analiz Et</button>
                    <button class="btn" onclick="clearResults('aiResult')">🗑️ Temizle</button>
                    <div id="aiResult"></div>
                </div>
            </div>
            
            <!-- Örnek Sorgular -->
            <div class="section">
                <h3>💡 Örnek Sorgular</h3>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                    <button class="btn" onclick="setQuery('sqlQuery', 'Veritabanında kaç film var?')">Film Sayısı</button>
                    <button class="btn" onclick="setQuery('sqlQuery', 'En uzun film hangisi?')">En Uzun Film</button>
                    <button class="btn" onclick="setQuery('sqlQuery', 'Drama kategorisinde kaç film var?')">Drama Filmleri</button>
                    <button class="btn" onclick="setQuery('aiQuery', 'En popüler film kategorileri analizi')">Kategori Analizi</button>
                    <button class="btn" onclick="setQuery('aiQuery', 'Aktör-film ilişkisi nasıl?')">Aktör Analizi</button>
                </div>
            </div>
            
            <!-- API Dokumentasyonu -->
            <div class="section">
                <h3>📚 API Dokumentasyonu</h3>
                <div class="api-docs">
<strong>Endpoints:</strong>
POST /api/query - Hızlı SQL sorgusu
  Body: {"question": "Sorunuz buraya"}
  
POST /api/analyze - AI detaylı analizi  
  Body: {"question": "Analiz sorunuz buraya"}
  
GET /api/status - Sistem durumu kontrolü

<strong>Örnek curl komutları:</strong>
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Kaç film var?"}'
                </div>
            </div>
        </div>
    </div>

    <script>
        // Sistem durumunu kontrol et
        async function checkSystemStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                const statusElement = document.getElementById('systemStatus');
                
                if (data.database_status === 'online') {
                    statusElement.innerHTML = `
                        <span class="status-indicator status-online"></span>
                        <strong>Sistem Çevrimiçi</strong><br>
                        🎬 Toplam Film: ${data.total_films}<br>
                        🤖 AI Model: ${data.ai_model}<br>
                        ⚡ Yanıt Süresi: ${data.response_time}ms
                    `;
                } else {
                    statusElement.innerHTML = `
                        <span class="status-indicator status-offline"></span>
                        <strong>Sistem Çevrimdışı</strong><br>
                        Lütfen bağlantıları kontrol edin.
                    `;
                }
            } catch (error) {
                document.getElementById('systemStatus').innerHTML = `
                    <span class="status-indicator status-offline"></span>
                    <strong>Bağlantı Hatası</strong><br>
                    ${error.message}
                `;
            }
        }
        
        // SQL sorgusu çalıştır
        async function runSQLQuery() {
            const query = document.getElementById('sqlQuery').value;
            const resultDiv = document.getElementById('sqlResult');
            
            if (!query.trim()) {
                showResult(resultDiv, 'Lütfen bir sorgu girin.', 'error');
                return;
            }
            
            showResult(resultDiv, 'Sorgu çalıştırılıyor...', 'loading');
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: query })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const resultText = `✅ Başarılı (${data.duration}s)
📊 ${data.count} kayıt bulundu

${JSON.stringify(data.results, null, 2)}`;
                    showResult(resultDiv, resultText, 'result');
                } else {
                    showResult(resultDiv, `❌ Hata: ${data.error}`, 'error');
                }
            } catch (error) {
                showResult(resultDiv, `❌ Bağlantı Hatası: ${error.message}`, 'error');
            }
        }
        
        // AI analizi çalıştır
        async function runAIAnalysis() {
            const query = document.getElementById('aiQuery').value;
            const resultDiv = document.getElementById('aiResult');
            
            if (!query.trim()) {
                showResult(resultDiv, 'Lütfen bir analiz sorusu girin.', 'error');
                return;
            }
            
            showResult(resultDiv, 'AI analizi yapılıyor... (Bu işlem uzun sürebilir)', 'loading');
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: query })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const resultText = `🤖 AI Analizi Tamamlandı (${data.duration}s)

${data.answer}`;
                    showResult(resultDiv, resultText, 'result');
                } else {
                    showResult(resultDiv, `❌ Hata: ${data.error}`, 'error');
                }
            } catch (error) {
                showResult(resultDiv, `❌ Bağlantı Hatası: ${error.message}`, 'error');
            }
        }
        
        // Sonuç göster
        function showResult(element, text, type) {
            element.className = `result ${type}`;
            element.textContent = text;
        }
        
        // Sonuçları temizle
        function clearResults(elementId) {
            document.getElementById(elementId).innerHTML = '';
        }
        
        // Sorgu ayarla
        function setQuery(inputId, query) {
            document.getElementById(inputId).value = query;
        }
        
        // Sayfa yüklendiğinde sistem durumunu kontrol et
        checkSystemStatus();
        
        // Her 30 saniyede bir durumu güncelle
        setInterval(checkSystemStatus, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Sistem durumu kontrolü"""
    try:
        start_time = time.time()
        
        # Test sorgusu
        result = run_async(ask_db("SELECT COUNT(*) as count FROM film"))
        
        response_time = int((time.time() - start_time) * 1000)
        
        if result:
            return jsonify({
                'success': True,
                'database_status': 'online',
                'total_films': result[0]['count'],
                'ai_model': os.environ.get('CHAT_MODEL', 'N/A'),
                'response_time': response_time,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'database_status': 'offline',
                'error': 'Database query returned no results'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'database_status': 'offline',
            'error': str(e)
        }), 500

@app.route('/api/query', methods=['POST'])
def api_query():
    """Hızlı SQL sorgusu API"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing question parameter'
            }), 400
        
        question = data['question']
        
        if not question.strip():
            return jsonify({
                'success': False,
                'error': 'Question cannot be empty'
            }), 400
        
        start_time = time.time()
        result = run_async(ask_db(question))
        duration = round(time.time() - start_time, 2)
        
        if result:
            return jsonify({
                'success': True,
                'question': question,
                'results': result[:100],  # İlk 100 kayıt
                'count': len(result),
                'duration': f"{duration}s",
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No results found',
                'question': question,
                'duration': f"{duration}s"
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'question': data.get('question', 'Unknown') if 'data' in locals() else 'Unknown'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """AI detaylı analizi API"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing question parameter'
            }), 400
        
        question = data['question']
        
        if not question.strip():
            return jsonify({
                'success': False,
                'error': 'Question cannot be empty'
            }), 400
        
        start_time = time.time()
        answer = run_async(generate_final_answer(question))
        duration = round(time.time() - start_time, 2)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'duration': f"{duration}s",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'question': data.get('question', 'Unknown') if 'data' in locals() else 'Unknown'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Sağlık kontrolü"""
    return jsonify({
        'status': 'healthy',
        'service': 'Pagila Database API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Pagila Database API...")
    print("📍 Available endpoints:")
    print("   • http://localhost:5000/ - Web Interface")
    print("   • http://localhost:5000/api/status - System Status")
    print("   • http://localhost:5000/api/query - SQL Query")
    print("   • http://localhost:5000/api/analyze - AI Analysis")
    print("   • http://localhost:5000/api/health - Health Check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
