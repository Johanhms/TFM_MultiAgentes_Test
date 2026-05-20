import os
import time
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

ASSETS = [
    "BTC-USD", 
    "EURUSD=X", 
    "GBPUSD=X", 
    "GC=F", 
    "^GSPC"
]

CURRENCY_MAP = {
    "BTC-USD": ["USD"],
    "EURUSD=X": ["EUR", "USD"],
    "GBPUSD=X": ["GBP", "USD"],
    "GC=F": ["USD"],
    "^GSPC": ["USD"]
}

def get_macro_calendar(currencies: list) -> list:
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    eventos = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
            
        tree = ET.fromstring(response.content)
        for item in tree.findall('event'):
            country = item.find('country').text
            impact = item.find('impact').text
            
            if country in currencies and impact in ['High', 'Medium']:
                title = item.find('title').text
                date_str = item.find('date').text
                time_str = item.find('time').text
                forecast = item.find('forecast').text or "N/A"
                previous = item.find('previous').text or "N/A"
                
                evento_txt = f"- [{impact} Impact] {country}: {title} (Hora: {date_str} {time_str} | Prev: {previous} | Pron: {forecast})"
                eventos.append(evento_txt)
                
        return eventos[:5]
    except Exception as e:
        print(f"   ⚠️ Fallo extrayendo calendario: {e}")
        return []

def test_multiactivo_gemini():
    print("🧪 [Sandbox] Iniciando extracción con Exponential Backoff...\n")
    
    model_name = "gemini-2.5-flash"
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    except Exception as e:
        print(f"❌ Error inicializando Gemini: {e}")
        return

    for asset in ASSETS:
        print(f"{'='*60}")
        print(f"📊 Analizando: {asset}")
        print(f"{'='*60}")
        
        # 1. NOTICIAS
        ticker = yf.Ticker(asset)
        news_list = ticker.news
        headlines = []
        
        if news_list:
            for n in news_list[:3]:
                if isinstance(n, dict):
                    title = n.get('title')
                    if not title and 'content' in n and isinstance(n['content'], dict):
                        title = n['content'].get('title')
                    if title:
                        headlines.append(f"- {title}")
                        
        news_text = "\n".join(headlines) if headlines else "Sin noticias recientes."
        
        # 2. CALENDARIO
        divisas_relevantes = CURRENCY_MAP.get(asset, ["USD"])
        eventos_macro = get_macro_calendar(divisas_relevantes)
        macro_text = "\n".join(eventos_macro) if eventos_macro else "Sin eventos macroeconómicos de alto impacto esta semana."

        print("\n📰 NOTICIAS:")
        print(news_text)
        print("\n📅 CALENDARIO:")
        print(macro_text)
        print("-" * 60)
        
        # 3. LLM PROMPT
        prompt = f"""
        Eres un analista cuantitativo experto de un fondo de inversión. Estás evaluando el activo: {asset}.
        Noticias financieras de última hora:
        {news_text}
        Calendario macroeconómico reciente y próximo para sus divisas base:
        {macro_text}
        Sintetiza ambos datos. ¿Cuál es el sentimiento general del mercado e impacto macroeconómico para {asset}?
        Responde ÚNICAMENTE con una de estas tres palabras: BULLISH, BEARISH o NEUTRAL. No incluyas explicaciones.
        """
        
        print(f"\n🤖 Consultando consenso a {model_name}...")
        
        # --- SISTEMA DE REINTENTOS (EXPONENTIAL BACKOFF) ---
        max_intentos = 3
        for intento in range(max_intentos):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                sentiment = response.content.strip().upper()
                
                if "BULLISH" in sentiment: sentiment = "BULLISH"
                elif "BEARISH" in sentiment: sentiment = "BEARISH"
                else: sentiment = "NEUTRAL"
                    
                print(f"✅ Veredicto de IA: {sentiment}\n")
                break  # Si tiene éxito, sale del bucle de reintentos
                
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    tiempo_espera = 15 * (intento + 1)
                    print(f"   ⚠️ Límite de API detectado. Esperando {tiempo_espera}s para reintentar (Intento {intento + 1}/{max_intentos})...")
                    time.sleep(tiempo_espera)
                else:
                    print(f"❌ Error crítico conectando con Gemini: {e}\n")
                    break  # Si es otro error (ej. sin internet), cancela el activo
        
        # Pausa estructural entre cada activo del portafolio
        print("⏳ Pausa base de 10 segundos antes de analizar el siguiente activo...\n")
        time.sleep(10)

if __name__ == "__main__":
    test_multiactivo_gemini()