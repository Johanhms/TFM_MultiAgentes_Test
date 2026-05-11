import os
import yfinance as yf
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Forzamos la recarga de las variables de entorno
load_dotenv(override=True)

def test_gemini_connection():
    print("🧪 [Test Mode] Iniciando prueba de conexión con Gemini vía LangChain...")
    
    print("📰 Descargando noticias recientes de BTC-USD...")
    ticker = yf.Ticker("BTC-USD")
    news_list = ticker.news
    
    headlines = []
    if news_list:
        for n in news_list[:5]:
            if isinstance(n, dict):
                title = n.get('title')
                if not title and 'content' in n and isinstance(n['content'], dict):
                    title = n['content'].get('title')
                    
                publisher = n.get('publisher') or n.get('provider')
                if not publisher and 'content' in n and isinstance(n['content'], dict):
                    provider_data = n['content'].get('provider', {})
                    if isinstance(provider_data, dict):
                        publisher = provider_data.get('displayName')
                    elif isinstance(provider_data, str):
                        publisher = provider_data
                
                publisher = publisher or "Agencia Externa"
                
                if title:
                    headlines.append(f"- {title} (Fuente: {publisher})")
                
    if not headlines:
        print("❌ No se pudieron obtener noticias.")
        return
        
    news_text = "\n".join(headlines)
    print("\n--- NOTICIAS EXTRAÍDAS ---")
    print(news_text)
    print("--------------------------\n")
    
    # --- EL CEREBRO DEFINITIVO ---
    model_name = "gemini-2.5-flash" 
    print(f"🤖 Conectando con el modelo: {model_name}...")
    
    prompt = f"""
    Eres un analista financiero experto. Revisa los siguientes titulares recientes sobre BTC-USD:
    {news_text}
    
    Determina el sentimiento general del mercado para este activo.
    Responde ÚNICAMENTE con una de estas tres palabras: BULLISH, BEARISH o NEUTRAL.
    No incluyas explicaciones.
    """
    
    try:
        # Instanciamos el modelo a través de LangChain
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        
        # Hacemos la consulta
        response = llm.invoke([HumanMessage(content=prompt)])
        sentiment = response.content.strip().upper()
        
        print("\n✅ ¡CONEXIÓN EXITOSA!")
        print(f"📊 Sentimiento devuelto por Gemini: {sentiment}")
        
    except Exception as e:
        print(f"\n❌ Error conectando con Gemini:\n{e}")

if __name__ == "__main__":
    test_gemini_connection()