import os
import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime   
import pandas_ta as ta
from dotenv import load_dotenv

# --- CONFIGURACIÓN DE ACTIVOS ---
ASSET_MAPPING = {
    "BTC-USD": "BTCUSD",
    "EURUSD=X": "EURUSD.sml",
    "GBPUSD=X": "GBPUSD.sml",
    "GC=F": "XAUUSD.sml",
    "^GSPC": "US500",      
    "CL=F": "USOIL.sml",       
    "^DJI": "US30",            
    "NVDA": "NVDA_CFD.US"      
}

def is_market_open(asset_yahoo: str) -> bool:
    """
    Reloj interno que verifica la disponibilidad del mercado basándose 
    en el horario oficial de Nueva York (EST/EDT).
    """
    ny_tz = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_tz)
    
    # 1. Criptomonedas (Nunca duermen)
    if asset_yahoo == "BTC-USD":
        return True
        
    # Bloqueo General de Fines de Semana
    if ny_time.weekday() == 5: 
        return False
    if ny_time.weekday() == 6 and ny_time.hour < 17: 
        return False
        
    # 2. Acciones e Índices Americanos (09:30 a 16:00 NY)
    if asset_yahoo in ["NVDA", "^DJI", "^GSPC"]:
        if ny_time.hour < 9 or ny_time.hour >= 16:
            return False
        if ny_time.hour == 9 and ny_time.minute < 30:
            return False
        return True
        
    # 3. Forex y Materias Primas (Oro, Petróleo)
    if asset_yahoo in ["EURUSD=X", "GBPUSD=X", "GC=F", "CL=F"]:
        if ny_time.hour == 17: 
            return False
        return True
        
    return True

def get_dynamic_atr_mt5(symbol_mt5: str, length: int = 14) -> float:
    """
    Descarga datos en vivo desde MT5 y calcula el ATR exacto del activo.
    Reemplaza completamente la antigua dependencia de yfinance.
    """
    try:
        # Descargamos 100 velas de 1H (Suficiente para el cálculo de ATR 14)
        rates = mt5.copy_rates_from_pos(symbol_mt5, mt5.TIMEFRAME_H1, 0, 100)
        
        if rates is None or len(rates) == 0:
            print(f"   ⚠️ No se pudieron obtener datos históricos de MT5 para {symbol_mt5}.")
            return 0.0
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Renombramos para compatibilidad con pandas_ta
        df.rename(columns={'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        df.set_index('time', inplace=True)
        
        # Calculamos el ATR nativo
        atr = df.ta.atr(length=length)
        return float(atr.iloc[-1])
        
    except Exception as e:
        print(f"   ⚠️ Error calculando ATR en MT5 para {symbol_mt5}: {e}")
        return 0.0

def update_trailing_stops(symbol_mt5: str, atr_value: float, multiplier: float = 1.5):
    """
    Función modular para auditar y mover el Stop Loss dinámicamente.
    """
    if atr_value <= 0:
        return f"   ⚠️ ATR inválido para {symbol_mt5}, saltando monitoreo."
        
    print(f"🔄 [Monitor] Auditando posiciones en {symbol_mt5} (ATR actual: {atr_value:.5f})...")
    
    positions = mt5.positions_get(symbol=symbol_mt5)
    if positions is None or len(positions) == 0:
        return f"   Sin posiciones activas en {symbol_mt5}."
        
    atr_distance = atr_value * multiplier
    mensajes = []

    for pos in positions:
        ticket = pos.ticket
        current_sl = pos.sl
        current_price = pos.price_current
        open_price = pos.price_open
        
        # Lógica VENTA (SHORT)
        if pos.type == mt5.ORDER_TYPE_SELL:
            new_sl = current_price + atr_distance
            
            # 1. ¿Estamos en ganancias? 
            # 2. ¿El nuevo SL es mejor (menor) que el anterior?
            if current_price < open_price and (new_sl < current_sl or current_sl == 0.0):
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "symbol": symbol_mt5,
                    "sl": new_sl,
                    "tp": pos.tp
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    mensajes.append(f"   ✅ SL ajustado a {new_sl:.5f} (Ticket: {ticket})")
                else:
                    mensajes.append(f"   ❌ Error ajustando SL: {result.comment}")
        
        # Lógica COMPRA (LONG)
        elif pos.type == mt5.ORDER_TYPE_BUY:
            new_sl = current_price - atr_distance
            
            if current_price > open_price and (new_sl > current_sl or current_sl == 0.0):
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "symbol": symbol_mt5,
                    "sl": new_sl,
                    "tp": pos.tp
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    mensajes.append(f"   ✅ SL ajustado a {new_sl:.5f} (Ticket: {ticket})")
                else:
                    mensajes.append(f"   ❌ Error ajustando SL: {result.comment}")

    return "\n".join(mensajes) if mensajes else "   El precio no ha avanzado lo suficiente para mover el SL."

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🛡️ INICIANDO MONITOR DE TRAILING STOPS (MULTIACTIVO / NATIVO MT5)")
    print("="*50)

    # 1. Inicializar credenciales
    load_dotenv()
    if not mt5.initialize(path=os.getenv("MT5_PATH")):
        print(f"❌ Error iniciando MT5. Código: {mt5.last_error()}")
        exit()
        
    authorized = mt5.login(login=int(os.getenv("MT5_LOGIN")), password=os.getenv("MT5_PASSWORD"), server=os.getenv("MT5_SERVER"))
    
    if not authorized:
        print(f"❌ Error de login en MT5. Código: {mt5.last_error()}")
        mt5.shutdown()
        exit()

    # 2. Iterar sobre todos los activos del portafolio
    for asset_yahoo, asset_mt5 in ASSET_MAPPING.items():
        print(f"\n--- Analizando: {asset_mt5} ---")
        
        # NUEVO: Obtenemos la volatilidad real directamente de MT5
        atr_real = get_dynamic_atr_mt5(asset_mt5)
        
        # Ejecutamos el monitor
        resultado = update_trailing_stops(asset_mt5, atr_real, multiplier=1.5)
        print(resultado)
    
    # 3. Cerramos la conexión de forma segura
    mt5.shutdown()
    print("\n✅ MONITOREO FINALIZADO.")