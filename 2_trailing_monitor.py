import os
import MetaTrader5 as mt5
from dotenv import load_dotenv

def update_trailing_stops(symbol_mt5: str, atr_value: float, multiplier: float = 1.5):
    """
    Función modular para auditar y mover el Stop Loss dinámicamente.
    """
    print(f"🔄 [Monitor] Auditando posiciones en {symbol_mt5}...")
    
    positions = mt5.positions_get(symbol=symbol_mt5)
    if positions is None or len(positions) == 0:
        return f"Sin posiciones activas en {symbol_mt5}."
        
    atr_distance = atr_value * multiplier
    mensajes = []

    for pos in positions:
        ticket = pos.ticket
        current_sl = pos.sl
        current_price = pos.price_current
        open_price = pos.price_open
        
        # Lógica VENTA (SHORT) - Como tu orden de BTCUSD es SELL
        if pos.type == mt5.ORDER_TYPE_SELL:
            new_sl = current_price + atr_distance
            
            # 1. ¿Estamos en ganancias? (precio actual < precio apertura)
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
                    mensajes.append(f"✅ SL ajustado a {new_sl:.2f} (Ticket: {ticket})")
                else:
                    mensajes.append(f"❌ Error ajustando SL: {result.comment}")
        
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
                    mensajes.append(f"✅ SL ajustado a {new_sl:.2f} (Ticket: {ticket})")
                else:
                    mensajes.append(f"❌ Error ajustando SL: {result.comment}")

    return "\n".join(mensajes) if mensajes else "El precio no ha avanzado lo suficiente."

if __name__ == "__main__":
    # 1. Inicializar credenciales
    load_dotenv()
    if not mt5.initialize(path=os.getenv("MT5_PATH")):
        print("Error iniciando MT5")
        exit()
        
    mt5.login(login=int(os.getenv("MT5_LOGIN")), password=os.getenv("MT5_PASSWORD"), server=os.getenv("MT5_SERVER"))

    # 2. Parámetros para la prueba manual
    # Para probar ahora mismo, simulamos un ATR manual de $500 para Bitcoin.
    # En la versión final, este valor lo extraeremos directamente de pandas-ta.
    activo_a_monitorizar = "BTCUSD"
    atr_simulado = 500.0  

    # 3. Ejecutar la función
    resultado = update_trailing_stops(activo_a_monitorizar, atr_simulado, multiplier=1.5)
    print(resultado)
    
    mt5.shutdown()