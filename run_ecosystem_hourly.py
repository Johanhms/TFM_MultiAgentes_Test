import time
from datetime import datetime, timedelta
import subprocess
import sys

def ejecutar_bot():
    print(f"\n⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Alarma horaria activada.")
    try:
        # Ejecuta tu archivo principal de agentes en un subproceso aislado
        # Esto evita que un error de conexión en MT5 tire abajo el orquestador principal
        resultado = subprocess.run([sys.executable, "1_trading_agents.py"], capture_output=False, text=True)
        if resultado.returncode == 0:
            print("✅ Ciclo horario completado con éxito.")
        else:
            print(f"⚠️ El ciclo de agentes terminó con un código de salida no estándar: {resultado.returncode}")
    except Exception as e:
        print(f"❌ Error crítico ejecutando el ecosistema de agentes: {str(e)}")

def orquestador_principal():
    print("==================================================================")
    print("🤖 ORQUESTADOR DE FLUJO AGÉNTICO ACTIVO - MÁSTER QUANT")
    print("==================================================================")
    print("Sincronizando reloj con el servidor de liquidez...")
    
    # Ejecutamos una primera vez al arrancar el script para validar que todo esté bien
    ejecutar_bot()
    
    while True:
        # 1. Calculamos el tiempo exacto que falta para la siguiente hora en punto
        ahora = datetime.now()
        siguiente_hora = (ahora + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        segundos_de_espera = (siguiente_hora - ahora).total_seconds()
        
        print(f"💤 Orquestador en modo latente. Próxima evaluación en {segundos_de_espera/60:.1f} minutos ({siguiente_hora.strftime('%H:%M:%S')}).")
        
        # 2. El script se duerme hasta el segundo exacto del cambio de hora
        time.sleep(segundos_de_espera)
        
        # 3. Despierta y ejecuta las leyes de los agentes
        ejecutar_bot()

if __name__ == "__main__":
    try:
        orquestador_principal()
    except KeyboardInterrupt:
        print("\n🛑 Orquestador detenido manualmente por el usuario. Cerrando flujos de trabajo.")