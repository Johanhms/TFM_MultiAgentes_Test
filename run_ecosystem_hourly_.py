import time
from datetime import datetime, timedelta
import subprocess
import sys
import os

SCRIPT_AGENTES = "trading_agents_.py"  # Sincronizado con tu archivo actual
ARCHIVO_LOGS = "sistema_trading_.log"

def ejecutar_bot():
    ahora_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n⏰ [{ahora_str}] Alarma horaria activada. Invocando ecosistema agéntico...")
    
    dia_semana = datetime.now().weekday()
    hora_actual = datetime.now().hour
    
    if dia_semana == 5: 
        print("   💤 Modo fin de semana activo. Subproceso suspendido.")
        return
    if dia_semana == 6 and hora_actual < 17: 
        print("   💤 Modo fin de semana activo. Subproceso suspendido.")
        return

    try:
        # CREAMOS EL ESCUDO DE ENCODING INSTITUCIONAL
        # Forzamos a Windows a lanzar el subproceso usando UTF-8 nativo desde el entorno
        entorno_sistema = os.environ.copy()
        entorno_sistema["PYTHONIOENCODING"] = "utf-8"

        with open(ARCHIVO_LOGS, "a", encoding="utf-8", errors="ignore") as log_file:
            log_file.write(f"\n--- INICIO DE CICLO OPERATIVO: {ahora_str} ---\n")
            log_file.flush()
            
            resultado = subprocess.run(
                [sys.executable, SCRIPT_AGENTES],
                stdout=log_file,
                stderr=log_file,
                text=True,
                env=entorno_sistema # Inyección de entorno UTF-8
            )
            
        # El sys.stdout del padre permanece intacto. Este print nunca fallará:
        if resultado.returncode == 0:
            print(f"   ✅ Ciclo horario completado. Registros salvados en '{ARCHIVO_LOGS}'.")
        else:
            print(f"   ⚠️ El ciclo de agentes terminó con código de salida no estándar: {resultado.returncode}")
            
    except Exception as e:
        print(f"   ❌ Error crítico en el despachador del orquestador: {str(e)}")

def orquestador_principal():
    print("==================================================================")
    print("🤖 DEPLOY: ORQUESTADOR DE FLUJO AGÉNTICO AUTÓNOMO - MÁSTER QUANT")
    print("==================================================================")
    print(f"Log de auditoría unificado en: '{os.path.abspath(ARCHIVO_LOGS)}'")
    
    if not os.path.exists(SCRIPT_AGENTES):
        print(f"❌ ERROR CONFIGURACIÓN: No se encuentra el script '{SCRIPT_AGENTES}'.")
        sys.exit(1)
        
    ejecutar_bot()
    
    while True:
        ahora = datetime.now()
        
        # 1. Proyectar el minuto 01 de la hora actual
        siguiente_hora = ahora.replace(minute=1, second=0, microsecond=0)
        
        # 2. Si ya pasó el minuto 01 de esta hora, programar para la hora siguiente
        if siguiente_hora <= ahora:
            siguiente_hora += timedelta(hours=1)
            
        # 3. Calcular la espera exacta en segundos
        segundos_de_espera = (siguiente_hora - ahora).total_seconds()
        
        print(f"💤 Latente... Próxima evaluación en {segundos_de_espera/60:.1f} minutos ({siguiente_hora.strftime('%H:%M:%S')}).")
        time.sleep(segundos_de_espera)
        ejecutar_bot()

if __name__ == "__main__":
    try:
        orquestador_principal()
    except KeyboardInterrupt:
        print("\n🛑 Orquestador detenido manualmente por el operador.")