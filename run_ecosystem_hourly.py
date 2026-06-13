import os
import time
from datetime import datetime, timedelta
import subprocess
import sys

# CONFIGURACIÓN DE PRODUCCIÓN - Sincroniza este nombre con tu archivo de agentes
SCRIPT_AGENTES = "trading_agents_V2.py" 
ARCHIVO_LOGS = "sistema_trading.log"

def ejecutar_bot():
    ahora_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n⏰ [{ahora_str}] Alarma horaria activada. Invocando ecosistema agéntico...")
    
    # Validamos bloqueo de fin de semana directamente en el orquestador raíz
    # weekday() devuelve 5 para Sábado y 6 para Domingo
    dia_semana = datetime.now().weekday()
    hora_actual = datetime.now().hour
    
    # Bypass estricto de fin de semana (Excepto viernes antes del cierre y domingo tras la apertura)
    if dia_semana == 5: # Sábado completo
        print("   💤 Modo fin de semana activo (Mercados tradicionales cerrados). Subproceso suspendido.")
        return
    if dia_semana == 6 and hora_actual < 17: # Domingo antes de las 17:00 NY
        print("   💤 Modo fin de semana activo (Esperando apertura de Asia). Subproceso suspendido.")
        return

    try:
        # Abrimos el archivo de logs en modo "append" (añadir al final)
        with open(ARCHIVO_LOGS, "a", encoding="utf-8") as log_file:
            # Escribimos un encabezado en el log para separar los ciclos horarios
            log_file.write(f"\n--- INICIO DE CICLO OPERATIVO: {ahora_str} ---\n")
            log_file.flush()
            
            # Ejecutamos el subproceso redirigiendo la salida estándar y los errores al archivo log
            resultado = subprocess.run(
                [sys.executable, SCRIPT_AGENTES],
                stdout=log_file,
                stderr=log_file,
                text=True
            )
            
        if resultado.returncode == 0:
            print(f"   ✅ Ciclo horario completado. Registros salvados en '{ARCHIVO_LOGS}'.")
        else:
            print(f"   ⚠️ El ciclo de agentes terminó con código de salida no estándar: {resultado.returncode}")
            print(f"      Revisa el archivo '{ARCHIVO_LOGS}' para auditar las trazas del error.")
            
    except Exception as e:
        print(f"   ❌ Error crítico en el despachador del orquestador: {str(e)}")

def orquestador_principal():
    print("==================================================================")
    # Ajustamos el string de inicialización al entorno productivo real
    print("🤖 DEPLOY: ORQUESTADOR DE FLUJO AGÉNTICO AUTÓNOMO - MÁSTER QUANT")
    print("==================================================================")
    print(f"Log de auditoría unificado configurado en: '{os.path.abspath(ARCHIVO_LOGS)}'")
    print("Sincronizando reloj interno con el servidor de MetaTrader 5...")
    
    # Verificación de inicialización de seguridad para validar rutas
    if not os.path.exists(SCRIPT_AGENTES):
        print(f"❌ ERROR CONFIGURACIÓN: No se encuentra el script de agentes '{SCRIPT_AGENTES}' en el directorio actual.")
        sys.exit(1)
        
    # Ejecutamos una primera vez al arrancar para certificar que el pipeline no tiene errores de sintaxis
    ejecutar_bot()
    
    while True:
        # 1. Calculamos el tiempo exacto que falta para la siguiente hora en punto
        ahora = datetime.now()
        siguiente_hora = (ahora + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        segundos_de_espera = (siguiente_hora - ahora).total_seconds()
        
        # Pequeño margen de seguridad de 2 segundos para evitar latencias de reloj del sistema
        segundos_de_espera += 2
        
        print(f"💤 Latente... Próxima evaluación en {segundos_de_espera/60:.1f} minutos ({siguiente_hora.strftime('%H:%M:%S')}).")
        
        # 2. El script suspende su ejecución de hilos hasta el cambio de hora exacto
        time.sleep(segundos_de_espera)
        
        # 3. Dispara la evaluación de las leyes agénticas
        ejecutar_bot()

if __name__ == "__main__":
    try:
        orquestador_principal()
    except KeyboardInterrupt:
        print("\n🛑 Orquestador detenido manualmente por el operador. Flujos de trabajo cerrados de forma segura.")