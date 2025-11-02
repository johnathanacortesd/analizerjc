import streamlit as st
import sys
import os
import subprocess

# --- CÓDIGO DE DIAGNÓSTICO ---
# Esta sección se ejecuta ANTES de cualquier comando de Streamlit para imprimir en los logs.
print("--- INICIANDO SCRIPT DE DIAGNÓSTICO ---")
print(f"Versión de Python (sys.version): {sys.version}")
print(f"Ejecutable de Python (sys.executable): {sys.executable}")
print(f"Directorio de trabajo actual (os.getcwd): {os.getcwd()}")
try:
    files_in_dir = os.listdir('.')
    print(f"Archivos en el directorio raíz: {files_in_dir}")
except Exception as e:
    print(f"No se pudieron listar los archivos: {e}")
print("--- FIN DE LOGS PRE-STREAMLIT ---")
# ------------------------------------


# LA PRIMERA LLAMADA A STREAMLIT DEBE SER set_page_config
st.set_page_config(
    page_title="App de Diagnóstico",
    page_icon="🔧",
    layout="wide"
)

# --- INFORMACIÓN MOSTRADA EN PANTALLA ---
st.title("🔧 App de Diagnóstico del Entorno")
st.header("1. Verificación de Versión de Python")

st.info(f"La versión de Python que está ejecutando esta app es: **{sys.version}**")

if "3.13" in sys.version:
    st.error(
        "¡ALERTA! La app se está ejecutando en Python 3.13. "
        "Esta es la causa confirmada del error. Streamlit Cloud no está leyendo "
        "correctamente tu archivo `runtime.txt`. Verifica que el archivo exista en la "
        "raíz de tu repositorio y contenga `python-3.11`."
    )
else:
    st.success(
        "¡BUENAS NOTICIAS! La versión de Python parece ser correcta (no es 3.13). "
        "El error original debería estar resuelto."
    )

st.header("2. Verificación de Archivos del Repositorio")
st.write("Streamlit Cloud ve los siguientes archivos en el directorio principal de tu app:")

try:
    files_in_dir = os.listdir('.')
    st.code('\n'.join(files_in_dir))
    if 'runtime.txt' not in files_in_dir:
        st.warning("ADVERTENCIA: No se encontró el archivo `runtime.txt` en el directorio.")
    if 'requirements.txt' not in files_in_dir:
        st.warning("ADVERTENCIA: No se encontró el archivo `requirements.txt` en el directorio.")
except Exception as e:
    st.error(f"No se pudieron listar los archivos desde la app: {e}")

st.header("3. Verificación de Paquetes Instalados")
st.write("Haz clic en el botón para ver qué librerías están realmente instaladas en el entorno.")

if st.button("Mostrar paquetes instalados (pip freeze)"):
    with st.spinner("Ejecutando `pip freeze`..."):
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'freeze'],
                capture_output=True,
                text=True,
                timeout=30
            )
            st.code(result.stdout)
            if result.stderr:
                st.warning("Salida de error del comando pip:")
                st.code(result.stderr)
        except Exception as e:
            st.error(f"Error al ejecutar pip freeze: {e}")

st.header("4. Próximos Pasos")
st.markdown("""
- **Si ves el mensaje de ALERTA sobre Python 3.13:** El problema es 100% el entorno. La solución es asegurar que el archivo `runtime.txt` sea leído por Streamlit. Intenta hacer un cambio menor (añadir un espacio en un comentario) en tu `app.py`, guarda, y sube ambos archivos de nuevo a GitHub para forzar una reconstrucción completa.
- **Si ves el mensaje de BUENAS NOTICIAS:** ¡Excelente! El entorno ya es correcto. Ahora puedes reemplazar este código de diagnóstico con el código completo de tu aplicación de análisis de noticias, y debería funcionar.
- **Si la app sigue sin cargar y muestra el error `TypeError`:** Ve a "Manage app" -> Logs, y copia y pega TODO el contenido que veas. Los logs que imprimimos al principio nos darán la respuesta definitiva.
""")
