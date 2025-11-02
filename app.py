import streamlit as st
from groq import Groq
import pandas as pd
import json
import io
from datetime import datetime
import numpy as np
from difflib import SequenceMatcher
import openpyxl
import unicodedata # MEJORA: Librería para manejar acentos correctamente

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="Analizador de Noticias IA | Colombia", icon="📰", layout="wide")
MODEL_NAME = "llama-3.1-70b-versatile"

# --- FUNCIÓN DE NORMALIZACIÓN (NUEVA) ---
def normalizar_texto(texto):
    """Convierte un texto a minúsculas y elimina los acentos."""
    if not isinstance(texto, str):
        return texto
    # NFD descompone caracteres en base + acento.
    # Luego, se eliminan los caracteres diacríticos (acentos).
    s = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return s.lower()

# --- AUTENTICACIÓN ---
if "password_correct" not in st.session_state: st.session_state.password_correct = False
def validate_password():
    if st.session_state.get("password") == st.secrets.get("PASSWORD"):
        st.session_state.password_correct = True
        if "password" in st.session_state: del st.session_state["password"]
    else: st.session_state.password_attempted = True
if not st.session_state.password_correct:
    st.markdown("<div style='text-align: center; padding: 2rem 0;'><h1 style='color: #2E86AB; font-size: 3rem;'>📰</h1><h2>Analizador Inteligente de Noticias</h2><p style='color: #666; margin-bottom: 2rem;'>Análisis de sentimiento y temas dinámicos con IA para el contexto colombiano.</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1]); 
    with col2: st.text_input("🔐 Contraseña", type="password", on_change=validate_password, key="password")
    if st.session_state.get("password_attempted", False): st.error("❌ Contraseña incorrecta")
    st.stop()

# --- INICIALIZACIÓN ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except KeyError:
    st.error("❌ Configure GROQ_API_KEY en los Secrets de Streamlit.")
    st.stop()
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False

# --- FUNCIONES DE DETECCIÓN DE DUPLICADOS (DINÁMICAS) ---
def calcular_similitud_titulo(titulo1, titulo2):
    if pd.isna(titulo1) or pd.isna(titulo2): return 0.0
    return SequenceMatcher(None, str(titulo1).lower().strip(), str(titulo2).lower().strip()).ratio()

def detectar_duplicados_exactos(df, columnas_criterio):
    if not columnas_criterio or not all(c in df.columns for c in columnas_criterio):
        st.warning("⚠️ Columnas para duplicados exactos no válidas. Se omitirá este paso.")
        df['Es_Duplicado_Exacto'] = False; df['Grupo_Duplicado_Exacto'] = -1
        return df
    df['Es_Duplicado_Exacto'] = df.duplicated(subset=columnas_criterio, keep=False)
    df['Grupo_Duplicado_Exacto'] = -1
    if df['Es_Duplicado_Exacto'].any():
        df.loc[df['Es_Duplicado_Exacto'], 'Grupo_Duplicado_Exacto'] = df[df['Es_Duplicado_Exacto']].groupby(columnas_criterio, sort=False).ngroup()
    return df

def detectar_duplicados_similares(df, columnas_agrupacion, umbral_similitud=0.85):
    if not columnas_agrupacion or not all(c in df.columns for c in columnas_agrupacion):
        st.warning("⚠️ Columnas para agrupar duplicados similares no válidas. Se omitirá este paso.")
        df['Es_Duplicado_Similar'] = False; df['Grupo_Duplicado_Similar'] = -1
        return df
    df['Es_Duplicado_Similar'] = False; df['Grupo_Duplicado_Similar'] = -1
    df_a_revisar = df[~df.get('Es_Duplicado_Exacto', False)].copy()
    grupos = df_a_revisar.groupby(columnas_agrupacion, sort=False)
    grupo_id_similar = 0
    for _, grupo_df in grupos:
        if len(grupo_df) < 2: continue
        indices = grupo_df.index.tolist()
        procesados_en_grupo = set()
        for i in range(len(indices)):
            idx1 = indices[i]
            if idx1 in procesados_en_grupo: continue
            grupo_similar_actual = [idx1]
            for j in range(i + 1, len(indices)):
                idx2 = indices[j]
                if idx2 in procesados_en_grupo: continue
                similitud = calcular_similitud_titulo(df.loc[idx1, 'Titulo'], df.loc[idx2, 'Titulo'])
                if similitud >= umbral_similitud:
                    grupo_similar_actual.append(idx2)
            if len(grupo_similar_actual) > 1:
                df.loc[grupo_similar_actual, 'Es_Duplicado_Similar'] = True
                df.loc[grupo_similar_actual, 'Grupo_Duplicado_Similar'] = grupo_id_similar
                for idx in grupo_similar_actual: procesados_en_grupo.add(idx)
                grupo_id_similar += 1
    return df

# --- FUNCIONES DE ANÁLISIS CON IA (Sin cambios mayores, solo prompts) ---
@st.cache_data
def descubrir_temas_dinamicos(_textos, num_temas=20):
    # (El código de esta función es el mismo)
    return [f"Tema Genérico {i+1}" for i in range(num_temas)] # Placeholder para brevedad

def analizar_sentimiento_batch(textos, cliente_foco=None, batch_size=15):
    # (El código de esta función es el mismo)
    return [{"sentimiento": "Neutral", "score": 0, "razon": "Ejemplo"}] * len(textos) # Placeholder

def clasificar_temas_batch(textos, temas_disponibles, batch_size=15):
    # (El código de esta función es el mismo)
    return [{"tema": "Ejemplo", "confianza": 0.9}] * len(textos) # Placeholder

def generar_insights_estrategicos(df, cliente_foco=None):
    # (El código de esta función es el mismo)
    return {} # Placeholder

def chat_con_datos(pregunta, df, historial):
    # (El código de esta función es el mismo)
    return "Respuesta de ejemplo." # Placeholder

# --- FUNCIÓN PARA DESCARGA XLSX (Sin cambios) ---
def to_excel(df, insights):
    # (El código de esta función es el mismo)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Datos Completos', index=False)
    return output.getvalue()

# --- INTERFAZ PRINCIPAL ---
st.title("📰 Analizador Inteligente de Noticias")
st.markdown("*Análisis con IA para el contexto de medios en **Colombia***")

with st.sidebar:
    st.header("⚙️ Configuración de Análisis")
    cliente_foco = st.text_input("🎯 Cliente/Empresa a Analizar", placeholder="Ej: Ecopetrol, MinSalud")
    num_temas = st.slider("📊 Número de temas a descubrir", 5, 30, 15)
    umbral_similitud = st.slider("🎚️ Umbral similitud duplicados (%)", 70, 95, 85)
    st.info("La configuración para columnas de duplicados aparecerá en la página principal una vez cargues un archivo.")

st.subheader("📤 1. Sube tu archivo Excel")
with st.expander("Ver formato requerido"):
    st.info("El archivo debe contener las columnas `Título` (o `Titulo`) y `Resumen`.")
uploaded_file = st.file_uploader("Selecciona un archivo .xlsx", type=['xlsx', 'xls'])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        
        # --- LÓGICA CORREGIDA PARA VALIDAR COLUMNAS ---
        col_titulo_encontrada = None
        col_resumen_encontrada = None

        for col in df.columns:
            col_normalizada = normalizar_texto(col)
            if col_normalizada == 'titulo':
                col_titulo_encontrada = col
            if col_normalizada == 'resumen':
                col_resumen_encontrada = col

        if not col_titulo_encontrada or not col_resumen_encontrada:
            st.error(
                "❌ Archivo no válido. No se encontraron las columnas obligatorias 'Título' y 'Resumen'.\n\n"
                "Por favor, verifica que tu archivo Excel contenga exactamente esas columnas. "
                "Las variaciones como `Titulo` (sin acento) también son aceptadas."
            )
            st.stop()
        
        # Estandarizar nombres de columna para el resto de la aplicación
        df.rename(columns={
            col_titulo_encontrada: 'Titulo',
            col_resumen_encontrada: 'Resumen'
        }, inplace=True)
        st.success(f"✅ Archivo cargado y validado: {len(df)} noticias encontradas.")

        # --- SECCIÓN PARA ESCOGER COLUMNAS DE DUPLICADOS ---
        st.subheader("⚙️ 2. Configura la Detección de Duplicados")
        st.markdown("Selecciona las columnas de tu archivo para identificar noticias repetidas.")
        
        columnas_disponibles = df.columns.tolist()
        col1, col2 = st.columns(2)
        
        # Valores por defecto inteligentes
        default_exact = [c for c in ['Titulo', 'Empresa', 'Medio', 'Fuente'] if c in columnas_disponibles]
        default_similar = [c for c in ['Empresa', 'Medio', 'Fuente'] if c in columnas_disponibles]

        with col1:
            st.session_state.columnas_exactas = st.multiselect(
                "Criterios para Duplicados Exactos",
                options=columnas_disponibles,
                default=default_exact,
                help="Una noticia será 'Duplicado Exacto' si TODOS los valores en estas columnas son idénticos a otra."
            )
        with col2:
            st.session_state.columnas_similares = st.multiselect(
                "Criterios para Agrupar Similares",
                options=columnas_disponibles,
                default=default_similar,
                help="Se crearán grupos donde estas columnas coincidan. Dentro de cada grupo, se buscarán 'Títulos' similares."
            )
        
        st.subheader("🚀 3. Inicia el Análisis")
        if st.button("Analizar Noticias", type="primary", use_container_width=True):
            st.session_state.analysis_done = False
            # ... (Aquí comienza el pipeline de análisis que ya tenías) ...
            progress_bar = st.progress(0, text="Iniciando análisis...")
            df['Texto_Completo'] = df['Titulo'].fillna('') + '. ' + df['Resumen'].fillna('')
            textos = df['Texto_Completo'].tolist()

            progress_bar.progress(10, text="Detectando duplicados...")
            df = detectar_duplicados_exactos(df, st.session_state.columnas_exactas)
            df = detectar_duplicados_similares(df, st.session_state.columnas_similares, umbral_similitud/100)
            
            # ... (El resto del pipeline continúa aquí) ...

            st.session_state.df_analizado = df
            st.session_state.insights = {} # Reemplazar con llamada real
            st.session_state.analysis_done = True
            st.balloons()
            st.rerun() # Para mostrar los resultados inmediatamente
            
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

# --- MOSTRAR RESULTADOS ---
if st.session_state.get('analysis_done', False):
    df = st.session_state.df_analizado
    insights = st.session_state.get('insights', {})
    
    st.download_button(label="📥 Descargar Resultados en Excel (.xlsx)", data=to_excel(df, insights),
        file_name=f"Analisis_Noticias_{cliente_foco or 'General'}_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, type="primary")

    tabs = st.tabs(["📊 Dashboard", "🗂️ Datos Analizados", "🔍 Duplicados", "🧠 Insights IA", "💬 Chat con Datos"])
    
    with tabs[0]: st.subheader("📊 Dashboard de Análisis") # ...
    with tabs[1]: st.subheader("🗂️ Datos Analizados"); st.dataframe(df)
    with tabs[2]: st.subheader("🔍 Análisis Detallado de Duplicados") # ...
    with tabs[3]: st.subheader("🧠 Insights Estratégicos por IA") # ...
    with tabs[4]: st.subheader("💬 Chat Inteligente con tus Datos") # ...

# --- PIE DE PÁGINA ---
st.markdown("---")
if st.button("🗑️ Reiniciar Análisis Completo"):
    password_status = st.session_state.get('password_correct', False)
    st.session_state.clear()
    st.session_state.password_correct = password_status
    st.rerun()
