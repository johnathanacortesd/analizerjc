import streamlit as st
from groq import Groq
import pandas as pd
import json
import re
from datetime import datetime
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(
    page_title="Analizador de Noticias IA", 
    page_icon="📰", 
    layout="wide"
)

# --- AUTENTICACIÓN ---
if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

def validate_password():
    if st.session_state.get("password") == st.secrets.get("PASSWORD", "demo123"):
        st.session_state.password_correct = True
        if "password" in st.session_state:
            del st.session_state["password"]
    else:
        st.session_state.password_attempted = True

if not st.session_state.password_correct:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #2E86AB; font-size: 3rem;'>📰</h1>
        <h2>Analizador Inteligente de Noticias</h2>
        <p style='color: #666; margin-bottom: 2rem;'>Análisis de sentimiento y temas dinámicos con IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input("🔐 Contraseña", type="password", on_change=validate_password, key="password")
    
    if st.session_state.get("password_attempted", False):
        st.error("❌ Contraseña incorrecta")
    st.stop()

# --- INICIALIZACIÓN ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except KeyError:
    st.error("❌ Configure GROQ_API_KEY en Secrets")
    st.stop()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# --- FUNCIONES DE DETECCIÓN DE DUPLICADOS ---
def calcular_similitud_titulo(titulo1, titulo2):
    """Calcula similitud entre dos títulos usando SequenceMatcher"""
    if pd.isna(titulo1) or pd.isna(titulo2):
        return 0.0
    return SequenceMatcher(None, titulo1.lower().strip(), titulo2.lower().strip()).ratio()

def detectar_duplicados_exactos(df):
    """
    Detecta menciones duplicadas:
    - Empresa igual, Titulo igual y Medio igual
    """
    columnas_requeridas = ['Empresa', 'Titulo', 'Medio']
    for col in columnas_requeridas:
        if col not in df.columns:
            st.warning(f"⚠️ Columna '{col}' no encontrada. No se puede detectar duplicados exactos.")
            return df
    
    df['Es_Duplicado_Exacto'] = False
    df['Grupo_Duplicado_Exacto'] = -1
    
    # Crear clave única
    df['_clave_duplicado'] = df['Empresa'].fillna('').str.lower().str.strip() + '|||' + \
                             df['Titulo'].fillna('').str.lower().str.strip() + '|||' + \
                             df['Medio'].fillna('').str.lower().str.strip()
    
    # Identificar grupos de duplicados
    grupos_duplicados = df.groupby('_clave_duplicado').filter(lambda x: len(x) > 1)
    
    if not grupos_duplicados.empty:
        grupo_id = 0
        for clave, grupo in grupos_duplicados.groupby('_clave_duplicado'):
            indices = grupo.index.tolist()
            df.loc[indices, 'Es_Duplicado_Exacto'] = True
            df.loc[indices, 'Grupo_Duplicado_Exacto'] = grupo_id
            grupo_id += 1
    
    df.drop('_clave_duplicado', axis=1, inplace=True)
    return df

def detectar_duplicados_similares(df, umbral_similitud=0.85):
    """
    Detecta menciones similares:
    - Empresa igual, Titulo similar (>85%) y Medio igual
    """
    columnas_requeridas = ['Empresa', 'Titulo', 'Medio']
    for col in columnas_requeridas:
        if col not in df.columns:
            st.warning(f"⚠️ Columna '{col}' no encontrada. No se puede detectar duplicados similares.")
            return df
    
    df['Es_Duplicado_Similar'] = False
    df['Grupo_Duplicado_Similar'] = -1
    
    # Agrupar por Empresa + Medio
    grupos_empresa_medio = df.groupby([df['Empresa'].fillna('').str.lower().str.strip(), 
                                       df['Medio'].fillna('').str.lower().str.strip()])
    
    grupo_id_similar = 0
    procesados = set()
    
    for (empresa, medio), grupo in grupos_empresa_medio:
        if len(grupo) < 2:
            continue
        
        indices_grupo = grupo.index.tolist()
        
        # Comparar títulos dentro del grupo
        for i, idx1 in enumerate(indices_grupo):
            if idx1 in procesados:
                continue
            
            grupo_similar = [idx1]
            
            for idx2 in indices_grupo[i+1:]:
                if idx2 in procesados:
                    continue
                
                titulo1 = df.loc[idx1, 'Titulo']
                titulo2 = df.loc[idx2, 'Titulo']
                
                similitud = calcular_similitud_titulo(titulo1, titulo2)
                
                if similitud >= umbral_similitud:
                    grupo_similar.append(idx2)
                    procesados.add(idx2)
            
            if len(grupo_similar) > 1:
                df.loc[grupo_similar, 'Es_Duplicado_Similar'] = True
                df.loc[grupo_similar, 'Grupo_Duplicado_Similar'] = grupo_id_similar
                procesados.add(idx1)
                grupo_id_similar += 1
    
    return df

# --- FUNCIONES DE ANÁLISIS CON IA ---
def descubrir_temas_dinamicos(textos, num_temas=20):
    """
    Descubre temas automáticamente del contenido usando IA
    """
    try:
        # Tomar muestra representativa
        muestra_size = min(100, len(textos))
        muestra_indices = np.random.choice(len(textos), muestra_size, replace=False)
        muestra_textos = [textos[i][:400] for i in muestra_indices]
        
        textos_muestra = "\n\n".join([f"{i+1}. {texto}" for i, texto in enumerate(muestra_textos[:30])])
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"""Eres un analista experto en clasificación de contenido. Analiza las noticias y descubre los {num_temas} TEMAS PRINCIPALES que mejor representan el contenido.

INSTRUCCIONES:
1. Identifica los {num_temas} temas más relevantes presentes en las noticias
2. Los temas deben ser ESPECÍFICOS y DESCRIPTIVOS del contenido real
3. Evita temas genéricos como "Noticias" u "Otros"
4. Usa nombres cortos y claros (máximo 4 palabras)
5. Los temas deben ser mutuamente excluyentes

FORMATO DE SALIDA (JSON):
{{
  "temas": [
    {{"id": 1, "nombre": "Nombre del Tema", "descripcion": "Breve descripción"}},
    ...
  ]
}}"""},
                {"role": "user", "content": f"Analiza estas {muestra_size} noticias y descubre los {num_temas} temas principales:\n\n{textos_muestra}"}
            ],
            model="llama-3.1-70b-versatile",
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        resultado = json.loads(chat_completion.choices[0].message.content)
        temas = resultado.get('temas', [])
        
        # Extraer solo nombres de temas
        nombres_temas = [tema.get('nombre', f'Tema {i+1}') for i, tema in enumerate(temas)]
        
        return nombres_temas[:num_temas]
        
    except Exception as e:
        st.error(f"Error descubriendo temas: {e}")
        return [f"Tema {i+1}" for i in range(num_temas)]

def analizar_sentimiento_batch(textos, cliente_foco=None, batch_size=15):
    """Analiza sentimiento de múltiples noticias en batches"""
    resultados = []
    
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i+batch_size]
        
        try:
            prompt_cliente = f"\n- IMPORTANTE: Analiza el sentimiento específicamente desde la perspectiva de '{cliente_foco}' (cómo le afecta a este cliente/empresa)." if cliente_foco else ""
            
            textos_numerados = "\n".join([f"{j+1}. {texto[:350]}" for j, texto in enumerate(batch)])
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"""Eres un analista de medios experto. Analiza el sentimiento de cada noticia.{prompt_cliente}

ESCALA DE SENTIMIENTO:
- Muy Positivo (+2): Noticias excelentes, logros importantes, beneficios claros
- Positivo (+1): Noticias favorables, desarrollos constructivos
- Neutral (0): Informativo, sin carga emocional clara
- Negativo (-1): Problemas, críticas, situaciones desfavorables
- Muy Negativo (-2): Crisis, escándalos, situaciones graves

FORMATO DE SALIDA (JSON):
{{"analisis": [
  {{"id": 1, "sentimiento": "Positivo", "score": 1, "razon": "Breve explicación"}},
  ...
]}}"""},
                    {"role": "user", "content": f"Analiza el sentimiento de estas noticias:\n\n{textos_numerados}"}
                ],
                model="llama-3.1-70b-versatile",
                temperature=0.1,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            
            resultado = json.loads(chat_completion.choices[0].message.content)
            resultados.extend(resultado.get('analisis', []))
            
        except Exception as e:
            st.warning(f"Error en batch {i//batch_size + 1}: {e}")
            # Rellenar con valores neutros
            for _ in batch:
                resultados.append({"sentimiento": "Neutral", "score": 0, "razon": "Error en análisis"})
    
    return resultados

def clasificar_temas_batch(textos, temas_disponibles, batch_size=15):
    """Clasifica múltiples noticias en los temas descubiertos"""
    resultados = []
    
    temas_str = "\n".join([f"{i+1}. {tema}" for i, tema in enumerate(temas_disponibles)])
    
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i+batch_size]
        
        try:
            textos_numerados = "\n".join([f"{j+1}. {texto[:350]}" for j, texto in enumerate(batch)])
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"""Eres un clasificador de noticias experto. Clasifica cada noticia en UNO de estos temas descubiertos:

{temas_str}

REGLAS:
- Asigna el tema MÁS RELEVANTE para cada noticia
- Si no encaja claramente, elige el más cercano
- Sé consistente: noticias similares deben tener el mismo tema

FORMATO DE SALIDA (JSON):
{{"clasificacion": [
  {{"id": 1, "tema": "Nombre exacto del tema", "confianza": 0.95}},
  ...
]}}"""},
                    {"role": "user", "content": f"Clasifica estas noticias:\n\n{textos_numerados}"}
                ],
                model="llama-3.1-70b-versatile",
                temperature=0.1,
                max_tokens=2500,
                response_format={"type": "json_object"}
            )
            
            resultado = json.loads(chat_completion.choices[0].message.content)
            resultados.extend(resultado.get('clasificacion', []))
            
        except Exception as e:
            st.warning(f"Error en clasificación batch {i//batch_size + 1}: {e}")
            for _ in batch:
                resultados.append({"tema": temas_disponibles[0] if temas_disponibles else "Sin clasificar", "confianza": 0})
    
    return resultados

def generar_insights_estrategicos(df, cliente_foco=None):
    """Genera insights estratégicos del análisis completo"""
    try:
        total_noticias = len(df)
        distribucion_sentimiento = df['Sentimiento'].value_counts().to_dict() if 'Sentimiento' in df.columns else {}
        temas_principales = df['Tema'].value_counts().head(5).to_dict() if 'Tema' in df.columns else {}
        
        duplicados_exactos = len(df[df['Es_Duplicado_Exacto']]) if 'Es_Duplicado_Exacto' in df.columns else 0
        duplicados_similares = len(df[df['Es_Duplicado_Similar']]) if 'Es_Duplicado_Similar' in df.columns else 0
        
        prompt_cliente = f"\n\nCLIENTE FOCO: '{cliente_foco}' - Genera insights específicos sobre cómo estas noticias afectan a este cliente/empresa." if cliente_foco else ""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"""Eres un analista estratégico de medios. Genera insights accionables y de alto valor.

FORMATO DE SALIDA (JSON):
{{
  "resumen_ejecutivo": "Párrafo ejecutivo de 2-3 líneas",
  "tendencias_clave": ["tendencia 1", "tendencia 2", "tendencia 3"],
  "oportunidades": ["oportunidad 1", "oportunidad 2"],
  "riesgos": ["riesgo 1", "riesgo 2"],
  "recomendaciones": ["acción 1", "acción 2", "acción 3"],
  "hallazgos_duplicados": "Análisis de duplicados encontrados"
}}"""},
                {"role": "user", "content": f"""Analiza estos datos de noticias:

Total de noticias: {total_noticias}
Duplicados exactos: {duplicados_exactos}
Duplicados similares: {duplicados_similares}
Distribución de sentimiento: {distribucion_sentimiento}
Temas principales: {temas_principales}

Primeras 5 noticias como muestra:
{df[['Titulo']].head().to_string() if 'Titulo' in df.columns else 'N/A'}
{prompt_cliente}"""}
            ],
            model="llama-3.1-70b-versatile",
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generando insights: {e}")
        return {}

def chat_con_datos(pregunta, df, historial):
    """Chat inteligente con el dataset de noticias"""
    try:
        contexto_datos = f"""
DATASET DE NOTICIAS:
- Total de noticias: {len(df)}
- Columnas disponibles: {', '.join(df.columns)}
- Temas principales: {df['Tema'].value_counts().head(3).to_dict() if 'Tema' in df.columns else 'N/A'}
- Sentimientos: {df['Sentimiento'].value_counts().to_dict() if 'Sentimiento' in df.columns else 'N/A'}
- Duplicados exactos: {len(df[df['Es_Duplicado_Exacto']]) if 'Es_Duplicado_Exacto' in df.columns else 0}
- Duplicados similares: {len(df[df['Es_Duplicado_Similar']]) if 'Es_Duplicado_Similar' in df.columns else 0}

MUESTRA DE DATOS (5 noticias):
{df.head().to_string()}
"""
        
        mensajes = [
            {"role": "system", "content": f"""Eres un analista de datos experto. Responde preguntas sobre el dataset de noticias.

INSTRUCCIONES:
- Usa los datos proporcionados para responder
- Sé específico y usa números/estadísticas cuando sea posible
- Si la información no está en los datos, indícalo claramente
- Proporciona insights accionables

{contexto_datos}"""}
        ]
        
        for item in historial[-5:]:
            mensajes.append({"role": "user", "content": item["pregunta"]})
            mensajes.append({"role": "assistant", "content": item["respuesta"]})
        
        mensajes.append({"role": "user", "content": pregunta})
        
        chat_completion = client.chat.completions.create(
            messages=mensajes,
            model="llama-3.1-70b-versatile",
            temperature=0.2,
            max_tokens=1500
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error al procesar la pregunta: {e}"

# --- INTERFAZ PRINCIPAL ---
st.title("📰 Analizador Inteligente de Noticias")
st.markdown("*Análisis de sentimiento con temas dinámicos y detección de duplicados*")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    cliente_foco = st.text_input(
        "🎯 Cliente/Empresa a Analizar",
        placeholder="Ej: Ministerio de Salud, Ecopetrol, etc.",
        help="El análisis se enfocará en cómo las noticias afectan a este cliente"
    )
    
    st.markdown("---")
    st.subheader("🎛️ Opciones de Análisis")
    
    num_temas = st.slider("📊 Número de temas a descubrir", 5, 30, 15, 
                          help="La IA descubrirá automáticamente los temas del contenido")
    
    analizar_sentimiento = st.checkbox("💭 Análisis de Sentimiento", value=True)
    clasificar_temas = st.checkbox("🏷️ Clasificación Temática", value=True)
    detectar_duplicados = st.checkbox("🔍 Detectar Duplicados", value=True)
    generar_insights = st.checkbox("🧠 Insights Estratégicos", value=True)
    
    umbral_similitud = st.slider("🎚️ Umbral similitud (%)", 70, 95, 85,
                                 help="Porcentaje de similitud para detectar duplicados")
    
    st.markdown("---")
    st.info("""
    📋 **Columnas requeridas:**
    - `Titulo` (obligatorio)
    - `Resumen` (obligatorio)
    - `Empresa` (para duplicados)
    - `Medio` (para duplicados)
    """)

# --- CARGA DE ARCHIVO ---
st.subheader("📤 Sube tu archivo Excel")
uploaded_file = st.file_uploader(
    "Selecciona un archivo XLSX con columnas 'Titulo' y 'Resumen'",
    type=['xlsx', 'xls'],
    help="Opcionalmente incluye 'Empresa' y 'Medio' para detección de duplicados"
)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        
        # Validar columnas requeridas
        if 'Titulo' not in df.columns or 'Resumen' not in df.columns:
            st.error("❌ El archivo debe contener las columnas 'Titulo' y 'Resumen'")
            st.stop()
        
        st.success(f"✅ Archivo cargado: {len(df)} noticias encontradas")
        
        # Validar columnas para duplicados
        columnas_duplicados = ['Empresa', 'Medio']
        columnas_faltantes = [col for col in columnas_duplicados if col not in df.columns]
        
        if columnas_faltantes and detectar_duplicados:
            st.warning(f"⚠️ Columnas faltantes para detección completa de duplicados: {', '.join(columnas_faltantes)}")
            st.info("💡 La detección de duplicados será limitada sin estas columnas")
        
        # Mostrar vista previa
        with st.expander("👀 Vista previa de datos"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # --- BOTÓN DE ANÁLISIS ---
        if st.button("🚀 Iniciar Análisis Inteligente", type="primary", use_container_width=True):
            st.session_state.analysis_done = False
            st.session_state.chat_history = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Preparar textos para análisis
            df['Texto_Completo'] = df['Titulo'].fillna('') + '. ' + df['Resumen'].fillna('')
            textos = df['Texto_Completo'].tolist()
            
            # 1. Descubrir Temas Dinámicamente
            if clasificar_temas:
                status_text.text("🔍 Descubriendo temas automáticamente...")
                progress_bar.progress(15)
                
                with st.spinner("Analizando el contenido para descubrir temas..."):
                    temas_descubiertos = descubrir_temas_dinamicos(textos, num_temas)
                    st.session_state.temas_descubiertos = temas_descubiertos
                
                st.success(f"✅ {len(temas_descubiertos)} temas descubiertos automáticamente")
                with st.expander("📋 Ver temas descubiertos"):
                    for i, tema in enumerate(temas_descubiertos, 1):
                        st.markdown(f"{i}. **{tema}**")
            
            # 2. Detectar Duplicados
            if detectar_duplicados:
                status_text.text("🔍 Detectando duplicados...")
                progress_bar.progress(30)
                
                # Duplicados exactos
                df = detectar_duplicados_exactos(df)
                num_duplicados_exactos = len(df[df['Es_Duplicado_Exacto']])
                
                # Duplicados similares
                df = detectar_duplicados_similares(df, umbral_similitud/100)
                num_duplicados_similares = len(df[df['Es_Duplicado_Similar']])
                
                st.info(f"📊 Duplicados exactos: {num_duplicados_exactos} | Duplicados similares: {num_duplicados_similares}")
            
            # 3. Análisis de Sentimiento (solo no duplicados)
            if analizar_sentimiento:
                status_text.text("💭 Analizando sentimiento...")
                progress_bar.progress(50)
                
                # Filtrar no duplicados para análisis
                mask_no_duplicados = ~(df.get('Es_Duplicado_Exacto', False) | df.get('Es_Duplicado_Similar', False))
                df_analizar = df[mask_no_duplicados]
                
                textos_analizar = df_analizar['Texto_Completo'].tolist()
                
                if len(textos_analizar) > 0:
                    resultados_sentimiento = analizar_sentimiento_batch(textos_analizar, cliente_foco)
                    
                    # Asignar resultados a no duplicados
                    df.loc[mask_no_duplicados, 'Sentimiento'] = [r.get('sentimiento', 'Neutral') for r in resultados_sentimiento]
                    df.loc[mask_no_duplicados, 'Score_Sentimiento'] = [r.get('score', 0) for r in resultados_sentimiento]
                    df.loc[mask_no_duplicados, 'Razon_Sentimiento'] = [r.get('razon', '') for r in resultados_sentimiento]
                    
                    # Propagar sentimiento a duplicados del mismo grupo
                    for grupo_col in ['Grupo_Duplicado_Exacto', 'Grupo_Duplicado_Similar']:
                        if grupo_col in df.columns:
                            for grupo_id in df[grupo_col].unique():
                                if grupo_id >= 0:
                                    grupo_mask = df[grupo_col] == grupo_id
                                    # Encontrar el primero no duplicado del grupo (si existe)
                                    primero_analizado = df[grupo_mask & mask_no_duplicados]
                                    if not primero_analizado.empty:
                                        sentimiento_grupo = primero_analizado.iloc[0]['Sentimiento']
                                        score_grupo = primero_analizado.iloc[0]['Score_Sentimiento']
                                        razon_grupo = primero_analizado.iloc[0]['Razon_Sentimiento']
                                        
                                        df.loc[grupo_mask, 'Sentimiento'] = sentimiento_grupo
                                        df.loc[grupo_mask, 'Score_Sentimiento'] = score_grupo
                                        df.loc[grupo_mask, 'Razon_Sentimiento'] = razon_grupo + " [Duplicado]"
                    
                    # Marcar duplicados sin análisis
                    duplicados_sin_analisis = df[~mask_no_duplicados & df['Sentimiento'].isna()]
                    if not duplicados_sin_analisis.empty:
                        df.loc[duplicados_sin_analisis.index, 'Sentimiento'] = 'Sin Analizar'
                        df.loc[duplicados_sin_analisis.index, 'Score_Sentimiento'] = 0
                        df.loc[duplicados_sin_analisis.index, 'Razon_Sentimiento'] = 'Duplicado sin análisis'
            
            # 4. Clasificación Temática (solo no duplicados)
            if clasificar_temas and 'temas_descubiertos' in st.session_state:
                status_text.text("🏷️ Clasificando temas...")
                progress_bar.progress(70)
                
                temas_disponibles = st.session_state.temas_descubiertos
                
                # Filtrar no duplicados
                mask_no_duplicados = ~(df.get('Es_Duplicado_Exacto', False) | df.get('Es_Duplicado_Similar', False))
                df_analizar = df[mask_no_duplicados]
                
                textos_analizar = df_analizar['Texto_Completo'].tolist()
                
                if len(textos_analizar) > 0:
                    resultados_temas = clasificar_temas_batch(textos_analizar, temas_disponibles)
                    
                    # Asignar resultados
                    df.loc[mask_no_duplicados, 'Tema'] = [r.get('tema', 'Sin clasificar') for r in resultados_temas]
                    df.loc[mask_no_duplicados, 'Confianza_Tema'] = [r.get('confianza', 0) for r in resultados_temas]
                    
                    # Propagar tema a duplicados del mismo grupo
                    for grupo_col in ['Grupo_Duplicado_Exacto', 'Grupo_Duplicado_Similar']:
                        if grupo_col in df.columns:
                            for grupo_id in df[grupo_col].unique():
                                if grupo_id >= 0:
                                    grupo_mask = df[grupo_col] == grupo_id
                                    primero_analizado = df[grupo_mask & mask_no_duplicados]
                                    if not primero_analizado.empty:
                                        tema_grupo = primero_analizado.iloc[0]['Tema']
                                        confianza_grupo = primero_analizado.iloc[0]['Confianza_Tema']
                                        
                                        df.loc[grupo_mask, 'Tema'] = tema_grupo
                                        df.loc[grupo_mask, 'Confianza_Tema'] = confianza_grupo
                    
                    # Marcar duplicados sin clasificar
                    duplicados_sin_clasificar = df[~mask_no_duplicados & df['Tema'].isna()]
                    if not duplicados_sin_clasificar.empty:
                        df.loc[duplicados_sin_clasificar.index, 'Tema'] = 'Sin clasificar'
                        df.loc[duplicados_sin_clasificar.index, 'Confianza_Tema'] = 0
            
            # 5. Insights Estratégicos
            if generar_insights:
                status_text.text("🧠 Generando insights...")
                progress_bar.progress(90)
                
                insights = generar_insights_estrategicos(df, cliente_foco)
                st.session_state.insights = insights
            
            progress_bar.progress(100)
            status_text.text("✅ ¡Análisis completado!")
            
            st.session_state.df_analizado = df
            st.session_state.analysis_done = True
            st.balloons()
            
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

# --- MOSTRAR RESULTADOS ---
if st.session_state.get('analysis_done', False):
    df = st.session_state.df_analizado
    
    st.markdown("---")
    
    # TABS PRINCIPALES
    tabs = st.tabs([
        "📊 Dashboard", 
        "🗂️ Datos Analizados", 
        "🔍 Duplicados",
        "🧠 Insights", 
        "💬 Chat IA"
    ])
    
    # TAB 1: DASHBOARD
    with tabs[0]:
        st.subheader("📊 Dashboard de Análisis")
        
        # Métricas principales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📰 Total Noticias", len(df))
        
        with col2:
            if 'Es_Duplicado_Exacto' in df.columns:
                duplicados_exactos = len(df[df['Es_Duplicado_Exacto']])
                st.metric("🔴 Duplicados Exactos", duplicados_exactos)
        
        with col3:
            if 'Es_Duplicado_Similar' in df.columns:
                duplicados_similares = len(df[df['Es_Duplicado_Similar']])
                st.metric("🟡 Duplicados Similares", duplicados_similares)
        
        with col4:
            noticias_unicas = len(df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)])
            st.metric("✅ Noticias Únicas", noticias_unicas)
        
        with col5:
            if 'Tema' in df.columns:
                temas_unicos = df['Tema'].nunique()
                st.metric("🏷️ Temas Descubiertos", temas_unicos)
        
        st.markdown("---")
        
        # Gráficos
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            if 'Sentimiento' in df.columns:
                st.markdown("#### 💭 Distribución de Sentimientos")
                # Filtrar solo noticias únicas para el gráfico
                df_unicos = df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)]
                
                fig_sentiment = px.pie(
                    df_unicos, 
                    names='Sentimiento',
                    title='(Solo noticias únicas analizadas)',
                    color='Sentimiento',
                    color_discrete_map={
                        'Muy Positivo': '#2ECC71',
                        'Positivo': '#85C1E2',
                        'Neutral': '#95A5A6',
                        'Negativo': '#E67E22',
                        'Muy Negativo': '#E74C3C',
                        'Sin Analizar': '#BDC3C7'
                    }
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col_g2:
            if 'Tema' in df.columns:
                st.markdown("#### 🏷️ Top 10 Temas Descubiertos")
                df_unicos = df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)]
                temas_count = df_unicos['Tema'].value_counts().head(10)
                
                fig_temas = px.bar(
                    x=temas_count.values,
                    y=temas_count.index,
                    orientation='h',
                    title='(Solo noticias únicas)',
                    labels={'x': 'Cantidad', 'y': 'Tema'}
                )
                fig_temas.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_temas, use_container_width=True)
        
        # Análisis por tema y sentimiento
        if 'Tema' in df.columns and 'Sentimiento' in df.columns:
            st.markdown("---")
            st.markdown("#### 🎯 Sentimiento por Tema (Noticias Únicas)")
            
            df_unicos = df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)]
            
            if not df_unicos.empty:
                tema_sentimiento = df_unicos.groupby(['Tema', 'Sentimiento']).size().reset_index(name='Cantidad')
                fig_heatmap = px.density_heatmap(
                    tema_sentimiento,
                    x='Sentimiento',
                    y='Tema',
                    z='Cantidad',
                    title='',
                    color_continuous_scale='Blues'
                )
                fig_heatmap.update_layout(height=500)
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Estadísticas de duplicados
        if 'Es_Duplicado_Exacto' in df.columns or 'Es_Duplicado_Similar' in df.columns:
            st.markdown("---")
            st.markdown("#### 📊 Análisis de Duplicados")
            
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                duplicados_exactos = len(df[df.get('Es_Duplicado_Exacto', False)])
                porcentaje_exactos = (duplicados_exactos / len(df) * 100) if len(df) > 0 else 0
                st.metric("Duplicados Exactos", f"{duplicados_exactos}", f"{porcentaje_exactos:.1f}%")
            
            with col_d2:
                duplicados_similares = len(df[df.get('Es_Duplicado_Similar', False)])
                porcentaje_similares = (duplicados_similares / len(df) * 100) if len(df) > 0 else 0
                st.metric("Duplicados Similares", f"{duplicados_similares}", f"{porcentaje_similares:.1f}%")
            
            with col_d3:
                total_duplicados = duplicados_exactos + duplicados_similares
                porcentaje_total = (total_duplicados / len(df) * 100) if len(df) > 0 else 0
                st.metric("Total Duplicados", f"{total_duplicados}", f"{porcentaje_total:.1f}%")
    
    # TAB 2: DATOS ANALIZADOS
    with tabs[1]:
        st.subheader("🗂️ Datos Analizados")
        
        # Filtros
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        with col_f1:
            if 'Tema' in df.columns:
                temas_filtro = ['Todos'] + sorted(df['Tema'].unique().tolist())
                tema_seleccionado = st.selectbox("Filtrar por Tema", temas_filtro)
        
        with col_f2:
            if 'Sentimiento' in df.columns:
                sentimientos_filtro = ['Todos'] + sorted(df['Sentimiento'].unique().tolist())
                sentimiento_seleccionado = st.selectbox("Filtrar por Sentimiento", sentimientos_filtro)
        
        with col_f3:
            tipo_duplicado = st.selectbox("Tipo de noticia", 
                                         ['Todas', 'Solo Únicas', 'Solo Duplicados Exactos', 'Solo Duplicados Similares'])
        
        with col_f4:
            busqueda = st.text_input("🔍 Buscar texto")
        
        # Aplicar filtros
        df_filtrado = df.copy()
        
        if 'Tema' in df.columns and tema_seleccionado != 'Todos':
            df_filtrado = df_filtrado[df_filtrado['Tema'] == tema_seleccionado]
        
        if 'Sentimiento' in df.columns and sentimiento_seleccionado != 'Todos':
            df_filtrado = df_filtrado[df_filtrado['Sentimiento'] == sentimiento_seleccionado]
        
        if tipo_duplicado == 'Solo Únicas':
            df_filtrado = df_filtrado[~df_filtrado.get('Es_Duplicado_Exacto', False) & ~df_filtrado.get('Es_Duplicado_Similar', False)]
        elif tipo_duplicado == 'Solo Duplicados Exactos':
            df_filtrado = df_filtrado[df_filtrado.get('Es_Duplicado_Exacto', False)]
        elif tipo_duplicado == 'Solo Duplicados Similares':
            df_filtrado = df_filtrado[df_filtrado.get('Es_Duplicado_Similar', False)]
        
        if busqueda:
            mask = df_filtrado['Titulo'].str.contains(busqueda, case=False, na=False) | \
                   df_filtrado['Resumen'].str.contains(busqueda, case=False, na=False)
            df_filtrado = df_filtrado[mask]
        
        st.info(f"📊 Mostrando {len(df_filtrado)} de {len(df)} noticias")
        
        # Mostrar datos
        columnas_mostrar = ['Titulo', 'Resumen']
        
        if 'Empresa' in df.columns:
            columnas_mostrar.append('Empresa')
        if 'Medio' in df.columns:
            columnas_mostrar.append('Medio')
        if 'Sentimiento' in df.columns:
            columnas_mostrar.extend(['Sentimiento', 'Score_Sentimiento'])
        if 'Tema' in df.columns:
            columnas_mostrar.append('Tema')
        if 'Es_Duplicado_Exacto' in df.columns:
            columnas_mostrar.append('Es_Duplicado_Exacto')
        if 'Es_Duplicado_Similar' in df.columns:
            columnas_mostrar.append('Es_Duplicado_Similar')
        
        # Filtrar solo columnas que existen
        columnas_mostrar = [col for col in columnas_mostrar if col in df_filtrado.columns]
        
        st.dataframe(
            df_filtrado[columnas_mostrar].style.apply(
                lambda x: ['background-color: #ffe6e6' if x.get('Es_Duplicado_Exacto', False) 
                          else 'background-color: #fff4e6' if x.get('Es_Duplicado_Similar', False) 
                          else '' for i in x], 
                axis=1
            ),
            use_container_width=True,
            height=500
        )
        
        # Leyenda de colores
        st.markdown("""
        **Leyenda:**
        - 🔴 <span style='background-color: #ffe6e6; padding: 2px 8px;'>Duplicado Exacto</span>
        - 🟡 <span style='background-color: #fff4e6; padding: 2px 8px;'>Duplicado Similar</span>
        - ⚪ Sin color: Noticia única
        """, unsafe_allow_html=True)
        
        # Descargar resultados
        st.markdown("---")
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Descargar CSV Completo",
                csv,
                "analisis_noticias_completo.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col_d2:
            # Solo noticias únicas
            df_unicos = df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)]
            csv_unicos = df_unicos.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Descargar Solo Únicas",
                csv_unicos,
                "analisis_noticias_unicas.csv",
                "text/csv",
                use_container_width=True
            )
    
    # TAB 3: DUPLICADOS
    with tabs[2]:
        st.subheader("🔍 Análisis Detallado de Duplicados")
        
        # Estadísticas generales
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            if 'Grupo_Duplicado_Exacto' in df.columns:
                grupos_exactos = df[df['Grupo_Duplicado_Exacto'] >= 0]['Grupo_Duplicado_Exacto'].nunique()
                st.metric("📑 Grupos Duplicados Exactos", grupos_exactos)
        
        with col_stat2:
            if 'Grupo_Duplicado_Similar' in df.columns:
                grupos_similares = df[df['Grupo_Duplicado_Similar'] >= 0]['Grupo_Duplicado_Similar'].nunique()
                st.metric("📑 Grupos Duplicados Similares", grupos_similares)
        
        with col_stat3:
            noticias_unicas = len(df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)])
            st.metric("✅ Noticias Únicas", noticias_unicas)
        
        st.markdown("---")
        
        # Selector de tipo de duplicado
        tipo_ver = st.radio("Ver duplicados:", ['Exactos', 'Similares'], horizontal=True)
        
        if tipo_ver == 'Exactos' and 'Grupo_Duplicado_Exacto' in df.columns:
            st.markdown("#### 🔴 Duplicados Exactos")
            st.info("Empresa + Titulo + Medio idénticos")
            
            grupos = df[df['Grupo_Duplicado_Exacto'] >= 0].groupby('Grupo_Duplicado_Exacto')
            
            if len(grupos) > 0:
                for grupo_id, grupo_df in grupos:
                    with st.expander(f"📑 Grupo {int(grupo_id) + 1} - {len(grupo_df)} menciones idénticas", expanded=False):
                        # Información del grupo
                        col_info1, col_info2, col_info3 = st.columns(3)
                        
                        with col_info1:
                            if 'Empresa' in grupo_df.columns:
                                st.markdown(f"**🏢 Empresa:** {grupo_df.iloc[0]['Empresa']}")
                        with col_info2:
                            if 'Medio' in grupo_df.columns:
                                st.markdown(f"**📰 Medio:** {grupo_df.iloc[0]['Medio']}")
                        with col_info3:
                            if 'Sentimiento' in grupo_df.columns:
                                st.markdown(f"**💭 Sentimiento:** {grupo_df.iloc[0]['Sentimiento']}")
                        
                        st.markdown("---")
                        st.markdown(f"**📰 Título:** {grupo_df.iloc[0]['Titulo']}")
                        st.markdown(f"**📝 Resumen:** {grupo_df.iloc[0]['Resumen'][:300]}...")
                        
                        if 'Tema' in grupo_df.columns:
                            st.markdown(f"**🏷️ Tema:** {grupo_df.iloc[0]['Tema']}")
            else:
                st.success("✅ No se encontraron duplicados exactos")
        
        elif tipo_ver == 'Similares' and 'Grupo_Duplicado_Similar' in df.columns:
            st.markdown("#### 🟡 Duplicados Similares")
            st.info(f"Empresa + Medio iguales, Título similar (≥{umbral_similitud}%)")
            
            grupos = df[df['Grupo_Duplicado_Similar'] >= 0].groupby('Grupo_Duplicado_Similar')
            
            if len(grupos) > 0:
                for grupo_id, grupo_df in grupos:
                    with st.expander(f"📑 Grupo {int(grupo_id) + 1} - {len(grupo_df)} menciones similares", expanded=False):
                        # Información del grupo
                        col_info1, col_info2, col_info3 = st.columns(3)
                        
                        with col_info1:
                            if 'Empresa' in grupo_df.columns:
                                st.markdown(f"**🏢 Empresa:** {grupo_df.iloc[0]['Empresa']}")
                        with col_info2:
                            if 'Medio' in grupo_df.columns:
                                st.markdown(f"**📰 Medio:** {grupo_df.iloc[0]['Medio']}")
                        with col_info3:
                            if 'Sentimiento' in grupo_df.columns:
                                st.markdown(f"**💭 Sentimiento:** {grupo_df.iloc[0]['Sentimiento']}")
                        
                        st.markdown("---")
                        
                        # Mostrar todas las variaciones del título
                        for idx, row in grupo_df.iterrows():
                            st.markdown(f"**Variación {idx+1}:**")
                            st.markdown(f"📰 {row['Titulo']}")
                            if 'Tema' in row:
                                st.markdown(f"🏷️ {row['Tema']}")
                            st.markdown("---")
            else:
                st.success("✅ No se encontraron duplicados similares")
    
    # TAB 4: INSIGHTS
    with tabs[3]:
        st.subheader("🧠 Insights Estratégicos")
        
        if 'insights' in st.session_state and st.session_state.insights:
            insights = st.session_state.insights
            
            # Resumen Ejecutivo
            if 'resumen_ejecutivo' in insights:
                st.markdown("### 📋 Resumen Ejecutivo")
                st.info(insights['resumen_ejecutivo'])
            
            # Hallazgos de duplicados
            if 'hallazgos_duplicados' in insights:
                st.markdown("### 🔍 Análisis de Duplicados")
                st.warning(insights['hallazgos_duplicados'])
            
            st.markdown("---")
            
            col_i1, col_i2 = st.columns(2)
            
            with col_i1:
                # Tendencias
                if 'tendencias_clave' in insights:
                    st.markdown("### 📈 Tendencias Clave")
                    for i, tendencia in enumerate(insights['tendencias_clave'], 1):
                        st.markdown(f"**{i}.** {tendencia}")
                
                st.markdown("")
                
                # Oportunidades
                if 'oportunidades' in insights:
                    st.markdown("### ✅ Oportunidades")
                    for i, oportunidad in enumerate(insights['oportunidades'], 1):
                        st.success(f"**{i}.** {oportunidad}")
            
            with col_i2:
                # Riesgos
                if 'riesgos' in insights:
                    st.markdown("### ⚠️ Riesgos")
                    for i, riesgo in enumerate(insights['riesgos'], 1):
                        st.warning(f"**{i}.** {riesgo}")
                
                st.markdown("")
                
                # Recomendaciones
                if 'recomendaciones' in insights:
                    st.markdown("### 💡 Recomendaciones")
                    for i, recomendacion in enumerate(insights['recomendaciones'], 1):
                        st.markdown(f"**{i}.** {recomendacion}")
        else:
            st.info("No se generaron insights. Activa la opción en el sidebar y vuelve a analizar.")
    
    # TAB 5: CHAT IA
    with tabs[4]:
        st.subheader("💬 Chat Inteligente con tus Datos")
        st.markdown("*Haz preguntas sobre el análisis de noticias*")
        
        # Mostrar historial
        for item in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(item["pregunta"])
            with st.chat_message("assistant"):
                st.markdown(item["respuesta"])
        
        # Input de pregunta
        pregunta_usuario = st.chat_input("Escribe tu pregunta aquí...")
        
        if pregunta_usuario:
            # Agregar pregunta al chat
            with st.chat_message("user"):
                st.markdown(pregunta_usuario)
            
            # Generar respuesta
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analizando..."):
                    respuesta = chat_con_datos(
                        pregunta_usuario, 
                        df, 
                        st.session_state.chat_history
                    )
                    st.markdown(respuesta)
            
            # Guardar en historial
            st.session_state.chat_history.append({
                "pregunta": pregunta_usuario,
                "respuesta": respuesta
            })
            st.rerun()
        
        # Botón para limpiar chat
        if st.session_state.chat_history:
            if st.button("🗑️ Limpiar Conversación"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Preguntas sugeridas
        st.markdown("---")
        st.markdown("**💡 Preguntas sugeridas:**")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        sugerencias = [
            "¿Cuáles son los temas más mencionados?",
            "¿Qué porcentaje de noticias son duplicadas?",
            "¿Cuál es el sentimiento general?",
            "¿Qué empresa tiene más menciones?",
            "Resume las noticias más negativas",
            "¿Qué tema tiene peor sentimiento?"
        ]
        
        for i, sugerencia in enumerate(sugerencias[:3]):
            with col_s1 if i == 0 else col_s2 if i == 1 else col_s3:
                if st.button(sugerencia, key=f"sug_{i}"):
                    st.session_state.chat_history.append({
                        "pregunta": sugerencia,
                        "respuesta": "Procesando..."
                    })
                    st.rerun()

# --- PIE DE PÁGINA ---
st.markdown("---")
if st.button("🗑️ Reiniciar Análisis Completo"):
    for key in list(st.session_state.keys()):
        if key not in ['password_correct']:
            del st.session_state[key]
    st.rerun()

st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p><strong>Analizador Inteligente de Noticias v2.0</strong></p>
    <p style='font-size: 0.9rem;'>🤖 Llama 3.1 70B Versatile | 🏷️ Temas Dinámicos | 🔍 Detección Avanzada de Duplicados</p>
    <p style='font-size: 0.85rem;'>✨ Duplicados NO son analizados - Sentimiento y Tema heredados del original</p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem;'>📊 Análisis inteligente con Machine Learning</p>
</div>
""", unsafe_allow_html=True)
