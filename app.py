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
    return SequenceMatcher(None, str(titulo1).lower().strip(), str(titulo2).lower().strip()).ratio()

def detectar_duplicados_exactos(df):
    """
    Detecta menciones duplicadas:
    - Empresa igual, Titulo igual y Medio igual
    """
    columnas_requeridas = ['Empresa', 'Titulo', 'Medio']
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    
    if columnas_faltantes:
        st.info(f"ℹ️ Detección de duplicados exactos omitida. Columnas faltantes: {', '.join(columnas_faltantes)}")
        df['Es_Duplicado_Exacto'] = False
        df['Grupo_Duplicado_Exacto'] = -1
        return df
    
    df['Es_Duplicado_Exacto'] = False
    df['Grupo_Duplicado_Exacto'] = -1
    
    # Crear clave única
    df['_clave_duplicado'] = df['Empresa'].fillna('').astype(str).str.lower().str.strip() + '|||' + \
                             df['Titulo'].fillna('').astype(str).str.lower().str.strip() + '|||' + \
                             df['Medio'].fillna('').astype(str).str.lower().str.strip()
    
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
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    
    if columnas_faltantes:
        st.info(f"ℹ️ Detección de duplicados similares omitida. Columnas faltantes: {', '.join(columnas_faltantes)}")
        df['Es_Duplicado_Similar'] = False
        df['Grupo_Duplicado_Similar'] = -1
        return df
    
    df['Es_Duplicado_Similar'] = False
    df['Grupo_Duplicado_Similar'] = -1
    
    # Agrupar por Empresa + Medio
    grupos_empresa_medio = df.groupby([df['Empresa'].fillna('').astype(str).str.lower().str.strip(), 
                                       df['Medio'].fillna('').astype(str).str.lower().str.strip()])
    
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
            model="llama3-70b-8192", # <-- CORREGIDO: Modelo actualizado
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
                model="llama3-70b-8192", # <-- CORREGIDO: Modelo actualizado
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
                model="llama3-70b-8192", # <-- CORREGIDO: Modelo actualizado
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
            model="llama3-70b-8192", # <-- CORREGIDO: Modelo actualizado
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
- Duplicados exactos: {len(df[df.get('Es_Duplicado_Exacto', False)])}
- Duplicados similares: {len(df[df.get('Es_Duplicado_Similar', False)])}

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
            model="llama3-70b-8192", # <-- CORREGIDO: Modelo actualizado
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
    📋 **Proceso de carga:**
    1. Sube tu archivo Excel
    2. Selecciona las columnas a usar
    3. Configura las opciones
    4. Inicia el análisis
    
    **Columnas necesarias:**
    - Título (obligatorio)
    - Resumen/Aclaración (obligatorio)
    - Empresa (opcional - duplicados)
    - Medio (opcional - duplicados)
    """)

# --- CARGA DE ARCHIVO ---
st.subheader("📤 Sube tu archivo Excel")

# Ayuda expandible
with st.expander("❓ ¿Cómo funciona el selector de columnas?", expanded=False):
    st.markdown("""
    ### 🎯 Guía de Uso del Selector de Columnas
    
    **Paso 1: Sube tu archivo**
    - Formatos soportados: `.xlsx`, `.xls`
    - El archivo puede tener cualquier estructura
    
    **Paso 2: Selecciona las columnas**
    
    #### 📝 Columnas Obligatorias:
    - **Título:** Columna con los títulos o encabezados de las noticias
    - **Resumen/Aclaración:** Columna con el contenido, resumen o descripción
    
    #### 🔍 Columnas Opcionales (para duplicados):
    - **Empresa:** Nombre de la organización mencionada
    - **Medio:** Fuente o medio de comunicación
    
    **Paso 3: Vista previa**
    - Verifica que las columnas seleccionadas sean correctas
    - Revisa la vista previa del contenido procesado
    
    **Ejemplos de nombres de columnas reconocidos:**
    - Título: `Titulo`, `Title`, `Encabezado`, `Headline`
    - Resumen: `Resumen`, `Aclaración`, `Summary`, `Descripción`, `Contenido`
    - Empresa: `Empresa`, `Company`, `Organización`, `Entidad`
    - Medio: `Medio`, `Fuente`, `Source`, `Periódico`, `Diario`
    
    💡 **Tip:** La aplicación intentará sugerir automáticamente las columnas correctas basándose en sus nombres.
    """)

uploaded_file = st.file_uploader(
    "Selecciona un archivo XLSX",
    type=['xlsx', 'xls'],
    help="Sube tu archivo Excel - puedes usar cualquier nombre de columna"
)

if uploaded_file:
    try:
        df_original = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Archivo cargado: {len(df_original)} filas encontradas")
        
        # Mostrar todas las columnas disponibles
        with st.expander("👀 Vista previa y selección de columnas", expanded=True):
            st.dataframe(df_original.head(10), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 🎯 Selecciona las columnas a usar")
            
            columnas_disponibles = df_original.columns.tolist()
            
            # Función auxiliar para sugerir columna por nombre
            def sugerir_columna(keywords, columnas):
                """Sugiere una columna basándose en palabras clave"""
                for keyword in keywords:
                    for col in columnas:
                        if keyword.lower() in col.lower():
                            return col
                return columnas[0] if columnas else None
            
            col_sel1, col_sel2 = st.columns(2)
            
            with col_sel1:
                st.markdown("#### 📝 Columnas de Contenido")
                st.caption("🔴 Obligatorias para el análisis")
                
                # Sugerir columna de título
                titulo_sugerido = sugerir_columna(['titulo', 'title', 'encabezado', 'headline'], columnas_disponibles)
                col_titulo = st.selectbox(
                    "Columna de TÍTULO:",
                    columnas_disponibles,
                    index=columnas_disponibles.index(titulo_sugerido) if titulo_sugerido else 0,
                    help="Selecciona la columna que contiene los títulos de las noticias"
                )
                
                # Vista previa del título seleccionado
                if col_titulo:
                    muestra_titulo = df_original[col_titulo].dropna().head(1).values
                    if len(muestra_titulo) > 0:
                        st.caption(f"📄 Ejemplo: *{str(muestra_titulo[0])[:80]}...*")
                
                # Sugerir columna de resumen
                resumen_sugerido = sugerir_columna(['resumen', 'aclaracion', 'aclaración', 'summary', 'descripcion', 'descripción', 'contenido', 'texto'], columnas_disponibles)
                col_resumen = st.selectbox(
                    "Columna de RESUMEN/ACLARACIÓN:",
                    columnas_disponibles,
                    index=columnas_disponibles.index(resumen_sugerido) if resumen_sugerido else 0,
                    help="Selecciona la columna que contiene el resumen o aclaración"
                )
                
                # Vista previa del resumen seleccionado
                if col_resumen:
                    muestra_resumen = df_original[col_resumen].dropna().head(1).values
                    if len(muestra_resumen) > 0:
                        st.caption(f"📄 Ejemplo: *{str(muestra_resumen[0])[:80]}...*")
                
                if col_titulo == col_resumen:
                    st.error("⚠️ Las columnas de Título y Resumen deben ser diferentes")
            
            with col_sel2:
                st.markdown("#### 🔍 Columnas para Duplicados")
                st.caption("🟢 Opcionales - mejoran la detección")
                
                # Sugerir columna de empresa
                empresa_sugerido = sugerir_columna(['empresa', 'company', 'organizacion', 'organización', 'entidad'], columnas_disponibles)
                col_empresa = st.selectbox(
                    "Columna de EMPRESA:",
                    ['No usar'] + columnas_disponibles,
                    index=columnas_disponibles.index(empresa_sugerido) + 1 if empresa_sugerido else 0,
                    help="Selecciona la columna que contiene el nombre de la empresa mencionada"
                )
                
                # Vista previa de empresa
                if col_empresa != 'No usar':
                    muestra_empresa = df_original[col_empresa].dropna().head(1).values
                    if len(muestra_empresa) > 0:
                        st.caption(f"🏢 Ejemplo: *{str(muestra_empresa[0])}*")
                
                # Sugerir columna de medio
                medio_sugerido = sugerir_columna(['medio', 'fuente', 'source', 'periodico', 'periódico', 'diario'], columnas_disponibles)
                col_medio = st.selectbox(
                    "Columna de MEDIO:",
                    ['No usar'] + columnas_disponibles,
                    index=columnas_disponibles.index(medio_sugerido) + 1 if medio_sugerido else 0,
                    help="Selecciona la columna que contiene el medio de comunicación"
                )
                
                # Vista previa de medio
                if col_medio != 'No usar':
                    muestra_medio = df_original[col_medio].dropna().head(1).values
                    if len(muestra_medio) > 0:
                        st.caption(f"📡 Ejemplo: *{str(muestra_medio[0])}*")
            
            # Validar selección
            if col_titulo == col_resumen:
                st.error("❌ Debes seleccionar columnas diferentes para Título y Resumen")
                st.stop()
            
            # Crear DataFrame de trabajo con columnas renombradas
            df = pd.DataFrame() # Crear un DF limpio
            df['Titulo'] = df_original[col_titulo]
            df['Resumen'] = df_original[col_resumen]
            
            if col_empresa != 'No usar':
                df['Empresa'] = df_original[col_empresa]
            
            if col_medio != 'No usar':
                df['Medio'] = df_original[col_medio]
            
            # Mostrar resumen de selección
            st.markdown("---")
            st.markdown("#### ✅ Configuración de Columnas")
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            
            with col_info1:
                st.info(f"📰 **Título:**\n`{col_titulo}`")
            with col_info2:
                st.info(f"📝 **Resumen:**\n`{col_resumen}`")
            with col_info3:
                if 'Empresa' in df.columns:
                    st.success(f"🏢 **Empresa:**\n`{col_empresa}`")
                else:
                    st.warning("🏢 **Empresa:**\nNo configurada")
            with col_info4:
                if 'Medio' in df.columns:
                    st.success(f"📡 **Medio:**\n`{col_medio}`")
                else:
                    st.warning("📡 **Medio:**\nNo configurado")
            
            # Vista previa de contenido procesado
            st.markdown("---")
            st.markdown("#### 📖 Vista Previa del Contenido Procesado")
            
            preview_df = pd.DataFrame({
                'Título': df['Titulo'].head(3),
                'Resumen': df['Resumen'].head(3).apply(lambda x: str(x)[:150] + '...' if pd.notna(x) and len(str(x)) > 150 else str(x))
            })
            
            if 'Empresa' in df.columns:
                preview_df['Empresa'] = df['Empresa'].head(3)
            if 'Medio' in df.columns:
                preview_df['Medio'] = df['Medio'].head(3)
            
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
        
        # Validar columnas para duplicados
        if detectar_duplicados:
            columnas_faltantes = []
            if 'Empresa' not in df.columns:
                columnas_faltantes.append('Empresa')
            if 'Medio' not in df.columns:
                columnas_faltantes.append('Medio')
            
            if columnas_faltantes:
                st.warning(f"⚠️ **Detección de duplicados limitada:** Para detección completa se recomiendan las columnas {', '.join(columnas_faltantes)}")
            else:
                st.success("✅ Todas las columnas necesarias para detección de duplicados están configuradas")
        
        # --- BOTÓN DE ANÁLISIS ---
        if st.button("🚀 Iniciar Análisis Inteligente", type="primary", use_container_width=True):
            st.session_state.analysis_done = False
            st.session_state.chat_history = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Preparar textos para análisis
            # <-- CORREGIDO: Convertido a string para evitar errores de tipo al concatenar
            df['Texto_Completo'] = df['Titulo'].fillna('').astype(str) + '. ' + df['Resumen'].fillna('').astype(str)
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
        st.exception(e) # Imprime el traceback completo para depuración

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
            noticias_unicas = len(df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar',
