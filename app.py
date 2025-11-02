import streamlit as st
from groq import Groq
import pandas as pd
import json
import io # MEJORA: Necesario para la descarga en XLSX
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
    page_title="Analizador de Noticias IA | Colombia", 
    page_icon="📰", 
    layout="wide"
)

# MEJORA: Centralizar el nombre del modelo para fácil actualización
MODEL_NAME = "llama-3.1-70b-versatile"

# --- AUTENTICACIÓN ---
if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

def validate_password():
    # SUGERENCIA: Es más seguro depender únicamente de st.secrets y no tener un valor por defecto
    if st.session_state.get("password") == st.secrets.get("PASSWORD"):
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
        <p style='color: #666; margin-bottom: 2rem;'>Análisis de sentimiento y temas dinámicos con IA para el contexto colombiano.</p>
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
    st.error("❌ Configure GROQ_API_KEY en los Secrets de Streamlit.")
    st.stop()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# --- FUNCIONES DE DETECCIÓN DE DUPLICADOS ---
def calcular_similitud_titulo(titulo1, titulo2):
    if pd.isna(titulo1) or pd.isna(titulo2): return 0.0
    return SequenceMatcher(None, str(titulo1).lower().strip(), str(titulo2).lower().strip()).ratio()

def detectar_duplicados_exactos(df):
    columnas_requeridas = ['Empresa', 'Titulo', 'Medio']
    if not all(col in df.columns for col in columnas_requeridas):
        st.warning(f"⚠️ Columnas {columnas_requeridas} no encontradas. No se puede detectar duplicados exactos.")
        return df
    
    df['Es_Duplicado_Exacto'] = df.duplicated(subset=columnas_requeridas, keep=False)
    
    # Asignar ID de grupo para fácil filtrado
    df['Grupo_Duplicado_Exacto'] = -1
    if df['Es_Duplicado_Exacto'].any():
        df.loc[df['Es_Duplicado_Exacto'], 'Grupo_Duplicado_Exacto'] = df[df['Es_Duplicado_Exacto']].groupby(columnas_requeridas).ngroup()
    return df

def detectar_duplicados_similares(df, umbral_similitud=0.85):
    columnas_requeridas = ['Empresa', 'Titulo', 'Medio']
    if not all(col in df.columns for col in columnas_requeridas):
        st.warning(f"⚠️ Columnas {columnas_requeridas} no encontradas. No se puede detectar duplicados similares.")
        return df
    
    df['Es_Duplicado_Similar'] = False
    df['Grupo_Duplicado_Similar'] = -1
    
    # Ignorar los que ya son duplicados exactos
    df_a_revisar = df[~df['Es_Duplicado_Exacto']].copy()
    
    indices = df_a_revisar.index.tolist()
    procesados = set()
    grupo_id_similar = 0
    
    for i in range(len(indices)):
        idx1 = indices[i]
        if idx1 in procesados: continue
        
        grupo_actual = [idx1]
        for j in range(i + 1, len(indices)):
            idx2 = indices[j]
            if idx2 in procesados: continue
            
            # Comparar si son de la misma empresa y medio
            if (df.loc[idx1, 'Empresa'] == df.loc[idx2, 'Empresa'] and 
                df.loc[idx1, 'Medio'] == df.loc[idx2, 'Medio']):
                
                similitud = calcular_similitud_titulo(df.loc[idx1, 'Titulo'], df.loc[idx2, 'Titulo'])
                if similitud >= umbral_similitud:
                    grupo_actual.append(idx2)
        
        if len(grupo_actual) > 1:
            for idx in grupo_actual:
                df.loc[idx, 'Es_Duplicado_Similar'] = True
                df.loc[idx, 'Grupo_Duplicado_Similar'] = grupo_id_similar
                procesados.add(idx)
            grupo_id_similar += 1
            
    return df

# --- FUNCIONES DE ANÁLISIS CON IA ---
# MEJORA: Añadir cache para optimizar. Si el input (textos, num_temas) es el mismo, no vuelve a llamar a la API.
@st.cache_data
def descubrir_temas_dinamicos(_textos, num_temas=20):
    try:
        muestra_size = min(100, len(_textos))
        muestra_textos = [str(t)[:400] for t in np.random.choice(_textos, muestra_size, replace=False)]
        textos_muestra = "\n\n".join([f"{i+1}. {texto}" for i, texto in enumerate(muestra_textos)])
        
        # MEJORA: Prompt enfocado en Colombia
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"""Eres un analista de medios experto en el **contexto colombiano**. Tu tarea es analizar un conjunto de noticias y descubrir los {num_temas} temas principales que representan el contenido.

INSTRUCCIONES:
1. Identifica los {num_temas} temas más relevantes.
2. Los temas deben ser específicos y descriptivos del contenido real, evitando generalidades como "Noticias".
3. Considera temas de relevancia nacional para **Colombia** (política, economía, social, orden público, etc.).
4. Usa nombres cortos y claros (máximo 4 palabras).
5. Los temas deben ser mutuamente excluyentes.

FORMATO DE SALIDA (JSON):
{{
  "temas": [
    {{"id": 1, "nombre": "Nombre del Tema", "descripcion": "Breve descripción del tema en el contexto colombiano."}},
    ...
  ]
}}"""},
                {"role": "user", "content": f"Analiza estas {muestra_size} noticias de Colombia y descubre los {num_temas} temas principales:\n\n{textos_muestra}"}
            ],
            model=MODEL_NAME, temperature=0.2, max_tokens=2500, response_format={"type": "json_object"}
        )
        
        resultado = json.loads(chat_completion.choices[0].message.content)
        return [tema.get('nombre') for tema in resultado.get('temas', []) if tema.get('nombre')]
        
    except Exception as e:
        st.error(f"Error descubriendo temas: {e}")
        return [f"Tema Genérico {i+1}" for i in range(num_temas)]

def analizar_sentimiento_batch(textos, cliente_foco=None, batch_size=15):
    resultados = []
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i+batch_size]
        try:
            prompt_cliente = f"\n- IMPORTANTE: Analiza el sentimiento específicamente desde la perspectiva de '{cliente_foco}', evaluando cómo le afecta a esta empresa/entidad en el **contexto colombiano**." if cliente_foco else ""
            textos_numerados = "\n".join([f"{j+1}. {str(texto)[:400]}" for j, texto in enumerate(batch)])
            
            # MEJORA: Prompt de sentimiento enfocado en Colombia
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"""Eres un analista de medios experto, con especialización en la **opinión pública de Colombia**. Analiza el sentimiento de cada noticia.{prompt_cliente}

ESCALA DE SENTIMIENTO (CONTEXTO COLOMBIA):
- Muy Positivo (+2): Logros excepcionales, reconocimientos importantes, beneficios claros para el país o la entidad. (Ej: Avances en el proceso de paz, importantes descubrimientos de Ecopetrol).
- Positivo (+1): Noticias favorables, desarrollos constructivos, buena gestión.
- Neutral (0): Informativo, sin carga emocional clara, reportes fácticos.
- Negativo (-1): Críticas, problemas operativos, situaciones desfavorables.
- Muy Negativo (-2): Crisis reputacional, escándalos de corrupción, graves accidentes, impacto nacional negativo.

FORMATO DE SALIDA (JSON):
{{"analisis": [
  {{"id": 1, "sentimiento": "Positivo", "score": 1, "razon": "Breve explicación basada en el texto."}},
  ...
]}}"""},
                    {"role": "user", "content": f"Analiza el sentimiento de estas noticias de Colombia:\n\n{textos_numerados}"}
                ],
                model=MODEL_NAME, temperature=0.1, max_tokens=4000, response_format={"type": "json_object"}
            )
            
            resultado = json.loads(chat_completion.choices[0].message.content)
            resultados.extend(resultado.get('analisis', []))
            
        except Exception as e:
            st.warning(f"Error en batch de sentimiento {i//batch_size + 1}: {e}")
            resultados.extend([{"sentimiento": "Neutral", "score": 0, "razon": "Error en análisis"}] * len(batch))
    
    return resultados

def clasificar_temas_batch(textos, temas_disponibles, batch_size=15):
    resultados = []
    temas_str = "\n".join([f"- {tema}" for tema in temas_disponibles])
    
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i+batch_size]
        try:
            textos_numerados = "\n".join([f"{j+1}. {str(texto)[:400]}" for j, texto in enumerate(batch)])
            
            # MEJORA: Prompt de clasificación con contexto
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"""Eres un clasificador de noticias experto en el **contexto de medios colombianos**. Clasifica cada noticia en UNO de los siguientes temas:

{temas_str}

REGLAS:
- Asigna el tema MÁS RELEVANTE.
- Si una noticia trata sobre varios temas, elige el principal.
- Usa exactamente los nombres de los temas proporcionados.

FORMATO DE SALIDA (JSON):
{{"clasificacion": [
  {{"id": 1, "tema": "Nombre exacto del tema", "confianza": 0.95}},
  ...
]}}"""},
                    {"role": "user", "content": f"Clasifica estas noticias de Colombia:\n\n{textos_numerados}"}
                ],
                model=MODEL_NAME, temperature=0.1, max_tokens=3000, response_format={"type": "json_object"}
            )
            
            resultado = json.loads(chat_completion.choices[0].message.content)
            resultados.extend(resultado.get('clasificacion', []))
            
        except Exception as e:
            st.warning(f"Error en batch de clasificación {i//batch_size + 1}: {e}")
            resultados.extend([{"tema": temas_disponibles[0] if temas_disponibles else "Sin clasificar", "confianza": 0}] * len(batch))
    
    return resultados

def generar_insights_estrategicos(df, cliente_foco=None):
    try:
        prompt_cliente = f"\n\nCLIENTE FOCO: '{cliente_foco}' - Genera insights específicos sobre cómo estas noticias afectan a este cliente/empresa en el **contexto colombiano**." if cliente_foco else ""
        
        # MEJORA: Prompt de insights enfocado en Colombia
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"""Eres un consultor estratégico especializado en el **mercado y la política colombiana**. Genera insights accionables y de alto valor basados en el resumen de datos de noticias proporcionado.

FORMATO DE SALIDA (JSON):
{{
  "resumen_ejecutivo": "Análisis conciso (2-3 líneas) de la situación mediática general, considerando el panorama actual de Colombia.",
  "tendencias_clave": ["Análisis de la tendencia principal (Ej: 'Creciente preocupación por la seguridad en Bogotá').", "Análisis de otra tendencia relevante."],
  "oportunidades": ["Oportunidad de comunicación o negocio basada en las noticias positivas o neutras.", "Otra oportunidad."],
  "riesgos": ["Riesgo reputacional o de negocio identificado en las noticias negativas.", "Otro riesgo."],
  "recomendaciones": ["Acción concreta y estratégica (Ej: 'Lanzar campaña aclarando X punto').", "Otra recomendación."],
  "hallazgos_duplicados": "Análisis sobre el nivel de republicación de noticias (duplicados), y lo que esto implica (Ej: 'Alta viralidad de un comunicado' o 'Campaña de desprestigio coordinada')."
}}"""},
                {"role": "user", "content": f"""Analiza estos datos de noticias de Colombia:

Total de noticias: {len(df)}
Distribución de sentimiento: {df['Sentimiento'].value_counts().to_dict() if 'Sentimiento' in df.columns else {}}
Temas principales: {df['Tema'].value_counts().head(5).to_dict() if 'Tema' in df.columns else {}}
Duplicados exactos: {df.get('Es_Duplicado_Exacto', pd.Series(False)).sum()}
Duplicados similares: {df.get('Es_Duplicado_Similar', pd.Series(False)).sum()}
{prompt_cliente}"""}
            ],
            model=MODEL_NAME, temperature=0.4, max_tokens=2500, response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generando insights: {e}")
        return {}

def chat_con_datos(pregunta, df, historial):
    try:
        contexto_datos = f"""
        - Total de noticias: {len(df)}
        - Columnas disponibles: {', '.join(df.columns)}
        - Temas principales: {df['Tema'].value_counts().head(3).to_dict() if 'Tema' in df.columns else 'N/A'}
        - Sentimientos: {df['Sentimiento'].value_counts().to_dict() if 'Sentimiento' in df.columns else 'N/A'}
        """
        mensajes = [
            {"role": "system", "content": f"""Eres un analista de datos experto en el **contexto de noticias de Colombia**. Responde preguntas sobre el dataset de noticias de forma concisa y directa. Basa tus respuestas únicamente en los datos proporcionados.
            Contexto: {contexto_datos}"""}
        ]
        for item in historial[-4:]: # Historial corto
            mensajes.extend([{"role": "user", "content": item["pregunta"]}, {"role": "assistant", "content": item["respuesta"]}])
        mensajes.append({"role": "user", "content": pregunta})
        
        chat_completion = client.chat.completions.create(
            messages=mensajes, model=MODEL_NAME, temperature=0.2, max_tokens=1500
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error al procesar la pregunta: {e}"

# --- FUNCIÓN PARA DESCARGA XLSX (MEJORA) ---
def to_excel(df, insights):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Hoja 1: Dashboard / Resumen
        df_resumen = pd.DataFrame({
            'Métrica': [
                'Total Noticias', 'Noticias Únicas', 'Duplicados Exactos', 
                'Duplicados Similares', 'Temas Descubiertos'
            ],
            'Valor': [
                len(df),
                len(df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)]),
                df.get('Es_Duplicado_Exacto', pd.Series(False)).sum(),
                df.get('Es_Duplicado_Similar', pd.Series(False)).sum(),
                df['Tema'].nunique() if 'Tema' in df.columns else 0
            ]
        })
        df_resumen.to_excel(writer, sheet_name='Dashboard', index=False, startrow=1)
        
        # Añadir insights al dashboard
        if insights:
            ws = writer.sheets['Dashboard']
            start_row = len(df_resumen) + 4
            ws.cell(row=start_row, column=1, value="Resumen Ejecutivo").font = openpyxl.styles.Font(bold=True)
            ws.cell(row=start_row + 1, column=1, value=insights.get('resumen_ejecutivo', ''))
            # ... se pueden añadir más insights aquí

        # Hoja 2: Datos Completos
        df.to_excel(writer, index=False, sheet_name='Datos Completos')
        
        # Hoja 3: Noticias Únicas
        df_unicos = df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)]
        df_unicos.to_excel(writer, index=False, sheet_name='Noticias Únicas')

        # Hoja 4: Grupos de Duplicados
        if 'Grupo_Duplicado_Exacto' in df.columns and 'Grupo_Duplicado_Similar' in df.columns:
            df_duplicados = df[df['Grupo_Duplicado_Exacto'] >= 0 | df['Grupo_Duplicado_Similar'] >= 0]
            df_duplicados.sort_values(by=['Grupo_Duplicado_Exacto', 'Grupo_Duplicado_Similar']).to_excel(writer, index=False, sheet_name='Grupos Duplicados')

    processed_data = output.getvalue()
    return processed_data

# --- INTERFAZ PRINCIPAL ---
st.title("📰 Analizador Inteligente de Noticias")
st.markdown("*Análisis con IA para el contexto de medios en **Colombia***")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    cliente_foco = st.text_input("🎯 Cliente/Empresa a Analizar", placeholder="Ej: Ecopetrol, MinSalud",
        help="El análisis se enfocará en cómo las noticias afectan a este cliente.")
    
    st.markdown("---")
    st.subheader("🎛️ Opciones de Análisis")
    num_temas = st.slider("📊 Número de temas a descubrir", 5, 30, 15)
    analizar_sentimiento = st.checkbox("💭 Análisis de Sentimiento", value=True)
    clasificar_temas = st.checkbox("🏷️ Clasificación Temática", value=True)
    detectar_duplicados = st.checkbox("🔍 Detectar Duplicados", value=True)
    generar_insights = st.checkbox("🧠 Insights Estratégicos", value=True)
    umbral_similitud = st.slider("🎚️ Umbral similitud duplicados (%)", 70, 95, 85)
    
# --- CARGA DE ARCHIVO ---
st.subheader("📤 Sube tu archivo Excel")

# MEJORA: Añadir instrucciones claras
with st.expander("Ver instrucciones y formato requerido"):
    st.info("""
    El archivo Excel (.xlsx) debe contener las siguientes columnas **obligatorias**:
    - `Titulo`: El titular de la noticia.
    - `Resumen`: El cuerpo o resumen de la noticia.

    Para una detección de duplicados más precisa, incluya estas columnas **opcionales**:
    - `Empresa`: Nombre de la empresa o entidad principal mencionada.
    - `Medio`: El medio de comunicación que publica la noticia.
    """)

uploaded_file = st.file_uploader("Selecciona un archivo .xlsx", type=['xlsx', 'xls'])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        
        if 'Titulo' not in df.columns or 'Resumen' not in df.columns:
            st.error("❌ El archivo debe contener las columnas 'Titulo' y 'Resumen'.")
            st.stop()
        
        st.success(f"✅ Archivo cargado: {len(df)} noticias encontradas.")
        
        if st.button("🚀 Iniciar Análisis Inteligente", type="primary", use_container_width=True):
            st.session_state.analysis_done = False
            st.session_state.chat_history = []
            
            progress_bar = st.progress(0, text="Iniciando análisis...")
            
            # Preparar textos
            df['Texto_Completo'] = df['Titulo'].fillna('') + '. ' + df['Resumen'].fillna('')
            textos = df['Texto_Completo'].tolist()
            
            if clasificar_temas:
                progress_bar.progress(10, text="Descubriendo temas...")
                temas_descubiertos = descubrir_temas_dinamicos(tuple(textos), num_temas) # tuple para cache
                st.session_state.temas_descubiertos = temas_descubiertos
                with st.expander("📋 Ver temas descubiertos por la IA"):
                    st.write(temas_descubiertos)

            if detectar_duplicados:
                progress_bar.progress(25, text="Detectando duplicados...")
                df = detectar_duplicados_exactos(df)
                df = detectar_duplicados_similares(df, umbral_similitud/100)

            # Optimización: Solo analizar noticias únicas
            mask_no_duplicados = ~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)
            df_analizar = df[mask_no_duplicados]
            textos_analizar = df_analizar['Texto_Completo'].tolist()

            if analizar_sentimiento and len(textos_analizar) > 0:
                progress_bar.progress(40, text=f"Analizando sentimiento de {len(textos_analizar)} noticias únicas...")
                resultados_sentimiento = analizar_sentimiento_batch(textos_analizar, cliente_foco)
                df.loc[mask_no_duplicados, 'Sentimiento'] = [r.get('sentimiento', 'Neutral') for r in resultados_sentimiento]
                df.loc[mask_no_duplicados, 'Score_Sentimiento'] = [r.get('score', 0) for r in resultados_sentimiento]
                df.loc[mask_no_duplicados, 'Razon_Sentimiento'] = [r.get('razon', '') for r in resultados_sentimiento]

            if clasificar_temas and 'temas_descubiertos' in st.session_state and len(textos_analizar) > 0:
                progress_bar.progress(65, text=f"Clasificando temas de {len(textos_analizar)} noticias únicas...")
                resultados_temas = clasificar_temas_batch(textos_analizar, st.session_state.temas_descubiertos)
                df.loc[mask_no_duplicados, 'Tema'] = [r.get('tema', 'Sin clasificar') for r in resultados_temas]
                df.loc[mask_no_duplicados, 'Confianza_Tema'] = [r.get('confianza', 0) for r in resultados_temas]

            # Propagar resultados a duplicados
            progress_bar.progress(80, text="Propagando análisis a duplicados...")
            for grupo_col, tipo in [('Grupo_Duplicado_Exacto', 'Exacto'), ('Grupo_Duplicado_Similar', 'Similar')]:
                if grupo_col in df.columns:
                    for grupo_id in df[df[grupo_col] >= 0][grupo_col].unique():
                        grupo_mask = df[grupo_col] == grupo_id
                        original = df[grupo_mask & mask_no_duplicados]
                        if not original.empty:
                            idx_orig = original.index[0]
                            if analizar_sentimiento:
                                df.loc[grupo_mask, 'Sentimiento'] = df.loc[idx_orig, 'Sentimiento']
                                df.loc[grupo_mask, 'Score_Sentimiento'] = df.loc[idx_orig, 'Score_Sentimiento']
                                df.loc[grupo_mask, 'Razon_Sentimiento'] = f"[Duplicado {tipo}] " + str(df.loc[idx_orig, 'Razon_Sentimiento'])
                            if clasificar_temas:
                                df.loc[grupo_mask, 'Tema'] = df.loc[idx_orig, 'Tema']
                                df.loc[grupo_mask, 'Confianza_Tema'] = df.loc[idx_orig, 'Confianza_Tema']

            if generar_insights:
                progress_bar.progress(90, text="Generando insights estratégicos...")
                st.session_state.insights = generar_insights_estrategicos(df, cliente_foco)
            
            progress_bar.progress(100, text="¡Análisis completado!")
            st.session_state.df_analizado = df
            st.session_state.analysis_done = True
            st.balloons()
            
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

# --- MOSTRAR RESULTADOS ---
if st.session_state.get('analysis_done', False):
    df = st.session_state.df_analizado
    insights = st.session_state.get('insights', {})
    
    st.markdown("---")
    
    # MEJORA: Botón de descarga XLSX prominente
    st.download_button(
        label="📥 Descargar Resultados en Excel (.xlsx)",
        data=to_excel(df, insights),
        file_name=f"Analisis_Noticias_{cliente_foco or 'General'}_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary"
    )

    tabs = st.tabs(["📊 Dashboard", "🗂️ Datos Analizados", "🔍 Duplicados", "🧠 Insights IA", "💬 Chat con Datos"])
    
    # El resto del código para mostrar los tabs es excelente y no necesita cambios mayores.
    # Se han mantenido tus visualizaciones y estructura de tabs.
    with tabs[0]:
        st.subheader("📊 Dashboard de Análisis")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        noticias_unicas = len(df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)])
        
        col1.metric("📰 Total Noticias", len(df))
        col2.metric("✅ Noticias Únicas", noticias_unicas)
        if 'Es_Duplicado_Exacto' in df.columns:
            col3.metric("🔴 Duplicados Exactos", df['Es_Duplicado_Exacto'].sum())
        if 'Es_Duplicado_Similar' in df.columns:
            col4.metric("🟡 Duplicados Similares", df['Es_Duplicado_Similar'].sum())

        st.markdown("---")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            if 'Sentimiento' in df.columns:
                st.markdown("#### 💭 Distribución de Sentimientos (Noticias Únicas)")
                df_unicos = df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)]
                fig_sentiment = px.pie(df_unicos, names='Sentimiento', color='Sentimiento',
                    color_discrete_map={
                        'Muy Positivo': '#2ECC71', 'Positivo': '#85C1E2', 'Neutral': '#95A5A6',
                        'Negativo': '#F39C12', 'Muy Negativo': '#E74C3C'
                    })
                st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col_g2:
            if 'Tema' in df.columns:
                st.markdown("#### 🏷️ Top Temas (Noticias Únicas)")
                df_unicos = df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)]
                temas_count = df_unicos['Tema'].value_counts().nlargest(10)
                fig_temas = px.bar(temas_count, y=temas_count.index, x=temas_count.values, orientation='h', labels={'y': 'Tema', 'x': 'Cantidad'})
                fig_temas.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_temas, use_container_width=True)

    with tabs[1]:
        st.subheader("🗂️ Datos Analizados")
        st.dataframe(df, use_container_width=True)
        # Aquí se podrían añadir los filtros que ya tenías, son una buena adición.

    with tabs[2]:
        st.subheader("🔍 Análisis Detallado de Duplicados")
        # El código para mostrar los grupos de duplicados es muy bueno y se mantiene.
        tipo_ver = st.radio("Ver duplicados:", ['Exactos', 'Similares'], horizontal=True)
        if tipo_ver == 'Exactos' and 'Grupo_Duplicado_Exacto' in df.columns and df['Grupo_Duplicado_Exacto'].max() >= 0:
            for grupo_id, grupo_df in df[df['Grupo_Duplicado_Exacto'] >= 0].groupby('Grupo_Duplicado_Exacto'):
                with st.expander(f"Grupo Exacto {int(grupo_id)+1} ({len(grupo_df)} menciones) - Título: {grupo_df.iloc[0]['Titulo']}"):
                    st.dataframe(grupo_df)
        elif tipo_ver == 'Similares' and 'Grupo_Duplicado_Similar' in df.columns and df['Grupo_Duplicado_Similar'].max() >= 0:
             for grupo_id, grupo_df in df[df['Grupo_Duplicado_Similar'] >= 0].groupby('Grupo_Duplicado_Similar'):
                with st.expander(f"Grupo Similar {int(grupo_id)+1} ({len(grupo_df)} menciones) - Empresa: {grupo_df.iloc[0]['Empresa']}"):
                    st.dataframe(grupo_df[['Titulo', 'Medio', 'Resumen']])

    with tabs[3]:
        st.subheader("🧠 Insights Estratégicos por IA")
        if insights:
            st.info(f"**Resumen Ejecutivo:** {insights.get('resumen_ejecutivo', 'N/A')}")
            st.warning(f"**Análisis de Duplicados:** {insights.get('hallazgos_duplicados', 'N/A')}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ✅ Oportunidades")
                for o in insights.get('oportunidades', []): st.success(f"- {o}")
                st.markdown("#### 📈 Tendencias Clave")
                for t in insights.get('tendencias_clave', []): st.markdown(f"- {t}")
            with col2:
                st.markdown("#### ⚠️ Riesgos")
                for r in insights.get('riesgos', []): st.warning(f"- {r}")
                st.markdown("#### 💡 Recomendaciones")
                for rec in insights.get('recomendaciones', []): st.markdown(f"- {rec}")
        else:
            st.info("No se generaron insights. Activa la opción en el sidebar y vuelve a analizar.")
    
    with tabs[4]:
        st.subheader("💬 Chat Inteligente con tus Datos")
        # Tu código de chat es excelente y se mantiene.
        for item in st.session_state.chat_history:
            with st.chat_message("user"): st.markdown(item["pregunta"])
            with st.chat_message("assistant"): st.markdown(item["respuesta"])
        
        if prompt := st.chat_input("Pregunta sobre los datos analizados..."):
            st.session_state.chat_history.append({"pregunta": prompt, "respuesta": "..."})
            st.rerun()

        if st.session_state.chat_history and st.session_state.chat_history[-1]["respuesta"] == "...":
            pregunta = st.session_state.chat_history[-1]["pregunta"]
            with st.chat_message("user"): st.markdown(pregunta)
            with st.chat_message("assistant"):
                respuesta = chat_con_datos(pregunta, df, st.session_state.chat_history[:-1])
                st.markdown(respuesta)
                st.session_state.chat_history[-1]["respuesta"] = respuesta

# --- PIE DE PÁGINA ---
st.markdown("---")
if st.button("🗑️ Reiniciar Análisis Completo"):
    # Conservar estado de login
    password_status = st.session_state.password_correct
    st.session_state.clear()
    st.session_state.password_correct = password_status
    st.rerun()

st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p><strong>Analizador Inteligente de Noticias v2.1 (Enfoque Colombia)</strong></p>
    <p style='font-size: 0.9rem;'>🤖 Potenciado por Llama 3.1 70B Versatile | 🏷️ Temas Dinámicos | 🔍 Detección Avanzada de Duplicados</p>
</div>
""", unsafe_allow_html=True)
