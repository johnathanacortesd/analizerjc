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
    page_title="Analizador Estratégico de Noticias IA", 
    page_icon="⭐", 
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
        <h1 style='color: #2E86AB; font-size: 3rem;'>⭐</h1>
        <h2>Analizador Estratégico de Noticias</h2>
        <p style='color: #666; margin-bottom: 2rem;'>Insights avanzados cruzando contenido, métricas y tiempo con IA</p>
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
    MODELO_IA = "llama-3.3-70b-versatile" # Modelo centralizado y actualizado
except KeyError:
    st.error("❌ Configure GROQ_API_KEY en Secrets")
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
    if any(col not in df.columns for col in columnas_requeridas):
        st.info("ℹ️ Detección de duplicados exactos omitida. Faltan columnas: Empresa, Titulo o Medio.")
        df['Es_Duplicado_Exacto'], df['Grupo_Duplicado_Exacto'] = False, -1
        return df
    df['_clave'] = df.apply(lambda row: f"{row['Empresa']}_{row['Titulo']}_{row['Medio']}".lower().strip(), axis=1)
    duplicados = df['_clave'].duplicated(keep=False)
    df['Es_Duplicado_Exacto'] = duplicados
    df['Grupo_Duplicado_Exacto'] = df.groupby('_clave').ngroup()
    df.loc[~duplicados, 'Grupo_Duplicado_Exacto'] = -1
    df.drop(columns=['_clave'], inplace=True)
    return df

def detectar_duplicados_similares(df, umbral_similitud=0.85):
    columnas_requeridas = ['Empresa', 'Titulo', 'Medio']
    if any(col not in df.columns for col in columnas_requeridas):
        st.info("ℹ️ Detección de duplicados similares omitida. Faltan columnas: Empresa, Titulo o Medio.")
        df['Es_Duplicado_Similar'], df['Grupo_Duplicado_Similar'] = False, -1
        return df
    df['Es_Duplicado_Similar'], df['Grupo_Duplicado_Similar'] = False, -1
    grupos = df.groupby(['Empresa', 'Medio'])
    grupo_id_similar = 0
    indices_procesados = set()
    for _, grupo in grupos:
        if len(grupo) < 2: continue
        indices = grupo.index.tolist()
        for i in range(len(indices)):
            if indices[i] in indices_procesados: continue
            grupo_actual = [indices[i]]
            for j in range(i + 1, len(indices)):
                if indices[j] in indices_procesados: continue
                sim = calcular_similitud_titulo(df.loc[indices[i], 'Titulo'], df.loc[indices[j], 'Titulo'])
                if sim >= umbral_similitud:
                    grupo_actual.append(indices[j])
            if len(grupo_actual) > 1:
                df.loc[grupo_actual, 'Es_Duplicado_Similar'] = True
                df.loc[grupo_actual, 'Grupo_Duplicado_Similar'] = grupo_id_similar
                indices_procesados.update(grupo_actual)
                grupo_id_similar += 1
    return df

# --- FUNCIONES DE ANÁLISIS CON IA ---
def descubrir_temas_dinamicos(textos, num_temas=20):
    try:
        muestra_size = min(100, len(textos))
        muestra_indices = np.random.choice(len(textos), muestra_size, replace=False)
        muestra_textos = [str(textos[i])[:400] for i in muestra_indices]
        textos_muestra = "\n\n".join([f"{i+1}. {t}" for i, t in enumerate(muestra_textos[:30])])
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"""Eres un analista experto en clasificación de contenido. Analiza las siguientes noticias y extrae los {num_temas} temas principales más representativos. Los temas deben ser específicos, descriptivos (2-4 palabras) y mutuamente excluyentes. Evita temas genéricos. Formato de salida JSON: {{"temas": [{{"id": 1, "nombre": "Nombre del Tema"}}]}}"""},
                {"role": "user", "content": f"Analiza estas {muestra_size} noticias y descubre los {num_temas} temas principales:\n\n{textos_muestra}"}
            ], model=MODELO_IA, temperature=0.2, max_tokens=4000, response_format={"type": "json_object"})
        temas = json.loads(chat_completion.choices[0].message.content).get('temas', [])
        return [tema.get('nombre', f'Tema {i+1}') for i, tema in enumerate(temas)][:num_temas]
    except Exception as e:
        st.error(f"Error descubriendo temas: {e}")
        return [f"Tema {i+1}" for i in range(num_temas)]

def analizar_sentimiento_batch(textos, cliente_foco=None, batch_size=15):
    resultados = []
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i+batch_size]
        try:
            prompt_cliente = f"Analiza el sentimiento específicamente desde la perspectiva de '{cliente_foco}'." if cliente_foco else ""
            textos_numerados = "\n".join([f"{j+1}. {str(texto)[:350]}" for j, texto in enumerate(batch)])
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"""Eres un analista de medios experto. Analiza el sentimiento de cada noticia. {prompt_cliente} Escala: Muy Positivo (2), Positivo (1), Neutral (0), Negativo (-1), Muy Negativo (-2). Formato de salida JSON: {{"analisis": [{{"id": 1, "sentimiento": "Positivo", "score": 1, "razon": "Breve explicación"}}]}}"""},
                    {"role": "user", "content": f"Analiza el sentimiento de estas noticias:\n\n{textos_numerados}"}
                ], model=MODELO_IA, temperature=0.1, max_tokens=4000, response_format={"type": "json_object"})
            resultado = json.loads(chat_completion.choices[0].message.content)
            resultados.extend(resultado.get('analisis', []))
        except Exception as e:
            st.warning(f"Error en batch de sentimiento {i//batch_size + 1}: {e}")
            resultados.extend([{"sentimiento": "Neutral", "score": 0, "razon": "Error en análisis"} for _ in batch])
    return resultados

def clasificar_temas_batch(textos, temas_disponibles, batch_size=15):
    resultados = []
    temas_str = "\n".join([f"{i+1}. {tema}" for i, tema in enumerate(temas_disponibles)])
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i+batch_size]
        try:
            textos_numerados = "\n".join([f"{j+1}. {str(texto)[:350]}" for j, texto in enumerate(batch)])
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"""Eres un clasificador de noticias experto. Clasifica cada noticia en UNO de los siguientes temas: {temas_str}. Asigna el tema MÁS RELEVANTE. Formato de salida JSON: {{"clasificacion": [{{"id": 1, "tema": "Nombre exacto del tema"}}]}}"""},
                    {"role": "user", "content": f"Clasifica estas noticias:\n\n{textos_numerados}"}
                ], model=MODELO_IA, temperature=0.1, max_tokens=4000, response_format={"type": "json_object"})
            resultado = json.loads(chat_completion.choices[0].message.content)
            resultados.extend(resultado.get('clasificacion', []))
        except Exception as e:
            st.warning(f"Error en batch de clasificación {i//batch_size + 1}: {e}")
            resultados.extend([{"tema": "Sin clasificar"} for _ in batch])
    return resultados

def generar_insights_estrategicos(df, cliente_foco=None):
    try:
        contexto_datos = f"Total de noticias analizadas: {len(df)}\n"
        if cliente_foco: contexto_datos += f"Cliente Foco del Análisis: '{cliente_foco}'\n"

        if 'Sentimiento' in df.columns: contexto_datos += f"Distribución de Sentimiento: {df['Sentimiento'].value_counts().to_dict()}\n"
        if 'Tema' in df.columns: contexto_datos += f"Top 5 Temas: {df['Tema'].value_counts().head(5).to_dict()}\n"
        if 'Tier' in df.columns: contexto_datos += f"Distribución por Tier: {df['Tier'].value_counts().to_dict()}\n"
        if 'Tipo de Medio' in df.columns: contexto_datos += f"Distribución por Tipo de Medio: {df['Tipo de Medio'].value_counts().to_dict()}\n"
        
        if 'Audiencia' in df.columns and pd.to_numeric(df['Audiencia'], errors='coerce').notna().any():
            df['Audiencia'] = pd.to_numeric(df['Audiencia'], errors='coerce')
            contexto_datos += f"Audiencia Total Acumulada: {df['Audiencia'].sum():,.0f}\n"
            contexto_datos += f"Audiencia Promedio por Noticia: {df['Audiencia'].mean():,.0f}\n"

        if 'Fecha' in df.columns:
            try:
                df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
                contexto_datos += f"Rango de Fechas: {df['Fecha'].min().strftime('%Y-%m-%d')} a {df['Fecha'].max().strftime('%Y-%m-%d')}\n"
                contexto_datos += f"Días con más noticias: {df['Fecha'].dt.day_name().value_counts().head(3).to_dict()}\n"
            except Exception: pass
        
        if 'Hora' in df.columns:
            try:
                df['Hora_obj'] = pd.to_datetime(df['Hora'], errors='coerce', format='%H:%M:%S').dt.hour
                if df['Hora_obj'].isnull().all():
                     df['Hora_obj'] = pd.to_datetime(df['Hora'], errors='coerce').dt.hour
                contexto_datos += f"Horas pico de publicación: {df['Hora_obj'].value_counts().head(3).to_dict()}\n"
            except Exception: pass
            
        columnas_muestra = [col for col in ['Titulo', 'Sentimiento', 'Tema', 'Tier', 'Audiencia', 'Medio'] if col in df.columns]
        muestra_noticias = df[columnas_muestra].head(5).to_string()

        system_prompt = """Eres un estratega de comunicación de élite. Analiza los datos de noticias para extraer insights profundos y accionables. No solo describas los datos; interprétalos, encuentra conexiones ocultas y proporciona recomendaciones estratégicas. Cruza variables como Sentimiento, Tema, Tier, Audiencia, Tipo de Medio y Tiempo para responder preguntas clave sobre impacto y reputación.

FORMATO DE SALIDA (JSON):
{
  "resumen_ejecutivo": "Un párrafo conciso con el hallazgo más importante.",
  "hallazgos_clave": [
    {"hallazgo": "Descripción del insight.", "evidencia": "Datos específicos que lo respaldan (ej. 'El 75% de las noticias negativas provienen de medios Tier 3')."}
  ],
  "oportunidades_estrategicas": ["Oportunidad 1 basada en los datos.", "Oportunidad 2..."],
  "riesgos_emergentes": ["Riesgo 1 identificado en el análisis.", "Riesgo 2..."],
  "recomendaciones_accionables": ["Recomendación concreta (ej. 'Enfocar esfuerzos de relaciones públicas en medios digitales Tier 2').", "Recomendación 2..."]
}"""
        user_prompt = f"Analiza los siguientes datos de monitoreo de medios.\n\nRESUMEN CUANTITATIVO:\n{contexto_datos}\n\nMUESTRA DE NOTICIAS:\n{muestra_noticias}"

        chat_completion = client.chat.completions.create(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], model=MODELO_IA, temperature=0.3, max_tokens=4000, response_format={"type": "json_object"})
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generando insights: {e}")
        return {}

def chat_con_datos(pregunta, df, historial):
    try:
        contexto_datos = f"El dataset contiene {len(df)} filas. Resumen estadístico de cada columna:\n\n"
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].nunique() > 10 and col not in ['Grupo_Duplicado_Exacto', 'Grupo_Duplicado_Similar']:
                contexto_datos += f"- Columna '{col}' (Numérica):\n{df[col].describe().to_string()}\n\n"
            else:
                top_values = df[col].value_counts().head(5).to_dict()
                contexto_datos += f"- Columna '{col}' (Categórica):\n  Valores más comunes: {top_values}\n  Valores únicos: {df[col].nunique()}\n\n"

        system_prompt = f"Eres un analista de datos experto. Tu conocimiento se basa EXCLUSIVAMENTE en el siguiente resumen del dataset. Responde a las preguntas del usuario de forma precisa y basándote en los datos. Si te piden un cálculo, usa las estadísticas disponibles. Si la pregunta no se puede responder con el contexto, indícalo claramente.\n\nCONTEXTO:\n{contexto_datos}"
        
        mensajes = [{"role": "system", "content": system_prompt}]
        
        # Corregir la forma en que se extiende la lista de mensajes
        for item in historial[-5:]:
            mensajes.append({"role": "user", "content": item["pregunta"]})
            mensajes.append({"role": "assistant", "content": item["respuesta"]})
        
        mensajes.append({"role": "user", "content": pregunta})
        
        chat_completion = client.chat.completions.create(messages=mensajes, model=MODELO_IA, temperature=0.2, max_tokens=2000)
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error al procesar la pregunta: {e}"

# --- INTERFAZ PRINCIPAL ---
st.title("⭐ Analizador Estratégico de Noticias")
st.markdown("*Insights avanzados cruzando contenido, métricas y tiempo con IA*")

with st.sidebar:
    st.header("⚙️ Configuración de Análisis")
    cliente_foco = st.text_input("🎯 Cliente/Empresa Foco", placeholder="Ej: Ministerio de Salud, Ecopetrol")
    st.markdown("---")
    num_temas = st.slider("📊 Número de temas a descubrir", 5, 30, 15)
    umbral_similitud = st.slider("🎚️ Umbral similitud duplicados (%)", 70, 95, 85)
    st.markdown("---")
    st.subheader("✅ Módulos a ejecutar")
    analizar_sentimiento = st.checkbox("💭 Análisis de Sentimiento", value=True)
    clasificar_temas = st.checkbox("🏷️ Clasificación Temática", value=True)
    detectar_duplicados = st.checkbox("🔍 Detección de Duplicados", value=True)
    generar_insights = st.checkbox("🧠 Insights Estratégicos IA", value=True)

st.subheader("📤 1. Sube tu archivo Excel")
uploaded_file = st.file_uploader("El archivo puede tener cualquier estructura de columnas.", type=['xlsx', 'xls'], label_visibility="collapsed")

if uploaded_file:
    try:
        df_original = pd.read_excel(uploaded_file)
        st.success(f"✅ Archivo '{uploaded_file.name}' cargado: {len(df_original)} filas.")

        with st.expander("👀 2. Mapea tus columnas y previsualiza", expanded=True):
            st.dataframe(df_original.head(5), use_container_width=True)
            st.markdown("---")
            
            columnas_disponibles = df_original.columns.tolist()
            def sugerir_columna(keywords, cols):
                for kw in keywords:
                    for col in cols:
                        if kw.lower() in col.lower(): return col
                return None

            col_map1, col_map2, col_map3 = st.columns(3)
            with col_map1:
                st.markdown("#### Contenido (Obligatorio)")
                col_titulo = st.selectbox("Columna de TÍTULO:", columnas_disponibles, index=columnas_disponibles.index(sugerir_columna(['titulo', 'title'], columnas_disponibles)) if sugerir_columna(['titulo', 'title'], columnas_disponibles) else 0)
                col_resumen = st.selectbox("Columna de RESUMEN:", columnas_disponibles, index=columnas_disponibles.index(sugerir_columna(['resumen', 'aclaracion'], columnas_disponibles)) if sugerir_columna(['resumen', 'aclaracion'], columnas_disponibles) else 1)
            with col_map2:
                st.markdown("#### Metadatos (Recomendado)")
                col_empresa = st.selectbox("Columna de EMPRESA:", ['No usar'] + columnas_disponibles, index=columnas_disponibles.index(sugerir_columna(['empresa', 'entidad'], columnas_disponibles)) + 1 if sugerir_columna(['empresa', 'entidad'], columnas_disponibles) else 0)
                col_medio = st.selectbox("Columna de MEDIO:", ['No usar'] + columnas_disponibles, index=columnas_disponibles.index(sugerir_columna(['medio', 'fuente'], columnas_disponibles)) + 1 if sugerir_columna(['medio', 'fuente'], columnas_disponibles) else 0)
                col_tipo_medio = st.selectbox("Columna de TIPO MEDIO:", ['No usar'] + columnas_disponibles, index=columnas_disponibles.index(sugerir_columna(['tipo'], columnas_disponibles)) + 1 if sugerir_columna(['tipo'], columnas_disponibles) else 0)
            with col_map3:
                st.markdown("#### Métricas (Opcional)")
                col_tier = st.selectbox("Columna de TIER:", ['No usar'] + columnas_disponibles, index=columnas_disponibles.index(sugerir_columna(['tier'], columnas_disponibles)) + 1 if sugerir_columna(['tier'], columnas_disponibles) else 0)
                col_audiencia = st.selectbox("Columna de AUDIENCIA:", ['No usar'] + columnas_disponibles, index=columnas_disponibles.index(sugerir_columna(['audiencia'], columnas_disponibles)) + 1 if sugerir_columna(['audiencia'], columnas_disponibles) else 0)
                col_cpe = st.selectbox("Columna de CPE:", ['No usar'] + columnas_disponibles, index=columnas_disponibles.index(sugerir_columna(['cpe', 'valor'], columnas_disponibles)) + 1 if sugerir_columna(['cpe', 'valor'], columnas_disponibles) else 0)
                col_fecha = st.selectbox("Columna de FECHA:", ['No usar'] + columnas_disponibles, index=columnas_disponibles.index(sugerir_columna(['fecha'], columnas_disponibles)) + 1 if sugerir_columna(['fecha'], columnas_disponibles) else 0)
                col_hora = st.selectbox("Columna de HORA:", ['No usar'] + columnas_disponibles, index=columnas_disponibles.index(sugerir_columna(['hora'], columnas_disponibles)) + 1 if sugerir_columna(['hora'], columnas_disponibles) else 0)

            df = pd.DataFrame()
            column_mapping = {
                'Titulo': col_titulo, 'Resumen': col_resumen, 'Empresa': col_empresa,
                'Medio': col_medio, 'Tier': col_tier, 'Audiencia': col_audiencia,
                'CPE': col_cpe, 'Fecha': col_fecha, 'Hora': col_hora,
                'Tipo de Medio': col_tipo_medio
            }
            for new_name, old_name in column_mapping.items():
                if old_name and old_name != 'No usar':
                    df[new_name] = df_original[old_name]

            st.success("Columnas mapeadas. ¡Listo para analizar!")
        
        st.subheader("🚀 3. Iniciar Análisis")
        if st.button("✨ Analizar con IA ✨", type="primary", use_container_width=True):
            st.session_state.analysis_done = False
            st.session_state.chat_history = []
            
            with st.spinner("Realizando análisis completo... Esto puede tardar unos minutos."):
                progress_bar = st.progress(0, text="Iniciando análisis...")
                
                df['Texto_Completo'] = df['Titulo'].fillna('').astype(str) + '. ' + df['Resumen'].fillna('').astype(str)
                textos = df['Texto_Completo'].tolist()
                
                if clasificar_temas:
                    progress_bar.progress(10, text="🔍 Descubriendo temas...")
                    temas_descubiertos = descubrir_temas_dinamicos(textos, num_temas)
                    st.session_state.temas_descubiertos = temas_descubiertos
                
                if detectar_duplicados:
                    progress_bar.progress(25, text="🔍 Detectando duplicados...")
                    df = detectar_duplicados_exactos(df)
                    df = detectar_duplicados_similares(df, umbral_similitud/100)
                
                mask_no_duplicados = ~(df.get('Es_Duplicado_Exacto', False)) & ~(df.get('Es_Duplicado_Similar', False))
                textos_unicos = df.loc[mask_no_duplicados, 'Texto_Completo'].tolist()

                if analizar_sentimiento and textos_unicos:
                    progress_bar.progress(40, text="💭 Analizando sentimiento...")
                    resultados_sentimiento = analizar_sentimiento_batch(textos_unicos, cliente_foco)
                    df.loc[mask_no_duplicados, 'Sentimiento'] = [r.get('sentimiento', 'Neutral') for r in resultados_sentimiento]
                    df.loc[mask_no_duplicados, 'Score_Sentimiento'] = [r.get('score', 0) for r in resultados_sentimiento]
                    df.loc[mask_no_duplicados, 'Razon_Sentimiento'] = [r.get('razon', '') for r in resultados_sentimiento]

                if clasificar_temas and 'temas_descubiertos' in st.session_state and textos_unicos:
                    progress_bar.progress(65, text="🏷️ Clasificando noticias en temas...")
                    resultados_temas = clasificar_temas_batch(textos_unicos, st.session_state.temas_descubiertos)
                    df.loc[mask_no_duplicados, 'Tema'] = [r.get('tema', 'Sin clasificar') for r in resultados_temas]

                # Propagar análisis a duplicados
                for grupo_col in ['Grupo_Duplicado_Exacto', 'Grupo_Duplicado_Similar']:
                    if grupo_col in df.columns:
                        for grupo_id in df[df[grupo_col] >= 0][grupo_col].unique():
                            grupo_mask = df[grupo_col] == grupo_id
                            original = df[grupo_mask & mask_no_duplicados]
                            if not original.empty:
                                if analizar_sentimiento and 'Sentimiento' in original.columns:
                                    df.loc[grupo_mask, 'Sentimiento'] = original.iloc[0]['Sentimiento']
                                    df.loc[grupo_mask, 'Score_Sentimiento'] = original.iloc[0]['Score_Sentimiento']
                                if clasificar_temas and 'Tema' in original.columns:
                                    df.loc[grupo_mask, 'Tema'] = original.iloc[0]['Tema']

                if generar_insights:
                    progress_bar.progress(85, text="🧠 Generando insights estratégicos...")
                    st.session_state.insights = generar_insights_estrategicos(df, cliente_foco)
                
                progress_bar.progress(100, text="✅ ¡Análisis completado!")
            
            st.session_state.df_analizado = df
            st.session_state.analysis_done = True
            st.balloons()
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
        st.exception(e)

if st.session_state.get('analysis_done', False):
    st.markdown("---")
    st.header("Resultados del Análisis")
    df = st.session_state.df_analizado
    
    tabs = st.tabs(["🧠 Insights Estratégicos", "📊 Dashboard", "🗂️ Datos Analizados", "🔍 Duplicados", "💬 Chat IA"])

    with tabs[0]:
        st.subheader("Insights Estratégicos y Recomendaciones")
        insights = st.session_state.get('insights', {})
        if insights:
            st.markdown("#### Resumen Ejecutivo")
            st.info(insights.get('resumen_ejecutivo', "No disponible."))
            st.markdown("#### Hallazgos Clave")
            for hallazgo in insights.get('hallazgos_clave', []):
                with st.container(border=True):
                    st.markdown(f"**Hallazgo:** {hallazgo.get('hallazgo')}")
                    st.caption(f"Evidencia: {hallazgo.get('evidencia')}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ✅ Oportunidades Estratégicas")
                for op in insights.get('oportunidades_estrategicas', []): st.success(f"• {op}")
            with col2:
                st.markdown("#### ⚠️ Riesgos Emergentes")
                for risk in insights.get('riesgos_emergentes', []): st.warning(f"• {risk}")
            st.markdown("#### 💡 Recomendaciones Accionables")
            for rec in insights.get('recomendaciones_accionables', []): st.markdown(f"**• {rec}**")
        else: st.warning("No se generaron insights. Habilita el módulo en el sidebar y vuelve a analizar.")
    
    with tabs[1]:
        st.subheader("Dashboard General")
        unicas_mask = (~df.get('Es_Duplicado_Exacto', False)) & (~df.get('Es_Duplicado_Similar', False))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Menciones", len(df))
        col2.metric("Menciones Únicas", unicas_mask.sum())
        if 'Audiencia' in df.columns: col3.metric("Audiencia Total", f"{pd.to_numeric(df['Audiencia'], errors='coerce').sum():,.0f}")
        if 'CPE' in df.columns: col4.metric("CPE Acumulado", f"${pd.to_numeric(df['CPE'], errors='coerce').sum():,.0f}")

        df_unicos = df[unicas_mask]
        colg1, colg2 = st.columns(2)
        with colg1:
            if 'Sentimiento' in df_unicos.columns:
                st.markdown("##### Distribución de Sentimiento (Únicas)")
                fig = px.pie(df_unicos, names='Sentimiento', color='Sentimiento', hole=.3,
                             color_discrete_map={'Muy Positivo':'#2ECC71', 'Positivo':'#85C1E2', 'Neutral':'#BDC3C7', 'Negativo':'#F39C12', 'Muy Negativo':'#E74C3C'})
                st.plotly_chart(fig, use_container_width=True)
        with colg2:
            if 'Tema' in df_unicos.columns:
                st.markdown("##### Top Temas (Únicas)")
                top_temas = df_unicos['Tema'].value_counts().nlargest(10).sort_values()
                fig = px.bar(top_temas, x=top_temas.values, y=top_temas.index, orientation='h', text_auto=True)
                fig.update_layout(yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Explorador de Datos Analizados")
        st.dataframe(df, use_container_width=True, height=500)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Descargar CSV Completo", csv, "analisis_noticias.csv", "text/csv", use_container_width=True)

    with tabs[3]:
        st.subheader("Análisis de Duplicados")
        tipo_ver = st.radio("Ver duplicados:", ['Exactos', 'Similares'], horizontal=True, label_visibility="collapsed")
        
        if tipo_ver == 'Exactos' and 'Grupo_Duplicado_Exacto' in df.columns:
            grupos = df[df['Grupo_Duplicado_Exacto'] >= 0].groupby('Grupo_Duplicado_Exacto')
            st.info(f"Se encontraron {len(grupos)} grupos de duplicados exactos.")
            for id_grupo, grupo in grupos:
                # --- CORRECCIÓN AQUÍ ---
                if not grupo.empty:
                    with st.expander(f"Grupo {id_grupo+1} ({len(grupo)} menciones): {grupo.iloc[0]['Titulo'][:60]}..."):
                        st.dataframe(grupo, use_container_width=True)

        elif tipo_ver == 'Similares' and 'Grupo_Duplicado_Similar' in df.columns:
            grupos = df[df['Grupo_Duplicado_Similar'] >= 0].groupby('Grupo_Duplicado_Similar')
            st.info(f"Se encontraron {len(grupos)} grupos de duplicados similares.")
            for id_grupo, grupo in grupos:
                # --- CORRECCIÓN AQUÍ ---
                if not grupo.empty:
                    with st.expander(f"Grupo {id_grupo+1} ({len(grupo)} menciones): {grupo.iloc[0]['Titulo'][:60]}..."):
                        st.dataframe(grupo, use_container_width=True)

    with tabs[4]:
        st.subheader("💬 Chat Inteligente con tus Datos")
        for item in st.session_state.chat_history:
            with st.chat_message("user"): st.markdown(item["pregunta"])
            with st.chat_message("assistant"): st.markdown(item["respuesta"])
        
        if prompt := st.chat_input("Pregúntale algo a tus datos... (ej. ¿Cuál es el tema con peor sentimiento?)"):
            st.chat_message("user").markdown(prompt)
            st.session_state.chat_history.append({"pregunta": prompt, "respuesta": ""})

            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    respuesta = chat_con_datos(prompt, df, st.session_state.chat_history)
                    st.markdown(respuesta)
            
            st.session_state.chat_history[-1]["respuesta"] = respuesta
            st.rerun()

# --- Pie de página ---
st.markdown("---")
if st.button("🗑️ Reiniciar Análisis Completo"):
    keys_to_keep = ['password_correct']
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.rerun()
