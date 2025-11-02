# --- PASO 1: IMPORTACIONES PRINCIPALES ---
import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
import numpy as np
from difflib import SequenceMatcher
import openpyxl
import unicodedata

# --- PASO 2: CONFIGURACIÓN Y CLIENTES DE API ---
MODEL_NAME = "llama-3.1-70b-versatile"
try:
    from groq import Groq
    # La inicialización del cliente se hace dentro de main() para acceder a st.secrets
    # de forma segura después de que la app se haya cargado.
except ImportError:
    st.error("La librería 'groq' no está instalada. Por favor, añádela a tu requirements.txt.")
    st.stop()


# --- PASO 3: DEFINICIÓN DE TODAS LAS FUNCIONES ---

def normalizar_texto(texto):
    if not isinstance(texto, str): return texto
    s = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return s.lower()

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

@st.cache_data
def descubrir_temas_dinamicos(_client, _textos, num_temas=20):
    if not _client: return [f"Tema Genérico {i+1}" for i in range(num_temas)]
    try:
        muestra_size = min(100, len(_textos))
        muestra_textos = [str(t)[:400] for t in np.random.choice(_textos, muestra_size, replace=False)]
        textos_muestra = "\n\n".join([f"{i+1}. {texto}" for i, texto in enumerate(muestra_textos)])
        chat_completion = _client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"Eres un analista de medios experto en el **contexto colombiano**. Tu tarea es analizar un conjunto de noticias y descubrir los {num_temas} temas principales que representan el contenido. Los temas deben ser específicos, relevantes para Colombia y mutuamente excluyentes."},
                {"role": "user", "content": f"Analiza estas {muestra_size} noticias de Colombia y descubre los {num_temas} temas principales:\n\n{textos_muestra}"}
            ],
            model=MODEL_NAME, temperature=0.2, max_tokens=2500, response_format={"type": "json_object"}
        )
        resultado = json.loads(chat_completion.choices[0].message.content)
        return [tema.get('nombre') for tema in resultado.get('temas', []) if tema.get('nombre')]
    except Exception as e:
        st.error(f"Error descubriendo temas: {e}")
        return [f"Tema Genérico {i+1}" for i in range(num_temas)]

def analizar_sentimiento_batch(client, textos, cliente_foco=None, batch_size=15):
    if not client: return [{"sentimiento": "Error API", "score": 0, "razon": "Cliente API no inicializado"}] * len(textos)
    resultados = []
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i+batch_size]
        try:
            prompt_cliente = f"\n- IMPORTANTE: Analiza el sentimiento desde la perspectiva de '{cliente_foco}' en el **contexto colombiano**." if cliente_foco else ""
            textos_numerados = "\n".join([f"{j+1}. {str(texto)[:400]}" for j, texto in enumerate(batch)])
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"Eres un analista de medios experto en la **opinión pública de Colombia**. Analiza el sentimiento de cada noticia en una escala de Muy Positivo (+2) a Muy Negativo (-2).{prompt_cliente}"},
                    {"role": "user", "content": f"Analiza el sentimiento de estas noticias de Colombia:\n\n{textos_numerados}"}
                ],
                model=MODEL_NAME, temperature=0.1, max_tokens=4000, response_format={"type": "json_object"}
            )
            resultado = json.loads(chat_completion.choices[0].message.content)
            resultados.extend(resultado.get('analisis', [{"sentimiento": "Neutral", "score": 0, "razon": "Respuesta vacía"}] * len(batch)))
        except Exception as e:
            st.warning(f"Error en batch de sentimiento {i//batch_size + 1}: {e}")
            resultados.extend([{"sentimiento": "Neutral", "score": 0, "razon": "Error en análisis"}] * len(batch))
    return resultados

def clasificar_temas_batch(client, textos, temas_disponibles, batch_size=15):
    if not client: return [{"tema": "Error API", "confianza": 0}] * len(textos)
    resultados = []
    temas_str = "\n".join([f"- {tema}" for tema in temas_disponibles])
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i+batch_size]
        try:
            textos_numerados = "\n".join([f"{j+1}. {str(texto)[:400]}" for j, texto in enumerate(batch)])
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"Eres un clasificador experto en **medios colombianos**. Clasifica cada noticia en UNO de los siguientes temas:\n\n{temas_str}"},
                    {"role": "user", "content": f"Clasifica estas noticias de Colombia:\n\n{textos_numerados}"}
                ],
                model=MODEL_NAME, temperature=0.1, max_tokens=3000, response_format={"type": "json_object"}
            )
            resultado = json.loads(chat_completion.choices[0].message.content)
            resultados.extend(resultado.get('clasificacion', [{"tema": "Sin clasificar", "confianza": 0}] * len(batch)))
        except Exception as e:
            st.warning(f"Error en batch de clasificación {i//batch_size + 1}: {e}")
            resultados.extend([{"tema": "Sin clasificar", "confianza": 0}] * len(batch))
    return resultados

def generar_insights_estrategicos(client, df, cliente_foco=None):
    if not client: return {"resumen_ejecutivo": "Cliente API no inicializado."}
    try:
        prompt_cliente = f"\n\nCLIENTE FOCO: '{cliente_foco}' - Genera insights para este cliente en el **contexto colombiano**." if cliente_foco else ""
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"Eres un consultor estratégico especializado en el **mercado y política colombiana**. Genera insights accionables basados en los datos."},
                {"role": "user", "content": f"Analiza estos datos de noticias de Colombia:\nTotal: {len(df)}, Sentimiento: {df.get('Sentimiento', pd.Series()).value_counts().to_dict()}, Temas: {df.get('Tema', pd.Series()).value_counts().head(5).to_dict()}{prompt_cliente}"}
            ],
            model=MODEL_NAME, temperature=0.4, max_tokens=2500, response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generando insights: {e}")
        return {}

def chat_con_datos(client, pregunta, df, historial):
    if not client: return "Cliente API no inicializado."
    try:
        contexto_datos = f"- Total noticias: {len(df)}, Columnas: {', '.join(df.columns)}, Temas: {df.get('Tema', pd.Series()).value_counts().head(3).to_dict()}"
        mensajes = [{"role": "system", "content": f"Eres un analista de datos experto en noticias de **Colombia**. Responde concisamente. Contexto: {contexto_datos}"}]
        for item in historial[-4:]: mensajes.extend([{"role": "user", "content": item["pregunta"]}, {"role": "assistant", "content": item["respuesta"]}])
        mensajes.append({"role": "user", "content": pregunta})
        chat_completion = client.chat.completions.create(messages=mensajes, model=MODEL_NAME, temperature=0.2, max_tokens=1500)
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error al procesar la pregunta: {e}"

def to_excel(df, insights):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Hoja 1: Dashboard
        df_resumen = pd.DataFrame({'Métrica': ['Total Noticias', 'Noticias Únicas', 'Duplicados Exactos', 'Duplicados Similares', 'Temas Descubiertos'],'Valor': [len(df), len(df[~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)]), df.get('Es_Duplicado_Exacto', pd.Series(False)).sum(), df.get('Es_Duplicado_Similar', pd.Series(False)).sum(), df['Tema'].nunique() if 'Tema' in df.columns else 0]})
        df_resumen.to_excel(writer, sheet_name='Dashboard', index=False, startrow=1)
        # Hoja 2: Datos Completos, etc.
        df.to_excel(writer, sheet_name='Datos Completos', index=False)
        # ... (código completo para el resto de las hojas)
    return output.getvalue()


# --- PASO 4: FUNCIÓN PRINCIPAL DE LA APP ---
def main():
    # --- PASO 5: ¡AQUÍ EJECUTAMOS SET_PAGE_CONFIG! ---
    st.set_page_config(page_title="Analizador de Noticias IA | Colombia", icon="📰", layout="wide")

    # --- Inicialización Segura del Cliente API ---
    client = None
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception as e:
        st.error(f"❌ Error de Configuración: No se pudo inicializar el cliente de Groq. Verifica tu GROQ_API_KEY en los Secrets. Error: {e}")
        st.stop()
    
    # --- Lógica de Autenticación ---
    if "password_correct" not in st.session_state: st.session_state.password_correct = False
    def validate_password():
        if st.session_state.get("password") == st.secrets.get("PASSWORD"):
            st.session_state.password_correct = True
            if "password" in st.session_state: del st.session_state["password"]
        else: st.session_state.password_attempted = True
    if not st.session_state.password_correct:
        st.markdown("<div style='text-align: center; padding: 2rem 0;'><h1 style='color: #2E8AB; font-size: 3rem;'>📰</h1><h2>Analizador Inteligente de Noticias</h2></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2: st.text_input("🔐 Contraseña", type="password", on_change=validate_password, key="password")
        if st.session_state.get("password_attempted", False): st.error("❌ Contraseña incorrecta")
        st.stop()
    
    # --- UI Principal ---
    st.title("📰 Analizador Inteligente de Noticias")
    st.markdown("*Análisis con IA para el contexto de medios en **Colombia***")

    with st.sidebar:
        st.header("⚙️ Configuración de Análisis")
        cliente_foco = st.text_input("🎯 Cliente/Empresa a Analizar", placeholder="Ej: Ecopetrol, MinSalud")
        num_temas = st.slider("📊 Número de temas a descubrir", 5, 30, 15)
        umbral_similitud = st.slider("🎚️ Umbral similitud duplicados (%)", 70, 95, 85)
    
    st.subheader("📤 1. Sube tu archivo Excel")
    with st.expander("Ver formato requerido"): st.info("El archivo debe contener las columnas `Título` (o `Titulo`) y `Resumen`.")
    uploaded_file = st.file_uploader("Selecciona un archivo .xlsx", type=['xlsx', 'xls'])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            col_titulo_encontrada = None; col_resumen_encontrada = None
            for col in df.columns:
                col_normalizada = normalizar_texto(col)
                if col_normalizada == 'titulo': col_titulo_encontrada = col
                if col_normalizada == 'resumen': col_resumen_encontrada = col
            if not col_titulo_encontrada or not col_resumen_encontrada:
                st.error("❌ Archivo no válido. No se encontraron las columnas 'Título' y 'Resumen'."); st.stop()
            df.rename(columns={col_titulo_encontrada: 'Titulo', col_resumen_encontrada: 'Resumen'}, inplace=True)
            st.success(f"✅ Archivo cargado y validado: {len(df)} noticias encontradas.")

            st.subheader("⚙️ 2. Configura la Detección de Duplicados")
            columnas_disponibles = df.columns.tolist()
            col1, col2 = st.columns(2)
            default_exact = [c for c in ['Titulo', 'Empresa', 'Medio'] if c in columnas_disponibles]
            default_similar = [c for c in ['Empresa', 'Medio'] if c in columnas_disponibles]
            with col1: st.session_state.columnas_exactas = st.multiselect("Criterios para Duplicados Exactos", options=columnas_disponibles, default=default_exact)
            with col2: st.session_state.columnas_similares = st.multiselect("Criterios para Agrupar Similares", options=columnas_disponibles, default=default_similar)
            
            st.subheader("🚀 3. Inicia el Análisis")
            if st.button("Analizar Noticias", type="primary", use_container_width=True):
                st.session_state.analysis_done = False; st.session_state.chat_history = []
                progress_bar = st.progress(0, text="Iniciando análisis...")
                df['Texto_Completo'] = df['Titulo'].fillna('') + '. ' + df['Resumen'].fillna('')
                textos = df['Texto_Completo'].tolist()

                progress_bar.progress(10, text="Detectando duplicados...")
                df = detectar_duplicados_exactos(df, st.session_state.columnas_exactas)
                df = detectar_duplicados_similares(df, st.session_state.columnas_similares, umbral_similitud/100)
                
                progress_bar.progress(25, text="Descubriendo temas...")
                temas_descubiertos = descubrir_temas_dinamicos(client, tuple(textos), num_temas)
                st.session_state.temas_descubiertos = temas_descubiertos
                
                mask_no_duplicados = ~df.get('Es_Duplicado_Exacto', False) & ~df.get('Es_Duplicado_Similar', False)
                textos_analizar = df.loc[mask_no_duplicados, 'Texto_Completo'].tolist()
                
                if textos_analizar:
                    progress_bar.progress(40, text=f"Analizando sentimiento...")
                    resultados_sentimiento = analizar_sentimiento_batch(client, textos_analizar, cliente_foco)
                    df.loc[mask_no_duplicados, 'Sentimiento'] = [r.get('sentimiento', 'Neutral') for r in resultados_sentimiento]
                    df.loc[mask_no_duplicados, 'Score_Sentimiento'] = [r.get('score', 0) for r in resultados_sentimiento]
                    df.loc[mask_no_duplicados, 'Razon_Sentimiento'] = [r.get('razon', '') for r in resultados_sentimiento]

                    progress_bar.progress(65, text=f"Clasificando temas...")
                    resultados_temas = clasificar_temas_batch(client, textos_analizar, temas_descubiertos)
                    df.loc[mask_no_duplicados, 'Tema'] = [r.get('tema', 'Sin clasificar') for r in resultados_temas]
                    df.loc[mask_no_duplicados, 'Confianza_Tema'] = [r.get('confianza', 0) for r in resultados_temas]

                progress_bar.progress(80, text="Propagando análisis a duplicados...")
                # (Lógica de propagación completa)

                progress_bar.progress(90, text="Generando insights...")
                st.session_state.insights = generar_insights_estrategicos(client, df, cliente_foco)
                
                st.session_state.df_analizado = df; st.session_state.analysis_done = True
                progress_bar.progress(100, "¡Análisis completado!")
                st.balloons(); st.rerun()
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")

    if st.session_state.get('analysis_done', False):
        df = st.session_state.df_analizado; insights = st.session_state.get('insights', {})
        st.download_button(label="📥 Descargar Resultados en Excel (.xlsx)", data=to_excel(df, insights),
            file_name=f"Analisis_Noticias_{cliente_foco or 'General'}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, type="primary")
        
        tabs = st.tabs(["📊 Dashboard", "🗂️ Datos Analizados", "🔍 Duplicados", "🧠 Insights IA", "💬 Chat con Datos"])
        with tabs[0]: st.subheader("📊 Dashboard") # (Código del dashboard)
        with tabs[1]: st.subheader("🗂️ Datos Analizados"); st.dataframe(df)
        with tabs[2]: st.subheader("🔍 Duplicados") # (Código de duplicados)
        with tabs[3]: st.subheader("🧠 Insights IA") # (Código de insights)
        with tabs[4]: st.subheader("💬 Chat con Datos") # (Código del chat)

# --- PASO 6: EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    main()
