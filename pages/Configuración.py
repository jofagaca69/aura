"""
Página de Configuración del Sistema AURA
Permite gestionar variables de entorno y configuración
"""

from datetime import datetime
from pathlib import Path
import os
import sys

import streamlit as st
from dotenv import load_dotenv, set_key

# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.rag.document_loader import DocumentLoader
from src.rag.vector_store import VectorStore
from src.config import config

# Configuración de la página
st.set_page_config(
    page_title="Configuración - AURA",
    page_icon="⚙️",
    layout="wide"
)

# Inicializar session state
if "config_logs" not in st.session_state:
    st.session_state.config_logs = []

def log_message(message: str, level: str = "info"):
    """Registra un mensaje en el log (opcional, no crítico)"""
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        if "config_logs" not in st.session_state:
            st.session_state.config_logs = []
        st.session_state.config_logs.append({
            "timestamp": timestamp,
            "message": message,
            "level": level
        })
    except:
        pass  # Si falla el log, no es crítico

# Título de la página
st.markdown("""
<style>
    .config-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
</style>

<div class="config-header">
    <h1>⚙️ Configuración del Sistema AURA</h1>
    <p style="font-size: 1.1rem; margin-top: 1rem;">
        Gestiona las variables de entorno y configuración del sistema
    </p>
</div>
""", unsafe_allow_html=True)

# Contenido principal
st.markdown("""
Esta sección te permite gestionar la configuración del sistema AURA.
""")

# Crear tabs para organizar mejor
tab_env, tab_files, tab_rag = st.tabs([
    "🔐 Variables de Entorno",
    "📁 Gestión de Archivos rag",
    "🔮 Inicialización rag"
])

# ========================================
# TAB 1: VARIABLES DE ENTORNO
# ========================================
with tab_env:
    st.markdown("""
    Esta sección te permite gestionar las variables de entorno del sistema AURA.
    Las configuraciones se guardan en el archivo `.env` en la raíz del proyecto.
    """)

    # Encontrar o crear archivo .env
    env_path = Path(__file__).parent.parent / ".env"

    # Verificar si existe el archivo
    env_exists = env_path.exists()

    # Cargar variables actuales
    if env_exists:
        load_dotenv(env_path)

    # Información del archivo .env
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.info(f"📁 **Ubicación:** `{env_path}`")

    with col2:
        if env_exists:
            st.success("✅ Archivo existe")
        else:
            st.warning("⚠️ No existe")

    with col3:
        if st.button("📝 Crear .env", disabled=env_exists):
            env_path.touch()
            st.success("✅ Archivo .env creado")
            log_message("Archivo .env creado", "success")
            st.rerun()

    st.markdown("---")

    # Advertencia si no existe el archivo
    if not env_exists:
        st.error("""
        ⚠️ **El archivo `.env` no existe**
        
        Debes crear el archivo `.env` antes de poder configurar las variables de entorno.
        Haz click en el botón **"📝 Crear .env"** arriba para crearlo.
        """)
        st.info("""
        💡 **Nota:** Una vez creado el archivo, podrás configurar todas las variables de entorno
        necesarias para el funcionamiento del sistema AURA.
        """)
        st.stop()  # Detener la ejecución aquí

    # FORMULARIO DE CONFIGURACIÓN
    st.markdown("### 🔧 Variables de Entorno")

    # ============================================
    # CHECKBOX DE LANGSMITH FUERA DEL FORMULARIO
    # ============================================
    # IMPORTANTE: Debe estar fuera para ser interactivo
    st.markdown("#### 📊 LangSmith (Opcional - Monitoreo)")

    enable_langsmith = st.checkbox(
        "✅ Habilitar LangSmith Tracing",
        value=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        help="Activa el monitoreo y trazabilidad de las ejecuciones con LangSmith",
        key="langsmith_toggle"
    )

    if enable_langsmith:
        st.success("✅ **LangSmith habilitado.** Los campos de configuración aparecerán en el formulario de abajo.")
    else:
        st.info("ℹ️ **LangSmith deshabilitado.** Las variables de LangSmith no se guardarán.")

    st.markdown("---")

    with st.form("env_configuration"):
        st.markdown("#### 🤖 Google Gemini (Requerido)")

        google_api_key = st.text_input(
            "GOOGLE_API_KEY *",
            value=os.getenv("GOOGLE_API_KEY", ""),
            type="password",
            help="Tu API Key de Google Gemini para los agentes multi-agente",
            placeholder="AIzaSy..."
        )

        st.markdown("#### ⚙️ Configuración del Modelo")

        col1, col2 = st.columns(2)

        with col1:
            model_name = st.selectbox(
                "MODEL_NAME",
                options=["gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash"],
                index=0,
                help="Modelo de Google Gemini a utilizar"
            )

        with col2:
            temperature = st.slider(
                "TEMPERATURE",
                min_value=0.0,
                max_value=1.0,
                value=float(os.getenv("TEMPERATURE", "0.3")),
                step=0.05,
                help="Creatividad del modelo: 0=determinista, 1=creativo"
            )

        st.markdown("#### 📄 Configuración rag")

        col1, col2, col3 = st.columns(3)

        with col1:
            chunk_size = st.number_input(
                "CHUNK_SIZE",
                min_value=200,
                max_value=2000,
                value=int(os.getenv("CHUNK_SIZE", "800")),
                step=100,
                help="Tamaño de cada fragmento de texto"
            )

        with col2:
            chunk_overlap = st.number_input(
                "CHUNK_OVERLAP",
                min_value=0,
                max_value=500,
                value=int(os.getenv("CHUNK_OVERLAP", "150")),
                step=25,
                help="Solapamiento entre fragmentos consecutivos"
            )

        with col3:
            top_k_results = st.number_input(
                "TOP_K_RESULTS",
                min_value=1,
                max_value=20,
                value=int(os.getenv("TOP_K_RESULTS", "4")),
                step=1,
                help="Número de documentos a recuperar"
            )

        # Campos de LangSmith - Solo aparecen si el checkbox de arriba está marcado
        if enable_langsmith:
            st.markdown("#### 🔑 Credenciales de LangSmith")
            st.caption("💡 Configura tus credenciales de LangSmith para el monitoreo:")

            col1, col2 = st.columns(2)

            with col1:
                langchain_api_key = st.text_input(
                    "LANGCHAIN_API_KEY *",
                    value=os.getenv("LANGCHAIN_API_KEY", ""),
                    type="password",
                    help="Tu API Key de LangSmith",
                    placeholder="ls__..."
                )

            with col2:
                langchain_project = st.text_input(
                    "LANGCHAIN_PROJECT",
                    value=os.getenv("LANGCHAIN_PROJECT", ""),
                    help="Nombre del proyecto en LangSmith"
                )
        else:
            # Si LangSmith no está habilitado, establecer valores vacíos
            langchain_api_key = ""
            langchain_project = ""

        st.markdown("---")

        # Botones de acción
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            submit_button = st.form_submit_button(
                "💾 Guardar Configuración",
                use_container_width=True,
                type="primary"
            )

        with col2:
            show_current = st.form_submit_button(
                "👁️ Ver Actual",
                use_container_width=True
            )

        with col3:
            show_template = st.form_submit_button(
                "📋 Ver Template",
                use_container_width=True
            )

        with col4:
            verify_button = st.form_submit_button(
                "✅ Verificar",
                use_container_width=True
            )

        # LÓGICA DE LOS BOTONES
        if submit_button:
            # Validar que al menos Google API Key esté presente
            if not google_api_key or google_api_key.strip() == "":
                st.error("❌ GOOGLE_API_KEY es obligatoria. Por favor, proporciona una API Key válida.")
            else:
                try:
                    # Crear archivo si no existe
                    if not env_path.exists():
                        env_path.touch()

                    # Guardar todas las variables
                    set_key(str(env_path), "GOOGLE_API_KEY", google_api_key)
                    set_key(str(env_path), "MODEL_NAME", model_name)
                    set_key(str(env_path), "TEMPERATURE", str(temperature))
                    set_key(str(env_path), "CHUNK_SIZE", str(chunk_size))
                    set_key(str(env_path), "CHUNK_OVERLAP", str(chunk_overlap))
                    set_key(str(env_path), "TOP_K_RESULTS", str(top_k_results))
                    set_key(str(env_path), "LANGCHAIN_TRACING_V2", "true" if enable_langsmith else "false")

                    if enable_langsmith:
                        set_key(str(env_path), "LANGCHAIN_API_KEY", langchain_api_key)
                        set_key(str(env_path), "LANGCHAIN_PROJECT", langchain_project)

                    # Actualizar en runtime (para esta sesión)
                    os.environ["GOOGLE_API_KEY"] = google_api_key
                    os.environ["MODEL_NAME"] = model_name
                    os.environ["TEMPERATURE"] = str(temperature)
                    os.environ["CHUNK_SIZE"] = str(chunk_size)
                    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
                    os.environ["TOP_K_RESULTS"] = str(top_k_results)
                    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enable_langsmith else "false"

                    if enable_langsmith:
                        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
                        os.environ["LANGCHAIN_PROJECT"] = langchain_project

                    st.success("✅ Configuración guardada exitosamente en `.env`")
                    log_message("Configuración guardada en .env", "success")

                    # Instrucciones
                    st.info("""
                    **📝 Próximos Pasos:**
                    1. Los cambios se aplicarán en la próxima ejecución del sistema
                    2. Si ya tienes el sistema corriendo, reinícialo para aplicar los cambios
                    3. Verifica que todo funciona correctamente
                    """)

                except Exception as e:
                    st.error(f"❌ Error al guardar: {str(e)}")
                    log_message(f"Error guardando .env: {str(e)}", "error")

        if show_current:
            st.markdown("#### 📋 Configuración Actual")

            current_vars = {
                "GOOGLE_API_KEY": f"***{os.getenv('GOOGLE_API_KEY', '')[-4:]}" if os.getenv('GOOGLE_API_KEY') else "❌ No configurada",
                "MODEL_NAME": os.getenv("MODEL_NAME", "❌ No configurado"),
                "TEMPERATURE": os.getenv("TEMPERATURE", "❌ No configurado"),
                "CHUNK_SIZE": os.getenv("CHUNK_SIZE", "❌ No configurado"),
                "CHUNK_OVERLAP": os.getenv("CHUNK_OVERLAP", "❌ No configurado"),
                "TOP_K_RESULTS": os.getenv("TOP_K_RESULTS", "❌ No configurado"),
                "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "false"),
                "LANGCHAIN_API_KEY": f"***{os.getenv('LANGCHAIN_API_KEY', '')[-4:]}" if os.getenv('LANGCHAIN_API_KEY') else "No configurada",
                "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "No configurado"),
            }

            st.json(current_vars)

        if show_template:
            st.markdown("#### 📄 Template .env")

            template = """# ========================================
    # CONFIGURACIÓN SISTEMA AURA
    # ========================================
    
    # Google Gemini API (REQUERIDO)
    GOOGLE_API_KEY=your_google_api_key_here
    
    # Configuración del Modelo
    MODEL_NAME=gemini-2.5-flash-lite
    TEMPERATURE=0.3
    
    # Configuración rag
    CHUNK_SIZE=800
    CHUNK_OVERLAP=150
    TOP_K_RESULTS=4
    
    # LangSmith (Opcional - Monitoreo)
    LANGCHAIN_TRACING_V2=false
    LANGCHAIN_API_KEY=your_langsmith_api_key
    LANGCHAIN_PROJECT=AURA-Sistema-Multiagentes
    """

            st.code(template, language="bash")

            st.download_button(
                "📥 Descargar Template",
                data=template,
                file_name=".env.example",
                mime="text/plain",
                help="Descarga este template para usarlo como referencia"
            )

        if verify_button:
            st.markdown("#### ✅ Verificación de Variables")

            # Recargar variables desde .env
            load_dotenv(env_path, override=True)

            # Contador de estado
            errors = []
            warnings = []
            success = []

            # Variables requeridas
            st.markdown("**🔑 Variables Requeridas:**")

            google_key = os.getenv("GOOGLE_API_KEY")
            if google_key and google_key.strip():
                st.success(f"✅ **GOOGLE_API_KEY**: Configurada (termina en ...{google_key[-4:]})")
                success.append("GOOGLE_API_KEY")
            else:
                st.error("❌ **GOOGLE_API_KEY**: NO configurada - ⚠️ El sistema no funcionará sin esta key")
                errors.append("GOOGLE_API_KEY")

            st.markdown("---")

            # Variables opcionales del sistema
            st.markdown("**⚙️ Variables del Sistema:**")

            optional_vars = {
                "MODEL_NAME": ("gemini-2.5-flash-lite", "Modelo de IA"),
                "TEMPERATURE": ("0.3", "Temperatura del modelo"),
                "CHUNK_SIZE": ("800", "Tamaño de fragmentos"),
                "CHUNK_OVERLAP": ("150", "Solapamiento de fragmentos"),
                "TOP_K_RESULTS": ("4", "Documentos a recuperar")
            }

            for var_name, (default_value, description) in optional_vars.items():
                current_value = os.getenv(var_name)
                if current_value and current_value.strip():
                    st.success(f"✅ **{var_name}**: `{current_value}` - {description}")
                    success.append(var_name)
                else:
                    st.warning(f"⚠️ **{var_name}**: No configurada (usando default: `{default_value}`) - {description}")
                    warnings.append(var_name)

            st.markdown("---")

            # LangSmith
            st.markdown("**📊 LangSmith (Monitoreo Opcional):**")

            langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

            if langsmith_enabled:
                st.info("✅ **LangSmith**: Habilitado")

                langsmith_key = os.getenv("LANGCHAIN_API_KEY")
                if langsmith_key and langsmith_key.strip():
                    st.success(f"✅ **LANGCHAIN_API_KEY**: Configurada (termina en ...{langsmith_key[-4:]})")
                    success.append("LANGCHAIN_API_KEY")
                else:
                    st.error("❌ **LANGCHAIN_API_KEY**: No configurada - LangSmith no funcionará")
                    errors.append("LANGCHAIN_API_KEY")

                project = os.getenv("LANGCHAIN_PROJECT", "No configurado")
                st.info(f"📊 **LANGCHAIN_PROJECT**: `{project}`")
            else:
                st.info("ℹ️ **LangSmith**: Deshabilitado")

            st.markdown("---")

            # Resumen general
            st.markdown("### 📊 Resumen de Verificación")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("✅ Configuradas", len(success))

            with col2:
                st.metric("⚠️ Con Default", len(warnings))

            with col3:
                st.metric("❌ Faltantes", len(errors))

            # Estado general
            if len(errors) > 0:
                st.error(f"""
                **⚠️ Acción Requerida**
                
                Hay {len(errors)} variable(s) crítica(s) sin configurar: {', '.join(errors)}
                
                El sistema no podrá funcionar correctamente hasta que configures estas variables.
                """)
            elif len(warnings) > 0:
                st.warning(f"""
                **ℹ️ Configuración Parcial**
                
                {len(warnings)} variable(s) están usando valores por defecto: {', '.join(warnings)}
                
                El sistema funcionará, pero puedes optimizarlo configurando estas variables.
                """)
            else:
                st.info("ℹ️ LangSmith: Deshabilitado")

        # SECCIÓN DE AYUDA
        st.markdown("---")
        st.markdown("### 💡 Ayuda y Recursos")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🔑 Obtener API Keys:**
            
            - **Google Gemini:**
              1. Ve a [Google AI Studio](https://makersuite.google.com/app/apikey)
              2. Crea o selecciona un proyecto
              3. Genera una API Key
              4. Cópiala aquí
            
            - **LangSmith (Opcional):**
              1. Ve a [LangSmith](https://smith.langchain.com)
              2. Crea una cuenta
              3. Ve a Settings → API Keys
              4. Genera una nueva key
            """)

        with col2:
            st.markdown("""
            **⚙️ Descripción de Variables:**
            
            - **MODEL_NAME:** Modelo de Gemini a usar
              - `2.0-flash-exp`: Último modelo experimental
              - `1.5-pro`: Más potente y completo
              - `1.5-flash`: Rápido y económico
            
            - **TEMPERATURE:** Creatividad (0-1)
              - Bajo (0-0.3): Respuestas consistentes
              - Alto (0.7-1): Respuestas creativas
            
            - **CHUNK_SIZE:** Tamaño de fragmentos
              - Menor: Más precisión
              - Mayor: Más contexto
            
            - **TOP_K_RESULTS:** Documentos a recuperar
              - Menor: Más rápido
              - Mayor: Más completo
            """)

        # Advertencia de seguridad
        st.warning("""
        **🔒 Seguridad:**
        - NUNCA compartas tu archivo `.env` públicamente
        - Asegúrate de tener `.env` en tu `.gitignore`
        - Las API Keys son sensibles y personales
        """)

# ========================================
# TAB 2: GESTIÓN DE ARCHIVOS rag
# ========================================
with tab_files:
    st.markdown("### 📁 Gestión de Archivos para rag")

    st.markdown("""
    Esta sección te permite gestionar los archivos de documentos que serán procesados
    y almacenados en el sistema rag (Retrieval Augmented Generation).
    
    **Formatos soportados:** pdf, txt, csv, json, docx, doc, xlsx, xls
    """)

    # Crear directorio para uploads si no existe   Path(__file__).parent.parent / ".env"
    upload_dir = Path(__file__).parent.parent / "data/uploads"
    upload_dir.mkdir(exist_ok=True)

    # Área de subida de archivos
    st.markdown("#### 📤 Subir Nuevos Archivos")

    uploaded_files = st.file_uploader(
        "Selecciona archivos para el sistema rag",
        accept_multiple_files=True,
        type=['pdf','txt','csv','json','docx','doc','xlsx','xls'],
        help="Puedes subir múltiples archivos a la vez"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} archivo(s) seleccionado(s)")

        # Mostrar lista de archivos
        st.markdown("**📄 Archivos seleccionados:**")
        for file in uploaded_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.text(f"📄 {file.name}")
            with col2:
                # Formatear tamaño
                size_kb = len(file.getvalue()) / 1024
                st.text(f"{size_kb:.1f} KB")
            with col3:
                st.text(Path(file.name).suffix.upper())

        # Botón para guardar archivos
        if st.button("💾 Guardar Archivos", type="primary", use_container_width=True):
            saved_count = 0
            for uploaded_file in uploaded_files:
                try:
                    # Guardar archivo
                    save_path = upload_dir / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_count += 1
                    log_message(f"Archivo guardado: {uploaded_file.name}", "success")
                except Exception as e:
                    st.error(f"Error guardando {uploaded_file.name}: {e}")
                    log_message(f"Error guardando {uploaded_file.name}: {e}", "error")

            if saved_count > 0:
                st.success(f"✅ {saved_count} archivo(s) guardado(s) en `{upload_dir}`")
                st.rerun()

    st.markdown("---")

    # Mostrar archivos existentes
    st.markdown("#### 📂 Archivos Existentes")

    existing_files = list(upload_dir.glob("*.*"))

    if existing_files:
        st.info(f"📊 Total: {len(existing_files)} archivo(s)")

        # Tabla de archivos
        for file in sorted(existing_files, key=lambda x: x.stat().st_mtime, reverse=True):
            with st.expander(f"📄 {file.name}"):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    st.text(f"**Nombre:** {file.name}")

                with col2:
                    size = file.stat().st_size / 1024
                    st.text(f"**Tamaño:** {size:.1f} KB")

                with col3:
                    modified = datetime.fromtimestamp(file.stat().st_mtime)
                    st.text(f"**Modificado:** {modified.strftime('%d/%m/%Y')}")

                with col4:
                    if st.button("🗑️ Eliminar", key=f"delete_{file.name}"):
                        try:
                            file.unlink()
                            st.success(f"✅ Archivo eliminado: {file.name}")
                            log_message(f"Archivo eliminado: {file.name}", "info")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error eliminando archivo: {e}")
                            log_message(f"Error eliminando {file.name}: {e}", "error")

                # Botón para descargar
                try:
                    with open(file, "rb") as f:
                        st.download_button(
                            "📥 Descargar",
                            data=f.read(),
                            file_name=file.name,
                            mime="application/octet-stream",
                            key=f"download_{file.name}"
                        )
                except Exception as e:
                    st.warning(f"⚠️ No se puede descargar: {e}")
    else:
        st.info("📭 No hay archivos en el sistema todavía. Sube algunos archivos arriba para comenzar.")

    st.markdown("---")

    # Información sobre el directorio
    st.markdown("#### ℹ️ Información del Sistema")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **📁 Directorio de Archivos:**
        `{upload_dir.absolute()}`
        
        Los archivos subidos se almacenan en este directorio y pueden ser procesados por el sistema rag.
        """)

    with col2:
        st.markdown("""
        **💡 Tipos de Archivos Recomendados:**
        
        - **📄 .txt**: Documentos de texto plano
        - **📊 .csv**: Datos estructurados (productos, inventario)
        - **📋 .json**: Datos estructurados y configuración
        - **📝 .md**: Documentación en Markdown
        - **📑 .pdf**: Documentos PDF (requiere lector)
        
        **Tamaño recomendado:** < 10 MB por archivo
        """)

# ========================================
# TAB 3: INICIALIZACIÓN rag
# ========================================
with tab_rag:
    st.markdown("### 🔮 Inicialización y Procesamiento rag")

    st.markdown("""
    Esta sección te permite inicializar y procesar los documentos para crear el vectorstore
    que será utilizado por el sistema rag (Retrieval Augmented Generation).
    
    **¿Qué hace este proceso?**
    - Carga documentos desde el directorio configurado
    - Divide los documentos en chunks más pequeños
    - Genera embeddings (representaciones vectoriales)
    - Almacena los embeddings en una base de datos vectorial (ChromaDB)
    """)

    st.markdown("---")

    # Verificar configuración
    st.markdown("#### ⚙️ Verificación de Configuración")

    # Verificar si existe el directorio de documentos
    documents_dir = Path(__file__).parent.parent / "data" / "uploads"
    vectorstore_dir = Path(__file__).parent.parent / "data" / "chroma_db"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📂 Directorio de Documentos:**")
        st.info(f"`{documents_dir.absolute()}`")

        if documents_dir.exists():
            files = list(documents_dir.glob("*.*"))
            st.success(f"✅ Directorio existe con {len(files)} archivo(s)")

            if files:
                st.markdown("**Archivos encontrados:**")
                for file in files:
                    size = file.stat().st_size / 1024
                    st.caption(f"📄 {file.name} ({size:.1f} KB)")
        else:
            st.error("❌ Directorio no existe")
            st.caption("Sube archivos en la pestaña 'Gestión de Archivos rag'")

    with col2:
        st.markdown("**🗄️ Vectorstore:**")
        st.info(f"`{vectorstore_dir.absolute()}`")

        vectorstore_exists = vectorstore_dir.exists() and len(list(vectorstore_dir.glob("*"))) > 0

        if vectorstore_exists:
            st.success("✅ Vectorstore existente encontrado")

            # Información del vectorstore
            files_in_vectorstore = list(vectorstore_dir.rglob("*.*"))
            st.caption(f"📊 {len(files_in_vectorstore)} archivo(s) de base de datos")
        else:
            st.warning("⚠️ Vectorstore no existe")
            st.caption("Necesitas procesar los documentos")

    st.markdown("---")

    # Información de configuración rag
    st.markdown("#### 📋 Configuración rag Actual")

    col1, col2, col3 = st.columns(3)

    with col1:
        chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        st.metric("📏 Tamaño de Chunk", f"{chunk_size} caracteres")

    with col2:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
        st.metric("🔄 Solapamiento", f"{chunk_overlap} caracteres")

    with col3:
        top_k = int(os.getenv("TOP_K_RESULTS", "4"))
        st.metric("🔢 Top K Results", top_k)

    st.markdown("---")

    # Acciones
    st.markdown("#### 🎯 Acciones")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🚀 Procesar Documentos**")
        st.markdown("""
        Esta acción va a:
        1. Cargar todos los documentos del directorio
        2. Dividirlos en chunks
        3. Generar embeddings
        4. Guardar en ChromaDB
        
        **Tiempo estimado:** 2-10 minutos dependiendo del tamaño
        """)

        # Botón para procesar
        if st.button("🚀 Procesar Documentos", type="primary", use_container_width=True):
            if not documents_dir.exists() or len(list(documents_dir.glob("*.*"))) == 0:
                st.error("❌ No hay documentos para procesar")
                st.info("💡 Sube archivos en la pestaña 'Gestión de Archivos rag'")
            else:
                try:
                    with st.spinner("🔄 Procesando documentos..."):
                        # Mostrar progreso
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Paso 1: Cargar documentos
                        status_text.text("📂 Paso 1/3: Cargando documentos...")
                        progress_bar.progress(20)

                        loader = DocumentLoader()
                        documents = loader.load_documents(str(documents_dir))

                        if not documents:
                            st.error("❌ No se pudieron cargar los documentos")
                            st.stop()

                        st.success(f"✅ {len(documents)} documento(s) cargado(s)")

                        # Paso 2: Crear vectorstore
                        status_text.text("🔮 Paso 2/3: Creando embeddings y vectorstore...")
                        progress_bar.progress(50)

                        vector_store = VectorStore()

                        # Eliminar vectorstore existente si existe
                        if vectorstore_dir.exists():
                            import shutil
                            shutil.rmtree(vectorstore_dir)

                        # Crear nuevo vectorstore
                        vector_store.create_vectorstore(documents)

                        progress_bar.progress(80)
                        status_text.text("💾 Paso 3/3: Guardando vectorstore...")

                        progress_bar.progress(100)

                        # Limpiar estado de progreso
                        progress_bar.empty()
                        status_text.empty()

                        st.success("✅ ¡Vectorstore creado exitosamente!")
                        st.balloons()

                        # Mostrar estadísticas
                        st.markdown("**📊 Estadísticas del Procesamiento:**")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("📄 Documentos", len(documents))

                        with col2:
                            # Estimar chunks (aproximadamente)
                            estimated_chunks = sum(len(doc.page_content) // chunk_size for doc in documents)
                            st.metric("📝 Chunks Estimados", estimated_chunks)

                        with col3:
                            st.metric("💾 Vectorstore", "✅ Creado")

                        log_message(f"Vectorstore creado con {len(documents)} documentos", "success")

                except Exception as e:
                    st.error(f"❌ Error procesando documentos: {str(e)}")
                    st.exception(e)
                    log_message(f"Error procesando: {str(e)}", "error")

    with col2:
        st.markdown("**🔄 Recargar Vectorstore**")
        st.markdown("""
        Si ya existe un vectorstore, puedes recargarlo sin reprocesar los documentos.
        
        **Útil cuando:**
        - Ya procesaste los documentos anteriormente
        - Solo necesitas recargar el vectorstore existente
        """)

        # Botón para recargar
        if st.button("🔄 Recargar Vectorstore", use_container_width=True):
            if not vectorstore_exists:
                st.error("❌ No existe un vectorstore para recargar")
                st.info("💡 Primero procesa los documentos")
            else:
                try:
                    with st.spinner("🔄 Recargando vectorstore..."):
                        vector_store = VectorStore()
                        vector_store.load_vectorstore()
                        st.success("✅ Vectorstore recargado exitosamente")
                        log_message("Vectorstore recargado", "success")
                except Exception as e:
                    st.error(f"❌ Error recargando vectorstore: {str(e)}")
                    log_message(f"Error recargando: {str(e)}", "error")

    st.markdown("---")

    # Información adicional
    st.markdown("#### 💡 Información Adicional")

    with st.expander("📖 ¿Qué es rag y cómo funciona?"):
        st.markdown("""
        **rag (Retrieval Augmented Generation)** es una técnica que combina:
        
        1. **Retrieval (Búsqueda)**: Busca información relevante en una base de datos vectorial
        2. **Augmentation (Aumento)**: Agrega esa información como contexto
        3. **Generation (Generación)**: Genera respuestas usando el contexto encontrado
        
        **Proceso completo:**
        - Los documentos se dividen en chunks más pequeños
        - Cada chunk se convierte en un vector (embedding)
        - Los vectores se almacenan en ChromaDB
        - Cuando el usuario hace una consulta:
          1. Se convierte la consulta en un vector
          2. Se buscan los chunks más similares
          3. Se usan esos chunks como contexto
          4. Se genera una respuesta informada
        
        **Ventajas:**
        - Respuestas más precisas basadas en datos reales
        - Puede actualizarse agregando nuevos documentos
        - Reduce alucinaciones del modelo
        """)

    with st.expander("🔧 Configuración Avanzada"):
        st.markdown("""
        **Puedes ajustar estos parámetros en la pestaña "Variables de Entorno":**
        
        - **CHUNK_SIZE**: Tamaño de cada fragmento de documento (default: 800)
          - Menor: Más granularidad, más fragmentos
          - Mayor: Más contexto por fragmento
        
        - **CHUNK_OVERLAP**: Caracteres que se solapan entre fragmentos (default: 150)
          - Ayuda a mantener contexto entre fragmentos
          - Evita cortar ideas a la mitad
        
        - **TOP_K_RESULTS**: Número de fragmentos a recuperar (default: 4)
          - Menor: Más rápido, menos contexto
          - Mayor: Más contexto, más lento
        
        **Recomendaciones:**
        - Para documentos técnicos: CHUNK_SIZE = 500-600
        - Para documentos largos: CHUNK_SIZE = 800-1000
        - Para documentos cortos: CHUNK_SIZE = 400-600
        """)

# Sidebar con logs
with st.sidebar:
    st.markdown("### 📋 Log de Configuración")

    if st.button("🗑️ Limpiar Log", use_container_width=True):
        st.session_state.config_logs = []
        st.rerun()

    # Mostrar últimos 10 mensajes
    if st.session_state.config_logs:
        for log_entry in st.session_state.config_logs[-10:]:
            level = log_entry['level']
            icon = {
                'info': 'ℹ️',
                'success': '✅',
                'warning': '⚠️',
                'error': '❌'
            }.get(level, 'ℹ️')

            st.text(f"{log_entry['timestamp']} {icon}")
            st.caption(log_entry['message'])
            st.divider()
    else:
        st.info("No hay mensajes en el log")

