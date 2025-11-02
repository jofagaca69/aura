"""
Chat con Agente de IA - Sistema AURA
Basado en la guía oficial de Streamlit v2
"""

import streamlit as st
import time
from datetime import datetime
import os

# Importar componentes del sistema AURA
from src.orchestator import MultiAgentOrchestrator
from src.rag.vector_store import VectorStore
from src.rag.document_loader import DocumentLoader
from src.config import config

# ========================================
# INICIALIZACIÓN DEL SISTEMA
# ========================================
@st.cache_resource
def initialize_system():
    """
    Inicializa el sistema AURA cargando el VectorStore existente
    Se ejecuta solo una vez gracias a @st.cache_resource
    
    Returns:
        VectorStore si existe y se carga correctamente, None si no existe
    """
    try:
        # Validar configuración
        config.validate()
        config.setup_langsmith()
        
        # Verificar si existe el VectorStore procesado
        if not os.path.exists(config.CHROMA_DIR):
            return None
        
        # Verificar que el directorio no esté vacío
        if not any(os.scandir(config.CHROMA_DIR)):
            return None
        
        # Cargar VectorStore existente
        print("🚀 Cargando VectorStore existente...")
        vector_store = VectorStore()
        vector_store.load_vectorstore()
        print("✓ VectorStore cargado correctamente")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Error cargando VectorStore: {str(e)}")
        return None


# ========================================
# GENERADOR DE RESPUESTAS CON STREAMING
# ========================================
def response_generator(response_text: str):
    """
    Genera respuestas con efecto de streaming preservando el formato
    
    Args:
        response_text: Texto de respuesta del agente
    """
    # Dividir el texto en líneas para preservar los saltos de línea
    lines = response_text.split('\n')
    
    for i, line in enumerate(lines):
        # Procesar cada palabra de la línea
        words = line.split()
        for j, word in enumerate(words):
            yield word + " "
            time.sleep(0.02)
        
        # Agregar salto de línea al final de cada línea (excepto la última)
        if i < len(lines) - 1:
            yield "\n"

# ========================================
# CONFIGURACIÓN DE LA PÁGINA
# ========================================
st.set_page_config(
    page_title="Chat AURA",
    page_icon="🤖",
    layout="centered"
)

# ========================================
# INICIALIZAR SISTEMA Y SESSION STATE
# ========================================
# Inicializar VectorStore (solo una vez)
vector_store = initialize_system()

# Verificar que el VectorStore existe
if vector_store is None:
    st.error("⚠️ **El sistema RAG no ha sido inicializado**")
    
    st.markdown("""
    ### 🔧 Configuración Requerida
    
    Antes de poder usar el chat, necesitas preparar el sistema siguiendo estos pasos:
    
    #### 📋 Pasos para inicializar el sistema:
    
    1. **Ve a la página de Configuración** (en la barra lateral izquierda)
    
    2. **Configura las variables de entorno:**
       - Ve a la pestaña "🔐 Variables de Entorno"
       - Ingresa tu `GOOGLE_API_KEY`
       - Configura los parámetros del modelo
       - Guarda la configuración
    
    3. **Sube tus archivos de productos:**
       - Ve a la pestaña "📁 Gestión de Archivos RAG"
       - Sube archivos con información de productos (Excel, CSV, JSON, PDF, etc.)
       - Los archivos deben contener la información que AURA usará para hacer recomendaciones
    
    4. **Procesa los documentos:**
       - Ve a la pestaña "🔮 Inicialización RAG"
       - Haz clic en "🚀 Procesar Documentos"
       - Espera a que el sistema procese los archivos y cree el VectorStore
       - Este proceso puede tardar 2-10 minutos dependiendo de la cantidad de datos
    
    5. **¡Listo!** Vuelve a esta página y podrás comenzar a chatear con AURA
    
    ---
    
    ### 📊 Estado del Sistema:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Verificar .env
        env_exists = os.path.exists(".env")
        if env_exists:
            st.success("✅ Archivo .env existe")
        else:
            st.error("❌ Falta archivo .env")
    
    with col2:
        # Verificar archivos en uploads
        try:
            if os.path.exists("data/uploads"):
                files = list(os.scandir("data/uploads"))
                num_files = len([f for f in files if f.is_file()])
                if num_files > 0:
                    st.success(f"✅ {num_files} archivo(s) en uploads")
                else:
                    st.error("❌ No hay archivos en uploads")
            else:
                st.error("❌ No hay archivos en uploads")
        except:
            st.error("❌ No hay archivos en uploads")
    
    with col3:
        # Verificar VectorStore
        try:
            if os.path.exists(config.CHROMA_DIR):
                files = list(os.scandir(config.CHROMA_DIR))
                if len(files) > 0:
                    st.success("✅ VectorStore procesado")
                else:
                    st.error("❌ VectorStore no procesado")
            else:
                st.error("❌ VectorStore no procesado")
        except:
            st.error("❌ VectorStore no procesado")
    
    st.markdown("---")
    
    # Botón para ir a configuración
    st.info("👉 **Haz clic en 'Configuración' en la barra lateral para comenzar**")
    
    st.stop()

# Inicializar orquestador en session_state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MultiAgentOrchestrator(vector_store)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

if "session_started" not in st.session_state:
    st.session_state.session_started = False

# ========================================
# SIDEBAR - GESTIÓN DE CONVERSACIONES
# ========================================
with st.sidebar:
    st.header("💬 Conversaciones")

    # Botón para nueva conversación
    if st.button("➕ Nueva Conversación", use_container_width=True, type="primary"):
        # Guardar conversación actual si existe
        if st.session_state.messages:
            conversation = {
                "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "title": st.session_state.messages[0]["content"][:30] + "..." if st.session_state.messages else "Nueva conversación",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "messages": st.session_state.messages.copy()
            }
            st.session_state.conversations.append(conversation)

        # Iniciar nueva conversación
        st.session_state.messages = []
        st.session_state.current_conversation_id = None
        st.session_state.session_started = False
        st.session_state.orchestrator.reset()
        st.rerun()

    st.divider()

    # Mostrar conversaciones antiguas
    if st.session_state.conversations:
        st.subheader("📚 Historial")

        for idx, conv in enumerate(reversed(st.session_state.conversations)):
            col1, col2 = st.columns([4, 1])

            with col1:
                # Botón para cargar conversación
                if st.button(
                    f"💬 {conv['title'][:25]}...",
                    key=f"conv_{idx}",
                    use_container_width=True
                ):
                    st.session_state.messages = conv["messages"].copy()
                    st.session_state.current_conversation_id = conv["id"]
                    st.rerun()

            with col2:
                # Botón para eliminar conversación
                if st.button("🗑️", key=f"del_{idx}"):
                    st.session_state.conversations.remove(conv)
                    st.rerun()

            # Mostrar timestamp
            st.caption(f"🕐 {conv['timestamp']}")
            st.divider()
    else:
        st.info("No hay conversaciones guardadas")

    # Estadísticas
    st.divider()
    st.subheader("📊 Estadísticas")
    st.metric("Conversaciones guardadas", len(st.session_state.conversations))
    st.metric("Mensajes en esta conversación", len(st.session_state.messages))

# ========================================
# TÍTULO
# ========================================
st.title("🤖 Chat con Agente AURA")

# Mostrar estado del sistema
status_col1, status_col2 = st.columns([3, 1])
with status_col1:
    if st.session_state.session_started:
        st.caption(f"🟢 Sesión activa | Estado: {st.session_state.orchestrator.get_state()}")
    else:
        st.caption("🟡 Listo para comenzar")
with status_col2:
    if st.button("🔄 Reiniciar", help="Reiniciar el flujo de recomendaciones"):
        st.session_state.orchestrator.reset()
        st.session_state.session_started = False
        st.session_state.messages = []
        st.rerun()

st.divider()

# ========================================
# INICIAR SESIÓN SI ES LA PRIMERA VEZ
# ========================================
if not st.session_state.session_started and not st.session_state.messages:
    with st.chat_message("assistant"):
        initial_message = st.session_state.orchestrator.start_session()
        st.markdown(initial_message)
    
    st.session_state.messages.append({"role": "assistant", "content": initial_message})
    st.session_state.session_started = True

# ========================================
# MOSTRAR MENSAJES DEL HISTORIAL
# ========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ========================================
# ACEPTAR INPUT DEL USUARIO
# ========================================
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Mostrar mensaje del usuario en el contenedor
    with st.chat_message("user"):
        st.markdown(prompt)

    # Procesar con el orquestador
    with st.chat_message("assistant"):
        with st.spinner("🤔 Pensando..."):
            # Procesar la entrada del usuario
            result = st.session_state.orchestrator.process_user_input(prompt)
            
            # Obtener la respuesta
            response_text = result.get("message", "Lo siento, hubo un error.")
            
            # Mostrar respuesta con streaming
            response = st.write_stream(response_generator(response_text))
            
            # Mostrar información adicional si está disponible
            if result.get("status") == "completed":
                # Mostrar productos encontrados
                if "products_found" in result:
                    st.info(f"📦 Se encontraron {result['products_found']} productos relevantes")
                
                # Opción para ver análisis detallado
                with st.expander("📊 Ver análisis detallado"):
                    if st.session_state.orchestrator.workflow_data.get('user_analysis'):
                        st.write("**Análisis de tus necesidades:**")
                        st.write(st.session_state.orchestrator.workflow_data['user_analysis'])
                    
                    if st.session_state.orchestrator.workflow_data.get('criteria'):
                        st.write("\n**Criterios de búsqueda:**")
                        st.write(st.session_state.orchestrator.workflow_data['criteria'])
            
            elif result.get("status") == "collecting":
                # Mostrar progreso
                if "progress" in result:
                    st.caption(f"📋 Progreso: {result['progress']}")
            
            elif result.get("status") == "error":
                st.error("❌ Ocurrió un error. Por favor, intenta de nuevo.")

    # Agregar respuesta del asistente al historial
    st.session_state.messages.append({"role": "assistant", "content": response})
