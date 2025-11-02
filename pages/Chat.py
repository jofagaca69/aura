"""
Chat con Agente de IA - Sistema AURA
Basado en la guía oficial de Streamlit v2
"""

import streamlit as st
import random
import time
from datetime import datetime

# ========================================
# GENERADOR DE RESPUESTAS CON STREAMING
# ========================================
def response_generator():
    """Emulador de respuestas con streaming"""
    response = random.choice(
        [
            "¡Hola! ¿Cómo puedo ayudarte hoy con tus consultas sobre productos?",
            "¡Hola! Soy el asistente de AURA. ¿Hay algo en lo que pueda ayudarte?",
            "¿Necesitas ayuda? Estoy aquí para recomendarte productos.",
            "Bienvenido al sistema AURA. ¿Qué estás buscando hoy?",
            "¡Hola! ¿Te gustaría que te ayude a encontrar el producto perfecto?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# ========================================
# CONFIGURACIÓN DE LA PÁGINA
# ========================================
st.set_page_config(
    page_title="Chat AURA",
    page_icon="🤖",
    layout="centered"
)

# ========================================
# INICIALIZAR SESSION STATE
# ========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

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

    # Mostrar respuesta del asistente en el contenedor
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())

    # Agregar respuesta del asistente al historial
    st.session_state.messages.append({"role": "assistant", "content": response})
