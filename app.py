"""
Aplicación principal Streamlit con múltiples páginas
Página de inicio con navegación y bienvenida
"""
import streamlit as st
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Multi-Page Streamlit App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado con Tailwind-inspired styles
st.markdown("""
<style>
    /* Estilos generales */
    .main {
        padding: 2rem;
    }

    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-align: center;
    }

    .hero-subtitle {
        font-size: 1.25rem;
        text-align: center;
        opacity: 0.95;
    }

    /* Cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        border-color: #667eea;
    }

    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }

    .feature-description {
        color: #6b7280;
        line-height: 1.6;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f9fafb;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">🚀 Bienvenido a AURA</h1>
    <p class="hero-subtitle">
        Una AI para tus recomendaciones
    </p>
</div>
""", unsafe_allow_html=True)

# Descripción de la aplicación
st.markdown("## 👋 ¿Qué es esta aplicación?")
st.markdown("""
Esta es una aplicación de demostración que muestra cómo crear aplicaciones 
multi-página en **Streamlit**. Incluye ejemplos de:
- Páginas interactivas con diferentes componentes
- Sistema de chatbot con historial
- Gestión de archivos con upload y descarga
- Diseño moderno con estilos personalizados
""")

# Features en columnas
st.markdown("## ✨ Características")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">👋</div>
        <div class="feature-title">Saludo Interactivo</div>
        <div class="feature-description">
            Página con formularios y widgets interactivos para personalizar tu experiencia.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📝</div>
        <div class="feature-title">Lorem Ipsum</div>
        <div class="feature-description">
            Ejemplos de componentes UI: texto, tablas, gráficos y más elementos visuales.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">💬</div>
        <div class="feature-title">Chatbot</div>
        <div class="feature-description">
            Sistema de chat interactivo con historial y respuestas personalizadas.
        </div>
    </div>
    """, unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📁</div>
        <div class="feature-title">Gestión de Archivos</div>
        <div class="feature-description">
            Upload, visualización, descarga y administración de archivos múltiples.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎨</div>
        <div class="feature-title">Diseño Moderno</div>
        <div class="feature-description">
            Interfaz atractiva con estilos personalizados inspirados en Tailwind CSS.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Rápido y Reactivo</div>
        <div class="feature-description">
            Navegación fluida entre páginas con estado persistente y carga rápida.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sección de estadísticas
st.markdown("## 📊 Estadísticas de la Aplicación")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Páginas Disponibles",
        value="5",
        delta="Múltiples ejemplos"
    )

with col2:
    st.metric(
        label="Componentes UI",
        value="20+",
        delta="Totalmente interactivos"
    )

with col3:
    # Contar archivos en uploads si existe
    uploads_dir = Path("data/uploads")
    file_count = len(list(uploads_dir.glob("*"))) if uploads_dir.exists() else 0
    st.metric(
        label="Archivos Subidos",
        value=file_count,
        delta="En gestión"
    )

with col4:
    st.metric(
        label="Estado",
        value="Activo",
        delta="100% funcional"
    )

# Instrucciones
st.markdown("## 🧭 Navegación")
st.info("""
**👈 Usa la barra lateral** para navegar entre las diferentes páginas de la aplicación.

Cada página demuestra diferentes capacidades de Streamlit y patrones de diseño útiles.
""")

# Quick Start Guide
with st.expander("📖 Guía Rápida de Uso"):
    st.markdown("""
    ### Cómo usar esta aplicación:

    1. **Navegación**: Usa el menú de la barra lateral para cambiar de página
    2. **Saludo**: Página interactiva con formularios y personalización
    3. **Lorem Ipsum**: Ejemplos de componentes y layouts
    4. **Chatbot**: Chatea con el asistente virtual
    5. **Gestión de Archivos**: Sube, visualiza y gestiona archivos

    ### Tecnologías utilizadas:
    - **Streamlit**: Framework principal
    - **Python**: Lenguaje de programación
    - **CSS personalizado**: Estilos modernos
    - **Session State**: Manejo de estado persistente
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <p style='font-size: 0.875rem;'>
        💡 José Fabián García Camargo.
    </p>
    <p style='font-size: 0.75rem; margin-top: 1rem;'>
        Creado con ❤️ usando Streamlit, LangChain y LangGraph | 2025
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar adicional
with st.sidebar:
    st.markdown("### 📌 Acerca de")
    st.info("""
    Esta es una aplicación de demostración que muestra las capacidades 
    de Streamlit para crear aplicaciones web interactivas con múltiples páginas.
    """)

    st.markdown("### 🔗 Enlaces Útiles")
    st.markdown("""
    - [Documentación Streamlit](https://docs.streamlit.io)
    - [Galería de Apps](https://streamlit.io/gallery)
    - [Foro de la Comunidad](https://discuss.streamlit.io)
    """)
