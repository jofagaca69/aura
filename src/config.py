"""
Configuración del sistema AURA
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuración centralizada del sistema"""

    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Modelo
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))  # Reducido para respuestas más rápidas y consistentes

    # RAG - Optimizado para velocidad
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))  # Reducido de 1000 para chunks más manejables
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # Reducido de 200
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))  # Reducido de 5 para búsquedas más rápidas

    # LangSmith - Monitoring y Trazabilidad
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")

    # Directorios
    DATA_DIR = "data"
    PRODUCTS_DIR = os.path.join(DATA_DIR, "products")
    CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

    @classmethod
    def validate(cls):
        """Valida que la configuración esté completa"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY no está configurada. "
                "Por favor, crea un archivo .env basado en .env.example"
            )

    @classmethod
    def setup_langsmith(cls):
        """Configura las variables de entorno para LangSmith"""
        if cls.LANGCHAIN_TRACING_V2:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if cls.LANGCHAIN_API_KEY:
                os.environ["LANGCHAIN_API_KEY"] = cls.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = cls.LANGCHAIN_PROJECT
            os.environ["LANGCHAIN_ENDPOINT"] = cls.LANGCHAIN_ENDPOINT
            print("✓ LangSmith tracing habilitado")
            print(f"  Proyecto: {cls.LANGCHAIN_PROJECT}")
        else:
            print("ℹ️  LangSmith tracing deshabilitado")


config = Config()

