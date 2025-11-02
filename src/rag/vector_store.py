"""
Sistema de almacenamiento vectorial con ChromaDB
"""
from typing import List, Optional, Any, Dict
import os
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import config


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Limpia metadata para que sea compatible con ChromaDB.
    ChromaDB solo acepta: str, int, float, bool, None

    Args:
        metadata: Diccionario de metadata original

    Returns:
        Diccionario de metadata limpio
    """
    cleaned = {}
    for key, value in metadata.items():
        # Si es None, mantenerlo
        if value is None:
            cleaned[key] = value
        # Si es tipo básico, mantenerlo
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        # Si es lista, convertir a string
        elif isinstance(value, list):
            cleaned[key] = ", ".join(str(item) for item in value)
        # Si es dict, convertir a JSON string
        elif isinstance(value, dict):
            cleaned[key] = json.dumps(value, ensure_ascii=False)
        # Para cualquier otro tipo, convertir a string
        else:
            cleaned[key] = str(value)

    return cleaned


class VectorStore:
    """Gestor del almacenamiento vectorial para RAG"""

    def __init__(self):
        # Usar embeddings locales multilingües optimizados con caché
        print("🔧 Inicializando modelo de embeddings local...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder="./data/model_cache"
        )
        print("✓ Modelo de embeddings listo")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.vectorstore: Optional[Chroma] = None
        self._search_cache = {}  # Caché de búsquedas

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Crea un vectorstore a partir de documentos

        Args:
            documents: Lista de documentos a indexar

        Returns:
            Vectorstore de Chroma
        """
        # Limpiar metadata de todos los documentos ANTES de procesar
        print(f"🧹 Limpiando metadata de {len(documents)} documentos...")
        for doc in documents:
            doc.metadata = clean_metadata(doc.metadata)

        # Dividir documentos en chunks
        splits = self.text_splitter.split_documents(documents)

        print(f"📄 Documentos divididos en {len(splits)} chunks")

        # Limpiar metadata de los splits también (por si el splitter agrega metadata)
        print(f"🧹 Limpiando metadata de {len(splits)} chunks...")
        for split in splits:
            split.metadata = clean_metadata(split.metadata)

        # Crear directorio si no existe
        os.makedirs(config.CHROMA_DIR, exist_ok=True)

        # Crear vectorstore
        print("💾 Creando vectorstore en ChromaDB...")
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=config.CHROMA_DIR
        )

        print(f"✓ Vectorstore creado con {len(splits)} embeddings")

        return self.vectorstore

    def load_vectorstore(self) -> Chroma:
        """
        Carga un vectorstore existente

        Returns:
            Vectorstore de Chroma
        """
        if not os.path.exists(config.CHROMA_DIR):
            raise ValueError(
                f"No existe vectorstore en {config.CHROMA_DIR}. "
                "Primero debes crear uno con create_vectorstore()"
            )

        self.vectorstore = Chroma(
            persist_directory=config.CHROMA_DIR,
            embedding_function=self.embeddings
        )

        print(f"✓ Vectorstore cargado desde {config.CHROMA_DIR}")

        return self.vectorstore

    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Busca documentos similares a la consulta

        Args:
            query: Consulta de búsqueda
            k: Número de resultados a devolver

        Returns:
            Lista de documentos relevantes
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore no inicializado")

        k = k or config.TOP_K_RESULTS

        results = self.vectorstore.similarity_search(query, k=k)

        return results

    def search_with_scores(self, query: str, k: int = None) -> List[tuple]:
        """
        Busca documentos con scores de similitud (con caché)

        Args:
            query: Consulta de búsqueda
            k: Número de resultados a devolver

        Returns:
            Lista de tuplas (documento, score)
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore no inicializado")

        k = k or config.TOP_K_RESULTS

        # Verificar caché (útil para consultas similares)
        cache_key = f"{query[:100]}_{k}"  # Limitar tamaño de clave
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Guardar en caché (máximo 50 búsquedas)
        if len(self._search_cache) >= 50:
            # Eliminar entrada más antigua
            self._search_cache.pop(next(iter(self._search_cache)))
        self._search_cache[cache_key] = results

        return results

    def get_retriever(self, k: int = None):
        """
        Obtiene un retriever para usar con chains

        Args:
            k: Número de resultados a devolver

        Returns:
            Retriever de LangChain
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore no inicializado")

        k = k or config.TOP_K_RESULTS

        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

