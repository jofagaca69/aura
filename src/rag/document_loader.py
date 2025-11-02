"""
Cargador de documentos para diferentes tipos de archivos
"""
import os
from typing import List
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)
from langchain_core.documents import Document
import json
import pandas as pd

# Dependencias opcionales
try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    WORD_SUPPORT = True
except ImportError:
    WORD_SUPPORT = False

try:
    from langchain_community.document_loaders import UnstructuredExcelLoader
    EXCEL_UNSTRUCTURED_SUPPORT = True
except ImportError:
    EXCEL_UNSTRUCTURED_SUPPORT = False


class DocumentLoader:
    """Cargador universal de documentos de productos"""

    def __init__(self):
        self.supported_extensions = {
            '.pdf': self._load_pdf,
            '.txt': self._load_text,
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.docx': self._load_docx,
            '.doc': self._load_docx,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
        }

    def load_documents(self, directory: str) -> List[Document]:
        """
        Carga todos los documentos de un directorio

        Args:
            directory: Ruta al directorio con los archivos

        Returns:
            Lista de documentos cargados
        """
        documents = []
        directory_path = Path(directory)

        if not directory_path.exists():
            raise ValueError(f"El directorio {directory} no existe")

        for file_path in directory_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in self.supported_extensions:
                    try:
                        docs = self.supported_extensions[ext](str(file_path))
                        documents.extend(docs)
                        print(f"✓ Cargado: {file_path.name}")
                    except Exception as e:
                        print(f"✗ Error cargando {file_path.name}: {e}")

        return documents

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Carga archivos PDF"""
        loader = PyPDFLoader(file_path)
        return loader.load()

    def _load_text(self, file_path: str) -> List[Document]:
        """Carga archivos de texto"""
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    def _load_csv(self, file_path: str) -> List[Document]:
        """Carga archivos CSV usando pandas (más rápido y robusto)"""
        try:
            # Usar pandas para mejor manejo de diferentes encodings
            df = pd.read_csv(file_path, encoding='utf-8')
            documents = []

            for idx, row in df.iterrows():
                # Crear contenido legible, omitiendo valores NaN
                content = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "row": idx,
                        "type": "csv",
                        "columns": list(df.columns)
                    }
                )
                documents.append(doc)

            return documents
        except UnicodeDecodeError:
            # Intentar con latin1 si UTF-8 falla
            try:
                df = pd.read_csv(file_path, encoding='latin1')
                documents = []
                for idx, row in df.iterrows():
                    content = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path, "row": idx, "type": "csv"}
                    )
                    documents.append(doc)
                return documents
            except Exception as e:
                raise ValueError(f"Error cargando CSV {file_path}: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Error cargando CSV {file_path}: {str(e)}") from e

    def _load_json(self, file_path: str) -> List[Document]:
        """Carga archivos JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []

        # Si es una lista de productos
        if isinstance(data, list):
            for idx, item in enumerate(data):
                content = json.dumps(item, indent=2, ensure_ascii=False)
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path, "index": idx}
                )
                documents.append(doc)
        # Si es un objeto único
        else:
            content = json.dumps(data, indent=2, ensure_ascii=False)
            doc = Document(
                page_content=content,
                metadata={"source": file_path}
            )
            documents.append(doc)

        return documents

    def _load_docx(self, file_path: str) -> List[Document]:
        """Carga archivos Word (.docx, .doc)"""
        if not WORD_SUPPORT:
            raise ImportError(
                "Para cargar archivos Word, necesitas instalar: pip install unstructured\n"
                "Alternativamente, convierte el archivo a PDF o TXT"
            )

        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()
        except Exception as e:
            raise ValueError(f"Error cargando archivo Word {file_path}: {str(e)}") from e

    def _load_excel(self, file_path: str) -> List[Document]:
        """Carga archivos Excel usando pandas (más rápido y sin dependencias extras)
        Procesa todas las hojas del archivo Excel"""
        try:
            # Leer todas las hojas del archivo Excel
            # sheet_name=None devuelve un diccionario {nombre_hoja: DataFrame}
            all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
            documents = []

            # Procesar cada hoja
            for sheet_name, df in all_sheets.items():
                for idx, row in df.iterrows():
                    # Crear contenido legible para cada fila
                    content = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "sheet": sheet_name,
                            "row": idx,
                            "type": "excel",
                            "columns": list(df.columns)
                        }
                    )
                    documents.append(doc)

            return documents
        except ImportError as e:
            raise ImportError(
                f"Para cargar archivos Excel, necesitas instalar openpyxl: pip install openpyxl"
            ) from e
        except Exception as e:
            raise ValueError(f"Error cargando archivo Excel {file_path}: {str(e)}") from e

