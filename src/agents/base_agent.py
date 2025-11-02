"""
Clase base para todos los agentes del sistema
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import config


class BaseAgent(ABC):
    """Clase base abstracta para agentes"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.llm = ChatGoogleGenerativeAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY
        )
        self.memory: Dict[str, Any] = {}
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa la entrada y devuelve un resultado
        
        Args:
            input_data: Datos de entrada para el agente
            
        Returns:
            Resultado del procesamiento
        """
        pass
    
    def update_memory(self, key: str, value: Any):
        """Actualiza la memoria del agente"""
        self.memory[key] = value
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de la memoria"""
        return self.memory.get(key, default)
    
    def clear_memory(self):
        """Limpia la memoria del agente"""
        self.memory.clear()
    
    def __str__(self):
        return f"{self.name} ({self.role})"

