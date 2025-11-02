"""
Agente analizador de preferencias del usuario
"""
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate

from src.agents.base_agent import BaseAgent


class PreferenceAnalyzerAgent(BaseAgent):
    """
    Agente que analiza en profundidad las preferencias del usuario
    y genera criterios de búsqueda optimizados
    """
    
    def __init__(self):
        super().__init__(
            name="Analizador de Preferencias",
            role="Analizar y priorizar las preferencias del usuario"
        )
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza las preferencias del usuario y genera criterios de búsqueda
        
        Args:
            input_data: Debe contener 'user_analysis' del agente anterior
            
        Returns:
            Criterios de búsqueda estructurados
        """
        user_analysis = input_data.get('user_analysis', '')
        
        if not user_analysis:
            raise ValueError("Se requiere 'user_analysis' del agente recolector")
        
        # Crear prompt para análisis profundo
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un experto en análisis de preferencias de clientes y recomendaciones de productos.
            
            Tu tarea es analizar la información del usuario y generar:
            1. Lista de criterios de búsqueda prioritarios (ordenados por importancia)
            2. Palabras clave para búsqueda en la base de datos
            3. Filtros específicos (rango de precio, categorías, características)
            4. Factores de decisión del usuario (qué valora más)
            
            Sé específico y estructurado en tu análisis."""),
            ("user", """Información del usuario:
{user_analysis}

Por favor, genera criterios de búsqueda detallados y optimizados.""")
        ])
        
        # Crear chain
        chain = prompt | self.llm
        
        # Procesar
        result = chain.invoke({"user_analysis": user_analysis})
        
        # Generar query de búsqueda optimizada
        search_query = self._generate_search_query(user_analysis, result.content)
        
        # Guardar en memoria
        self.update_memory("criteria", result.content)
        self.update_memory("search_query", search_query)
        
        return {
            "agent": self.name,
            "criteria": result.content,
            "search_query": search_query,
            "status": "completed"
        }
    
    def _generate_search_query(self, user_analysis: str, criteria: str) -> str:
        """
        Genera una query de búsqueda optimizada para el RAG
        
        Args:
            user_analysis: Análisis del usuario
            criteria: Criterios generados
            
        Returns:
            Query de búsqueda
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Genera una consulta de búsqueda concisa y efectiva para encontrar productos relevantes.
            La consulta debe incluir las palabras clave más importantes y características esenciales.
            Máximo 2-3 oraciones."""),
            ("user", """Información del usuario:
{user_analysis}

Criterios:
{criteria}

Genera la consulta de búsqueda:""")
        ])
        
        chain = prompt | self.llm
        result = chain.invoke({
            "user_analysis": user_analysis,
            "criteria": criteria
        })
        
        return result.content.strip()

