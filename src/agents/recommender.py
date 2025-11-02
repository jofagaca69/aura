"""
Agente recomendador con RAG
"""
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate

from src.agents.base_agent import BaseAgent
from src.rag.vector_store import VectorStore


class RecommenderAgent(BaseAgent):
    """
    Agente que realiza recomendaciones de productos
    utilizando RAG para buscar en la base de datos
    """
    
    def __init__(self, vector_store: VectorStore):
        super().__init__(
            name="Agente Recomendador",
            role="Generar recomendaciones personalizadas de productos"
        )
        self.vector_store = vector_store
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera recomendaciones basadas en las preferencias del usuario
        
        Args:
            input_data: Debe contener 'search_query' y 'criteria'
            
        Returns:
            Recomendaciones de productos
        """
        search_query = input_data.get('search_query', '')
        criteria = input_data.get('criteria', '')
        user_analysis = input_data.get('user_analysis', '')
        
        if not search_query:
            raise ValueError("Se requiere 'search_query' del analizador de preferencias")
        
        # Buscar productos relevantes en el vectorstore
        relevant_products = self.vector_store.search_with_scores(
            search_query,
            k=10  # Buscamos más productos para tener opciones
        )
        
        # Formatear productos encontrados
        products_context = self._format_products(relevant_products)
        
        # Generar recomendaciones personalizadas
        recommendations = self._generate_recommendations(
            products_context,
            user_analysis,
            criteria
        )
        
        # Guardar en memoria
        self.update_memory("relevant_products", relevant_products)
        self.update_memory("recommendations", recommendations)
        
        return {
            "agent": self.name,
            "recommendations": recommendations,
            "products_found": len(relevant_products),
            "status": "completed"
        }
    
    def _format_products(self, products_with_scores: List[tuple]) -> str:
        """
        Formatea los productos encontrados para el contexto
        
        Args:
            products_with_scores: Lista de tuplas (documento, score)
            
        Returns:
            Productos formateados como texto
        """
        formatted = []
        
        for i, (doc, score) in enumerate(products_with_scores, 1):
            formatted.append(f"--- Producto {i} (Relevancia: {1-score:.2f}) ---")
            formatted.append(doc.page_content)
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _generate_recommendations(
        self,
        products_context: str,
        user_analysis: str,
        criteria: str
    ) -> str:
        """
        Genera recomendaciones personalizadas
        
        Args:
            products_context: Productos encontrados
            user_analysis: Análisis del usuario
            criteria: Criterios de búsqueda
            
        Returns:
            Recomendaciones en texto
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un experto asesor de productos con años de experiencia.
            
            Tu tarea es analizar los productos disponibles y las necesidades del usuario,
            y generar recomendaciones personalizadas y detalladas.
            
            Para cada producto recomendado, incluye:
            1. Nombre y características principales
            2. Por qué es adecuado para este usuario específico
            3. Ventajas y posibles limitaciones
            4. Relación calidad-precio
            
            Ordena las recomendaciones por relevancia (mejor opción primero).
            Recomienda entre 3 y 5 productos.
            
            Sé específico, honesto y útil. Si un producto no es perfecto, menciona las alternativas."""),
            ("user", """INFORMACIÓN DEL USUARIO:
{user_analysis}

CRITERIOS DE BÚSQUEDA:
{criteria}

PRODUCTOS DISPONIBLES:
{products_context}

Por favor, genera tus recomendaciones personalizadas:""")
        ])
        
        chain = prompt | self.llm
        
        result = chain.invoke({
            "user_analysis": user_analysis,
            "criteria": criteria,
            "products_context": products_context
        })
        
        return result.content
    
    def get_detailed_comparison(self, product_names: List[str]) -> str:
        """
        Genera una comparación detallada entre productos específicos
        
        Args:
            product_names: Lista de nombres de productos a comparar
            
        Returns:
            Comparación detallada
        """
        # Buscar información específica de cada producto
        comparisons = []
        
        for name in product_names:
            products = self.vector_store.search(name, k=2)
            if products:
                comparisons.append(products[0].page_content)
        
        if not comparisons:
            return "No se encontraron los productos especificados."
        
        # Generar comparación
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Genera una comparación detallada entre los siguientes productos.
            Incluye una tabla comparativa con características clave y un análisis de cuál
            es mejor para diferentes tipos de usuarios."""),
            ("user", "Productos a comparar:\n\n{products}")
        ])
        
        chain = prompt | self.llm
        result = chain.invoke({"products": "\n\n---\n\n".join(comparisons)})
        
        return result.content

