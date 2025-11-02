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
        
        # Limitar a máximo 3 productos para recomendar
        products_to_recommend = relevant_products[:3]
        
        # Formatear productos encontrados
        products_context = self._format_products(products_to_recommend)
        
        # Generar recomendaciones personalizadas
        recommendations = self._generate_recommendations(
            products_context,
            user_analysis,
            criteria,
            num_products=len(products_to_recommend)
        )
        
        # Guardar en memoria
        self.update_memory("relevant_products", relevant_products)
        self.update_memory("recommendations", recommendations)
        
        return {
            "agent": self.name,
            "recommendations": recommendations,
            "products_found": len(products_to_recommend),
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
        criteria: str,
        num_products: int
    ) -> str:
        """
        Genera recomendaciones personalizadas
        
        Args:
            products_context: Productos encontrados
            user_analysis: Análisis del usuario
            criteria: Criterios de búsqueda
            num_products: Número de productos disponibles para recomendar
            
        Returns:
            Recomendaciones en texto
        """
        # Ajustar el mensaje según la cantidad de productos disponibles
        if num_products == 0:
            return "Lo siento, no encontré productos que coincidan con tus criterios. ¿Podrías darme más detalles o modificar tus preferencias?"
        
        products_instruction = f"DEBES recomendar EXACTAMENTE {num_products} producto{'s' if num_products > 1 else ''}, ni uno más ni uno menos."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un experto asesor de productos con años de experiencia.
            
            Tu tarea es analizar los productos disponibles y las necesidades del usuario,
            y generar recomendaciones personalizadas y detalladas.
            
            **FORMATO DE RESPUESTA:**
            Genera las recomendaciones usando EXACTAMENTE este formato Markdown:
            
            ### 🥇 Recomendación #1: [Nombre del Producto]
            
            **🎯 ¿Por qué es perfecto para ti?**
            [Explicación específica de por qué este producto se ajusta a las necesidades del usuario]
            
            **✨ Características principales:**
            - [Característica 1]
            - [Característica 2]
            - [Característica 3]
            
            **✅ Ventajas:**
            - [Ventaja 1]
            - [Ventaja 2]
            
            **⚠️ Consideraciones:**
            - [Limitación o consideración 1]
            - [Limitación o consideración 2]
            
            **💰 Relación calidad-precio:**  
            [Tu análisis en 1-2 líneas]
            
            ---
            
            [Si hay más productos, repite el MISMO formato exacto para las recomendaciones #2 y #3, con el separador --- entre cada una]
            
            **REGLAS CRÍTICAS:** 
            - {products_instruction}
            - NO inventes productos que no están en la lista
            - NO generes recomendaciones con "No hay más" o "N/A"
            - Solo recomienda los productos reales que se te proporcionan
            - USA SALTOS DE LÍNEA dobles entre cada sección
            - SEPARA cada recomendación con ---
            - Ordena por relevancia (mejor primero)
            - Sé específico pero CONCISO (2-3 líneas por explicación)
            - Mantén el formato EXACTO mostrado arriba"""),
            ("user", """INFORMACIÓN DEL USUARIO:
{user_analysis}

CRITERIOS DE BÚSQUEDA:
{criteria}

PRODUCTOS DISPONIBLES:
{products_context}

Por favor, genera tus recomendaciones personalizadas usando ÚNICAMENTE los productos listados arriba y siguiendo el formato Markdown especificado:""")
        ])
        
        chain = prompt | self.llm
        
        result = chain.invoke({
            "user_analysis": user_analysis,
            "criteria": criteria,
            "products_context": products_context,
            "products_instruction": products_instruction
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

