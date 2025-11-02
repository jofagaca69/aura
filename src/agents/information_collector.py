"""
Agente recolector de información del usuario
"""
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent


class UserPreferences(BaseModel):
    """Modelo de preferencias del usuario"""
    budget: str = Field(description="Presupuesto del usuario")
    category: str = Field(description="Categoría de producto de interés")
    features: List[str] = Field(description="Características deseadas")
    priorities: List[str] = Field(description="Prioridades del usuario")
    additional_info: str = Field(description="Información adicional relevante")


class InformationCollectorAgent(BaseAgent):
    """
    Agente encargado de recolectar información del usuario
    mediante preguntas estratégicas
    """
    
    def __init__(self):
        super().__init__(
            name="Recolector de Información",
            role="Recopilar preferencias y necesidades del usuario"
        )
        
        self.questions = [
            "¿Cuál es tu presupuesto aproximado para esta compra?",
            "¿Qué tipo de producto estás buscando? (categoría)",
            "¿Qué características son más importantes para ti?",
            "¿Tienes alguna marca o especificación preferida?",
            "¿Para qué uso principal necesitas este producto?",
        ]
        
        self.current_question_index = 0
        self.user_responses = []
    
    def get_next_question(self) -> str:
        """
        Obtiene la siguiente pregunta para el usuario
        
        Returns:
            Pregunta a realizar
        """
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            self.current_question_index += 1
            return question
        return None
    
    def add_response(self, response: str):
        """
        Añade una respuesta del usuario
        
        Args:
            response: Respuesta del usuario
        """
        self.user_responses.append(response)
    
    def has_more_questions(self) -> bool:
        """
        Verifica si hay más preguntas por hacer
        
        Returns:
            True si hay más preguntas
        """
        return self.current_question_index < len(self.questions)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa las respuestas del usuario y extrae información estructurada
        
        Args:
            input_data: Debe contener 'responses' con las respuestas del usuario
            
        Returns:
            Información estructurada del usuario
        """
        responses = input_data.get('responses', self.user_responses)
        
        # Crear prompt para analizar las respuestas
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un asistente experto en analizar las necesidades de los clientes.
            Analiza las siguientes respuestas del usuario y extrae información estructurada sobre:
            - Presupuesto
            - Categoría de producto
            - Características deseadas
            - Prioridades
            - Información adicional relevante
            
            Sé específico y detallado en tu análisis."""),
            ("user", "Respuestas del usuario:\n{responses}\n\nPor favor, analiza y estructura esta información.")
        ])
        
        # Crear chain
        chain = prompt | self.llm
        
        # Procesar
        responses_text = "\n".join([
            f"Pregunta {i+1}: {self.questions[i]}\nRespuesta: {resp}"
            for i, resp in enumerate(responses)
        ])
        
        result = chain.invoke({"responses": responses_text})
        
        # Guardar en memoria
        self.update_memory("raw_responses", responses)
        self.update_memory("analysis", result.content)
        
        return {
            "agent": self.name,
            "raw_responses": responses,
            "analysis": result.content,
            "status": "completed"
        }
    
    def reset(self):
        """Reinicia el agente para una nueva sesión"""
        self.current_question_index = 0
        self.user_responses = []
        self.clear_memory()

