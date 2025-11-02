"""
Orquestador del sistema multiagentes
"""
from typing import Dict, Any, Optional
from enum import Enum

from src.agents.questioner import QuestionerAgent
from src.agents.preference_analyzer import PreferenceAnalyzerAgent
from src.agents.recommender import RecommenderAgent
from src.rag.vector_store import VectorStore


class WorkflowState(Enum):
    """Estados del flujo de trabajo"""
    INIT = "init"
    COLLECTING_INFO = "collecting_info"
    ANALYZING_PREFERENCES = "analyzing_preferences"
    GENERATING_RECOMMENDATIONS = "generating_recommendations"
    COMPLETED = "completed"


class MultiAgentOrchestrator:
    """
    Orquestador que coordina el flujo de trabajo entre múltiples agentes
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Inicializa el orquestador con todos los agentes
        
        Args:
            vector_store: VectorStore con los productos indexados
        """
        self.vector_store = vector_store
        
        # Inicializar agentes (usando QuestionerAgent para preguntas dinámicas)
        self.questioner = QuestionerAgent()
        self.analyzer = PreferenceAnalyzerAgent()
        self.recommender = RecommenderAgent(vector_store)
        
        # Estado del flujo
        self.state = WorkflowState.INIT
        self.workflow_data: Dict[str, Any] = {}
    
    def start_session(self) -> str:
        """
        Inicia una nueva sesión de recomendación
        
        Returns:
            Primera pregunta para el usuario
        """
        # Reiniciar agentes
        self.questioner.reset()
        self.analyzer.clear_memory()
        self.recommender.clear_memory()
        
        # Reiniciar estado
        self.state = WorkflowState.COLLECTING_INFO
        self.workflow_data = {}
        
        # Obtener primera pregunta (generada dinámicamente por el LLM)
        question = self.questioner.generate_next_question()
        
        return question if question else "¡Hola! 👋 ¿Qué tipo de producto estás buscando hoy?"
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Procesa la entrada del usuario según el estado actual
        
        Args:
            user_input: Respuesta del usuario
            
        Returns:
            Respuesta del sistema con siguiente acción
        """
        if self.state == WorkflowState.COLLECTING_INFO:
            return self._handle_collection(user_input)
        
        elif self.state == WorkflowState.COMPLETED:
            # Permitir preguntas adicionales sobre las recomendaciones
            return self._handle_followup_question(user_input)
        
        else:
            return {
                "message": "Estado inválido del sistema. Por favor, reinicia la sesión.",
                "status": "error"
            }
    
    def _handle_collection(self, user_input: str) -> Dict[str, Any]:
        """
        Maneja la recolección de información con preguntas contextuales dinámicas
        
        Args:
            user_input: Respuesta del usuario
            
        Returns:
            Siguiente pregunta o inicio del análisis
        """
        # Guardar respuesta y actualizar contexto
        self.questioner.add_user_response(user_input)
        
        # Generar siguiente pregunta contextual (el LLM decide basado en las respuestas previas)
        next_question = self.questioner.generate_next_question()
        
        # Si hay siguiente pregunta, devolverla
        if next_question:
            return {
                "message": next_question,
                "status": "collecting",
                "progress": self.questioner.get_progress()
            }
        
        # No hay más preguntas, procesar información
        return self._process_workflow()
    
    def _process_workflow(self) -> Dict[str, Any]:
        """
        Ejecuta el flujo completo de análisis y recomendación
        
        Returns:
            Recomendaciones finales
        """
        try:
            # Paso 1: Analizar respuestas del usuario (usando QuestionerAgent)
            print("\n🔍 Analizando tus respuestas...")
            self.state = WorkflowState.ANALYZING_PREFERENCES
            
            questioner_result = self.questioner.process({})
            self.workflow_data['user_analysis'] = questioner_result['structured_analysis']
            
            # Paso 2: Generar criterios de búsqueda
            print("📊 Generando criterios de búsqueda...")
            
            analyzer_result = self.analyzer.process({
                'user_analysis': self.workflow_data['user_analysis']
            })
            self.workflow_data['criteria'] = analyzer_result['criteria']
            self.workflow_data['search_query'] = analyzer_result['search_query']
            
            # Paso 3: Generar recomendaciones
            print("🎯 Buscando los mejores productos para ti...")
            self.state = WorkflowState.GENERATING_RECOMMENDATIONS
            
            recommender_result = self.recommender.process({
                'search_query': self.workflow_data['search_query'],
                'criteria': self.workflow_data['criteria'],
                'user_analysis': self.workflow_data['user_analysis']
            })
            
            self.workflow_data['recommendations'] = recommender_result['recommendations']
            self.workflow_data['products_found'] = recommender_result['products_found']
            
            # Completado
            self.state = WorkflowState.COMPLETED
            
            return {
                "message": self._format_final_response(),
                "status": "completed",
                "recommendations": self.workflow_data['recommendations'],
                "products_found": self.workflow_data['products_found']
            }
            
        except Exception as e:
            return {
                "message": f"Error procesando la información: {str(e)}",
                "status": "error"
            }
    
    def _format_final_response(self) -> str:
        """
        Formatea la respuesta final con las recomendaciones
        
        Returns:
            Respuesta formateada
        """
        response = f"""
✨ ¡Análisis completado! He encontrado {self.workflow_data['products_found']} productos relevantes.

📋 RECOMENDACIONES PERSONALIZADAS:

{self.workflow_data['recommendations']}

---

💬 ¿Tienes alguna pregunta sobre estas recomendaciones?
Puedo ayudarte con:
- Comparar productos específicos
- Explicar más sobre alguna característica
- Buscar alternativas
- Cualquier otra duda
"""
        return response
    
    def _handle_followup_question(self, user_input: str) -> Dict[str, Any]:
        """
        Maneja preguntas de seguimiento sobre las recomendaciones
        
        Args:
            user_input: Pregunta del usuario
            
        Returns:
            Respuesta a la pregunta
        """
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres AURA, un asistente experto en productos.
            Ya has generado recomendaciones para el usuario.
            Ahora responde sus preguntas adicionales basándote en:
            1. Las recomendaciones que ya hiciste
            2. El análisis del usuario
            3. Los productos disponibles
            
            Sé útil, específico y amigable."""),
            ("user", """CONTEXTO:

Análisis del usuario:
{user_analysis}

Recomendaciones previas:
{recommendations}

PREGUNTA DEL USUARIO:
{question}

Por favor, responde la pregunta:""")
        ])
        
        chain = prompt | self.recommender.llm
        
        result = chain.invoke({
            "user_analysis": self.workflow_data.get('user_analysis', ''),
            "recommendations": self.workflow_data.get('recommendations', ''),
            "question": user_input
        })
        
        return {
            "message": result.content,
            "status": "followup"
        }
    
    def get_state(self) -> str:
        """Obtiene el estado actual del flujo"""
        return self.state.value
    
    def reset(self):
        """Reinicia el orquestador"""
        self.questioner.reset()
        self.analyzer.clear_memory()
        self.recommender.clear_memory()
        self.state = WorkflowState.INIT
        self.workflow_data = {}

