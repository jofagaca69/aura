"""
Agente preguntador interactivo para recopilar información del usuario
"""
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent


class ConversationContext(BaseModel):
    """Contexto de la conversación"""
    questions_asked: List[str] = Field(default_factory=list, description="Preguntas ya realizadas")
    user_answers: List[str] = Field(default_factory=list, description="Respuestas del usuario")
    topics_covered: List[str] = Field(default_factory=list, description="Temas ya cubiertos")
    current_question_number: int = Field(default=0, description="Número de pregunta actual")


class QuestionerAgent(BaseAgent):
    """
    Agente inteligente que hace preguntas dinámicas al usuario
    para recopilar información sobre sus necesidades.
    
    Características:
    - Máximo 5 preguntas
    - Preguntas adaptativas basadas en respuestas previas
    - Conversación natural y contextual
    - Extracción inteligente de información
    """
    
    MAX_QUESTIONS = 5
    
    def __init__(self):
        super().__init__(
            name="Agente Preguntador Interactivo",
            role="Recopilar información mediante preguntas inteligentes y adaptativas"
        )
        
        self.conversation_context = ConversationContext()
        
        # Prompt mejorado para generar preguntas ultra-personalizadas con Gemini
        self.question_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un asistente de compras experto y empático que hace preguntas INTELIGENTES 
            y PERSONALIZADAS para entender las necesidades del usuario. Tu objetivo es descubrir qué 
            producto necesita realmente y por qué.
            
            🎯 ESTRATEGIA DE PREGUNTAS:
            
            1. **ANALIZA EL CONTEXTO**: Lee cuidadosamente las respuestas previas
            2. **PROFUNDIZA**: Si el usuario mencionó algo interesante, pregunta más detalles
            3. **CONECTA IDEAS**: Relaciona la nueva pregunta con lo que ya sabes
            4. **SÉ ESPECÍFICO**: Usa la información que ya tienes para hacer preguntas más precisas
            5. **PRIORIZA**: Enfócate en lo que aún falta y es crítico
            
            📊 INFORMACIÓN CRÍTICA A OBTENER:
            - **Categoría**: ¿Qué tipo de producto? (laptop, teléfono, etc.)
            - **Presupuesto**: ¿Rango de precio aproximado?
            - **Uso principal**: ¿Para qué lo usará? (trabajo, gaming, estudio, etc.)
            - **Características clave**: ¿Qué especificaciones son importantes?
            - **Preferencias**: ¿Marcas, tamaños, colores, etc.?
            - **Restricciones**: ¿Limitaciones de tiempo, espacio, compatibilidad?
            
            💡 EJEMPLOS DE PREGUNTAS CONTEXTUALES:
            
            Ejemplo 1:
            Usuario dijo: "Necesito una laptop"
            Mal: "¿Qué tipo de producto buscas?"
            Bien: "Perfecto, ¿para qué usarás principalmente tu laptop? ¿Trabajo, estudio, gaming o entretenimiento?"
            
            Ejemplo 2:
            Usuario dijo: "Para programar"
            Mal: "¿Qué características quieres?"
            Bien: "Excelente, para programación. ¿Qué tipo de desarrollo haces? ¿Trabajas con IDEs pesados, 
            máquinas virtuales o desarrollo web principalmente?"
            
            Ejemplo 3:
            Usuario dijo: "Desarrollo web y algo de edición de video"
            Mal: "¿Cuánto quieres gastar?"
            Bien: "Interesante combinación. Para edición de video necesitarás buena potencia. 
            ¿Cuál es tu presupuesto aproximado para una máquina que maneje ambas tareas?"
            
            ⚠️ EVITA:
            - Preguntas genéricas que ignoran el contexto
            - Repetir información que ya diste
            - Preguntar lo que ya respondieron implícitamente
            - Ser robótico o formal en exceso
            
            ✅ REGLAS DE ORO:
            1. **USA lo que ya sabes**: Menciona detalles previos en tu pregunta
            2. **Una idea por pregunta**: No hagas preguntas compuestas
            3. **Conversacional**: Como si hablaras con un amigo
            4. **Empático**: Muestra que entiendes sus necesidades
            5. **Solo la pregunta**: No expliques, no des contexto extra
            
            📝 CONTEXTO ACTUAL:
            Pregunta número: {questions_count}/{max_questions}
            Temas ya cubiertos: {topics_covered}
            
            CONVERSACIÓN HASTA AHORA:
            {conversation_history}
            
            🎯 INSTRUCCIÓN: Basándote en TODO el contexto anterior, genera UNA pregunta inteligente, 
            específica y personalizada que profundice en la información más valiosa que aún falte."""),
            ("user", "Genera la siguiente pregunta personalizada:")
        ])
        
        # Prompt mejorado para analizar si necesitamos más información
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un analista experto en comprensión de necesidades de clientes.
            
            🎯 TU TAREA: Determinar si tenemos SUFICIENTE información para recomendar productos.
            
            📋 INFORMACIÓN MÍNIMA NECESARIA para una buena recomendación:
            1. **Categoría de producto** (qué tipo de producto busca)
            2. **Presupuesto** (rango de precio, aunque sea aproximado)
            3. **Uso principal** O **Características clave** (al menos uno de estos)
            
            ✅ TENEMOS SUFICIENTE SI:
            - Sabemos QUÉ busca, CUÁNTO puede gastar, y PARA QUÉ lo necesita
            - O tenemos suficiente contexto para hacer recomendaciones relevantes
            - O el usuario fue muy específico y claro en sus respuestas
            
            ⚠️ NECESITAMOS MÁS SI:
            - Falta información crítica (categoría, presupuesto o uso)
            - Las respuestas fueron muy vagas o generales
            - Hay contradicciones que necesitan clarificación
            - El usuario mencionó algo importante que no hemos profundizado
            
            📊 CONTEXTO DE LA CONVERSACIÓN:
            {conversation_history}
            
            Preguntas realizadas: {questions_count}/{max_questions}
            
            🎯 DECISIÓN REQUERIDA:
            Responde SOLO con una de estas dos palabras seguida de una breve explicación:
            - "CONTINUAR: [razón breve]" - Si necesitas información crítica adicional
            - "SUFICIENTE: [razón breve]" - Si ya puedes hacer buenas recomendaciones
            
            Sé crítico pero también eficiente. No necesitamos información perfecta, solo suficiente."""),
            ("user", "¿Tenemos suficiente información o debemos continuar preguntando?")
        ])
        
        # Prompt para generar la primera pregunta (también personalizada)
        self.initial_question_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un asistente de compras amigable y profesional.
            
            🎯 TAREA: Genera una pregunta de APERTURA cálida y efectiva para iniciar la conversación.
            
            ✅ LA PREGUNTA DEBE:
            1. Ser amigable y acogedora
            2. Preguntar qué tipo de producto busca
            3. Ser abierta pero enfocada
            4. Incluir un saludo breve
            5. Mostrar entusiasmo por ayudar
            
            💡 EJEMPLOS DE BUENAS PREGUNTAS INICIALES:
            - "¡Hola! 👋 Estoy aquí para ayudarte a encontrar el producto perfecto. ¿Qué estás buscando hoy?"
            - "¡Bienvenido! 😊 Me encantaría ayudarte. ¿Qué tipo de producto tienes en mente?"
            - "¡Hola! Soy tu asistente de compras. ¿En qué producto estás interesado hoy?"
            
            ⚠️ EVITA:
            - Ser demasiado formal o robótico
            - Hacer múltiples preguntas a la vez
            - Ser muy largo o explicativo
            
            Genera SOLO la pregunta, sin texto adicional."""),
            ("user", "Genera la pregunta de apertura:")
        ])
    
    def generate_next_question(self) -> Optional[str]:
        """
        Genera la siguiente pregunta basada en el contexto de la conversación usando Gemini
        
        Returns:
            Siguiente pregunta o None si no hay más preguntas
        """
        if self.conversation_context.current_question_number >= self.MAX_QUESTIONS:
            return None
        
        # Si es la primera pregunta, usar prompt especial de apertura
        if self.conversation_context.current_question_number == 0:
            try:
                chain = self.initial_question_prompt | self.llm
                result = chain.invoke({})
                question = result.content.strip()
                
                self.conversation_context.current_question_number += 1
                self.conversation_context.questions_asked.append(question)
                return question
            except Exception as e:
                print(f"Error generando pregunta inicial: {e}")
                # Fallback a pregunta por defecto si falla
                question = "¡Hola! 👋 ¿Qué tipo de producto estás buscando hoy?"
                self.conversation_context.current_question_number += 1
                self.conversation_context.questions_asked.append(question)
                return question
        
        # Verificar si necesitamos más información (después de 3 preguntas)
        if self.conversation_context.current_question_number >= 3:
            should_continue = self._should_continue_asking()
            if not should_continue:
                return None
        
        # Generar conversación histórica con contexto rico
        conversation_history = self._format_conversation_history()
        
        # Generar siguiente pregunta personalizada con Gemini
        try:
            chain = self.question_prompt | self.llm
            result = chain.invoke({
                "questions_count": self.conversation_context.current_question_number,
                "max_questions": self.MAX_QUESTIONS,
                "conversation_history": conversation_history,
                "topics_covered": ", ".join(self.conversation_context.topics_covered) if self.conversation_context.topics_covered else "Ninguno aún"
            })
            
            question = result.content.strip()
            
            # Limpiar la pregunta (remover comillas extras si las hay)
            question = question.strip('"').strip("'")
            
            self.conversation_context.current_question_number += 1
            self.conversation_context.questions_asked.append(question)
            
            return question
            
        except Exception as e:
            print(f"⚠️  Error generando pregunta con Gemini: {e}")
            return None
    
    def add_user_response(self, response: str):
        """
        Añade una respuesta del usuario al contexto y extrae información clave
        
        Args:
            response: Respuesta del usuario
        """
        self.conversation_context.user_answers.append(response)
        
        # Extraer temas mencionados (método simple)
        self._extract_topics(response)
        
        # Análisis más profundo con Gemini (solo después de la segunda respuesta)
        if len(self.conversation_context.user_answers) >= 2:
            self._analyze_user_intent(response)
    
    def _should_continue_asking(self) -> bool:
        """
        Determina si debemos continuar haciendo preguntas
        
        Returns:
            True si debemos continuar, False si tenemos suficiente información
        """
        if self.conversation_context.current_question_number >= self.MAX_QUESTIONS:
            return False
        
        conversation_history = self._format_conversation_history()
        
        try:
            chain = self.analysis_prompt | self.llm
            result = chain.invoke({
                "conversation_history": conversation_history,
                "questions_count": self.conversation_context.current_question_number,
                "max_questions": self.MAX_QUESTIONS
            })
            
            analysis = result.content.strip()
            
            # Si el análisis indica CONTINUAR, seguimos
            return "CONTINUAR" in analysis.upper()
            
        except Exception as e:
            print(f"Error analizando contexto: {e}")
            # En caso de error, continuamos si no hemos alcanzado el límite
            return self.conversation_context.current_question_number < self.MAX_QUESTIONS
    
    def _extract_topics(self, response: str):
        """
        Extrae temas mencionados en la respuesta para evitar preguntas redundantes
        
        Args:
            response: Respuesta del usuario
        """
        # Palabras clave para identificar temas
        topic_keywords = {
            "presupuesto": ["precio", "costo", "presupuesto", "dinero", "€", "$", "económico", "barato", "caro"],
            "categoría": ["laptop", "teléfono", "tablet", "auriculares", "teclado", "monitor", "televisor"],
            "uso": ["trabajo", "gaming", "estudio", "casa", "oficina", "portátil", "uso", "utilizar"],
            "características": ["pantalla", "memoria", "almacenamiento", "procesador", "batería", "cámara"],
            "marca": ["marca", "apple", "samsung", "sony", "lenovo", "hp", "dell", "asus"]
        }
        
        response_lower = response.lower()
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                if topic not in self.conversation_context.topics_covered:
                    self.conversation_context.topics_covered.append(topic)
    
    def _analyze_user_intent(self, response: str):
        """
        Analiza la intención y contexto profundo de la respuesta usando Gemini
        (Método opcional para mejorar la comprensión del contexto)
        
        Args:
            response: Última respuesta del usuario
        """
        try:
            # Prompt para análisis rápido de intención
            intent_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analiza BREVEMENTE la siguiente respuesta del usuario y extrae:
                1. Tema principal mencionado (una palabra: presupuesto/categoría/uso/características/marca)
                2. Nivel de especificidad (bajo/medio/alto)
                3. Si menciona restricciones o preferencias fuertes
                
                Responde en formato: TEMA|ESPECIFICIDAD|RESTRICCIONES_SI_O_NO
                Ejemplo: "categoria|alto|si" o "presupuesto|medio|no"
                """),
                ("user", f"Respuesta: {response}")
            ])
            
            chain = intent_prompt | self.llm
            result = chain.invoke({})
            analysis = result.content.strip().lower()
            
            # Guardar análisis en memoria para uso futuro
            self.update_memory(f"intent_analysis_{len(self.conversation_context.user_answers)}", analysis)
            
        except Exception as e:
            # Si falla el análisis, continuar sin problema
            pass
    
    def _format_conversation_history(self) -> str:
        """
        Formatea el historial de la conversación para el contexto
        
        Returns:
            Historial formateado
        """
        if not self.conversation_context.questions_asked:
            return "Conversación recién iniciada."
        
        history = []
        for i, (question, answer) in enumerate(zip(
            self.conversation_context.questions_asked,
            self.conversation_context.user_answers
        ), 1):
            history.append(f"Pregunta {i}: {question}")
            history.append(f"Respuesta {i}: {answer}")
            history.append("")
        
        return "\n".join(history)
    
    def has_more_questions(self) -> bool:
        """
        Verifica si hay más preguntas por hacer
        
        Returns:
            True si puede hacer más preguntas
        """
        return self.conversation_context.current_question_number < self.MAX_QUESTIONS
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa toda la información recopilada y genera un resumen estructurado
        
        Args:
            input_data: Datos de entrada (opcional)
            
        Returns:
            Resumen estructurado de la información recopilada
        """
        conversation_history = self._format_conversation_history()
        
        # Prompt para analizar toda la conversación
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un analista experto en comprender necesidades de usuarios.
            Analiza la siguiente conversación y extrae información estructurada sobre:
            
            1. **Categoría de producto**: Tipo de producto que busca
            2. **Presupuesto**: Rango de precio mencionado o implícito
            3. **Características prioritarias**: Qué características son más importantes
            4. **Uso previsto**: Para qué necesita el producto
            5. **Preferencias específicas**: Marcas, especificaciones técnicas, etc.
            6. **Restricciones**: Limitaciones mencionadas
            7. **Información adicional**: Cualquier otro dato relevante
            
            Formato tu respuesta de manera clara y estructurada.
            Si alguna información no fue proporcionada, indícalo."""),
            ("user", "Conversación:\n\n{conversation}\n\nAnaliza y estructura esta información:")
        ])
        
        try:
            chain = summary_prompt | self.llm
            result = chain.invoke({
                "conversation": conversation_history
            })
            
            summary = result.content
            
            # Guardar en memoria
            self.update_memory("conversation_history", conversation_history)
            self.update_memory("analysis", summary)
            self.update_memory("questions_asked", self.conversation_context.questions_asked)
            self.update_memory("user_answers", self.conversation_context.user_answers)
            
            return {
                "agent": self.name,
                "status": "completed",
                "questions_asked": len(self.conversation_context.questions_asked),
                "conversation_history": conversation_history,
                "structured_analysis": summary,
                "topics_covered": self.conversation_context.topics_covered
            }
            
        except Exception as e:
            return {
                "agent": self.name,
                "status": "error",
                "error": str(e),
                "conversation_history": conversation_history
            }
    
    def reset(self):
        """Reinicia el agente para una nueva sesión"""
        self.conversation_context = ConversationContext()
        self.clear_memory()
    
    def get_progress(self) -> str:
        """
        Obtiene el progreso actual de las preguntas
        
        Returns:
            String con el progreso (ej: "3/5")
        """
        return f"{self.conversation_context.current_question_number}/{self.MAX_QUESTIONS}"
    
    def get_summary(self) -> str:
        """
        Obtiene un resumen rápido de la información recopilada hasta ahora
        
        Returns:
            Resumen de la información
        """
        if not self.conversation_context.user_answers:
            return "No se ha recopilado información aún."
        
        return f"""
📊 Información recopilada:
- Preguntas realizadas: {len(self.conversation_context.questions_asked)}
- Respuestas obtenidas: {len(self.conversation_context.user_answers)}
- Temas cubiertos: {', '.join(self.conversation_context.topics_covered) if self.conversation_context.topics_covered else 'Ninguno específico'}
"""

