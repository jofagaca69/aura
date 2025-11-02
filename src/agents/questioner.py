"""
Agente preguntador interactivo para recopilar información del usuario
Utiliza Gemini (Google LLM) para generar preguntas contextuales e inteligentes
"""
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json

from src.agents.base_agent import BaseAgent


class ExtractedInfo(BaseModel):
    """Información extraída de las respuestas del usuario"""
    categoria_producto: Optional[str] = Field(default=None, description="Tipo de producto buscado")
    presupuesto_min: Optional[float] = Field(default=None, description="Presupuesto mínimo")
    presupuesto_max: Optional[float] = Field(default=None, description="Presupuesto máximo")
    uso_principal: Optional[str] = Field(default=None, description="Uso principal del producto")
    caracteristicas_clave: List[str] = Field(default_factory=list, description="Características importantes")
    preferencias_marca: List[str] = Field(default_factory=list, description="Marcas preferidas")
    restricciones: List[str] = Field(default_factory=list, description="Limitaciones o restricciones")
    nivel_urgencia: Optional[str] = Field(default=None, description="Qué tan urgente es la compra")
    contexto_adicional: Optional[str] = Field(default=None, description="Información adicional relevante")


class ConversationContext(BaseModel):
    """Contexto enriquecido de la conversación"""
    questions_asked: List[str] = Field(default_factory=list, description="Preguntas ya realizadas")
    user_answers: List[str] = Field(default_factory=list, description="Respuestas del usuario")
    topics_covered: List[str] = Field(default_factory=list, description="Temas ya cubiertos")
    current_question_number: int = Field(default=0, description="Número de pregunta actual")
    extracted_info: ExtractedInfo = Field(default_factory=ExtractedInfo, description="Información extraída")
    information_score: Dict[str, float] = Field(default_factory=dict, description="Score de información recopilada")


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
        
        # Prompt para extracción inteligente de información usando Gemini
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un experto analizador de conversaciones de ventas. Tu tarea es extraer 
            información estructurada de las respuestas del usuario.
            
            📋 INFORMACIÓN A EXTRAER:
            1. **categoria_producto**: Tipo de producto (laptop, teléfono, tablet, etc.) - STRING o null
            2. **presupuesto_min**: Presupuesto mínimo en números - FLOAT o null
            3. **presupuesto_max**: Presupuesto máximo en números - FLOAT o null
            4. **uso_principal**: Uso principal del producto - STRING o null
            5. **caracteristicas_clave**: Lista de características importantes - LIST[STRING]
            6. **preferencias_marca**: Marcas mencionadas o preferidas - LIST[STRING]
            7. **restricciones**: Limitaciones o restricciones - LIST[STRING]
            8. **nivel_urgencia**: ¿Qué tan urgente? (inmediato/pronto/sin_prisa) - STRING o null
            9. **contexto_adicional**: Cualquier otra información relevante - STRING o null
            
            🎯 INSTRUCCIONES:
            - Extrae SOLO información EXPLÍCITA o CLARAMENTE IMPLÍCITA
            - Si no hay información sobre un campo, usa null o lista vacía []
            - Para presupuestos, convierte texto a números (ej: "mil euros" → 1000.0)
            - Sé conservador: mejor null que información incorrecta
            
            📝 INFORMACIÓN YA RECOPILADA:
            {previous_info}
            
            💬 ÚLTIMA RESPUESTA DEL USUARIO:
            "{user_response}"
            
            🎯 RESPONDE EN FORMATO JSON VÁLIDO (sin markdown, sin comentarios):
            {{
                "categoria_producto": "valor o null",
                "presupuesto_min": número o null,
                "presupuesto_max": número o null,
                "uso_principal": "valor o null",
                "caracteristicas_clave": ["lista", "de", "características"],
                "preferencias_marca": ["lista", "de", "marcas"],
                "restricciones": ["lista", "de", "restricciones"],
                "nivel_urgencia": "valor o null",
                "contexto_adicional": "valor o null"
            }}"""),
            ("user", "Extrae la información de esta respuesta:")
        ])
        
        # Prompt mejorado para generar preguntas ultra-personalizadas con Gemini
        self.question_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un asistente de compras experto y empático que hace preguntas INTELIGENTES 
            y PERSONALIZADAS para entender las necesidades del usuario. Tu objetivo es descubrir qué 
            producto necesita realmente y por qué.
            
            🎯 ESTRATEGIA AVANZADA DE PREGUNTAS:
            
            1. **ANALIZA LA INFORMACIÓN EXTRAÍDA**: Revisa qué datos ya tienes
            2. **IDENTIFICA VACÍOS CRÍTICOS**: ¿Qué información esencial falta?
            3. **PRIORIZA INTELIGENTEMENTE**: Pregunta primero lo más importante
            4. **CONECTA Y PROFUNDIZA**: Usa lo que sabes para preguntas más específicas
            5. **SÉ NATURAL**: Haz que la conversación fluya orgánicamente
            
            📊 INFORMACIÓN YA RECOPILADA:
            {extracted_info_summary}
            
            🎯 INFORMACIÓN QUE AÚN FALTA:
            {missing_info}
            
            📝 CONVERSACIÓN COMPLETA:
            {conversation_history}
            
            💡 EJEMPLOS DE PREGUNTAS CONTEXTUALES:
            
            Escenario 1 - Ya sabes: laptop para programación
            Pregunta inteligente: "Genial, para programación. ¿Trabajas con herramientas pesadas como Docker, 
            máquinas virtuales o IDEs como Android Studio? Esto nos ayudará a definir cuánta RAM necesitas."
            
            Escenario 2 - Ya sabes: teléfono, presupuesto 500-700€
            Pregunta inteligente: "Perfecto, con ese presupuesto tienes buenas opciones. ¿Qué es más importante 
            para ti: la calidad de la cámara, la duración de batería, o el rendimiento para juegos?"
            
            Escenario 3 - Ya sabes: tablet, para estudiar y ver series
            Pregunta inteligente: "Entiendo, para estudiar y entretenimiento. ¿Prefieres algo ligero y portátil 
            como una tablet de 10 pulgadas, o una pantalla más grande tipo 12 pulgadas aunque pese un poco más?"
            
            ⚠️ REGLAS CRÍTICAS:
            1. **NO repitas información** que el usuario ya dio
            2. **NO preguntes** sobre campos que ya tienes completos
            3. **USA lo que sabes** para hacer preguntas más específicas
            4. **UNA pregunta a la vez**, clara y directa
            5. **SÉ conversacional**, no robótico
            
            🎯 GENERA UNA PREGUNTA que:
            - Esté basada en el contexto completo
            - Busque la información más crítica que falta
            - Sea natural y empática
            - Ayude a entender mejor las necesidades del usuario
            
            Responde SOLO con la pregunta, sin explicaciones adicionales."""),
            ("user", "Genera la siguiente pregunta contextual:")
        ])
        
        # Prompt mejorado para analizar si necesitamos más información
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un analista experto en comprensión de necesidades de clientes.
            
            🎯 TU TAREA: Determinar si tenemos SUFICIENTE información para recomendar productos.
            
            📊 INFORMACIÓN EXTRAÍDA HASTA AHORA:
            {extracted_info_summary}
            
            📋 CRITERIOS DE EVALUACIÓN:
            
            **INFORMACIÓN CRÍTICA** (debe estar presente):
            - ✓ Categoría de producto (qué busca)
            - ✓ Presupuesto aproximado (rango de precio)
            - ✓ Uso principal O características clave
            
            **INFORMACIÓN ÚTIL** (deseable pero no esencial):
            - Preferencias de marca
            - Restricciones específicas
            - Urgencia de compra
            - Contexto adicional
            
            ✅ TENEMOS SUFICIENTE SI:
            - Categoría + Presupuesto + (Uso O Características) están presentes
            - La información es lo suficientemente específica para recomendar
            - Tenemos al menos 2 de los 3 elementos críticos con buen detalle
            
            ⚠️ NECESITAMOS MÁS SI:
            - Falta categoría de producto (crítico)
            - No sabemos el presupuesto ni aproximado (crítico)
            - No tenemos idea del uso ni características deseadas
            - La información es muy vaga o ambigua
            
            🎯 ANÁLISIS ACTUAL:
            Preguntas realizadas: {questions_count}/{max_questions}
            Score de información: {information_score}%
            
            📝 CONVERSACIÓN:
            {conversation_history}
            
            🎯 DECISIÓN:
            Responde SOLO con una palabra seguida de breve explicación:
            - "CONTINUAR: [razón]" - Si falta información crítica
            - "SUFICIENTE: [razón]" - Si podemos hacer buenas recomendaciones
            
            Sé eficiente: mejor suficiente información que perfecta."""),
            ("user", "¿Debemos continuar preguntando o ya tenemos suficiente?")
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
        
        # Generar contexto enriquecido para Gemini
        conversation_history = self._format_conversation_history()
        extracted_info_summary = self._format_extracted_info()
        missing_info = self._identify_missing_info()
        
        # Generar siguiente pregunta personalizada con Gemini usando contexto completo
        try:
            chain = self.question_prompt | self.llm
            result = chain.invoke({
                "extracted_info_summary": extracted_info_summary,
                "missing_info": missing_info,
                "conversation_history": conversation_history
            })
            
            question = result.content.strip()
            
            # Limpiar la pregunta (remover comillas extras, markdown, etc.)
            question = question.strip('"').strip("'").strip('`')
            if question.startswith("Pregunta:"):
                question = question.replace("Pregunta:", "").strip()
            
            self.conversation_context.current_question_number += 1
            self.conversation_context.questions_asked.append(question)
            
            print(f"✅ Pregunta {self.conversation_context.current_question_number} generada")
            
            return question
            
        except Exception as e:
            print(f"⚠️  Error generando pregunta con Gemini: {e}")
            return None
    
    def add_user_response(self, response: str):
        """
        Añade una respuesta del usuario al contexto y extrae información clave usando Gemini
        
        Args:
            response: Respuesta del usuario
        """
        self.conversation_context.user_answers.append(response)
        
        # Extraer información estructurada usando Gemini
        self._extract_information_with_llm(response)
        
        # Extraer temas mencionados (método complementario rápido)
        self._extract_topics(response)
        
        # Calcular score de información recopilada
        self._calculate_information_score()
    
    def _should_continue_asking(self) -> bool:
        """
        Determina si debemos continuar haciendo preguntas usando análisis inteligente con Gemini
        
        Returns:
            True si debemos continuar, False si tenemos suficiente información
        """
        if self.conversation_context.current_question_number >= self.MAX_QUESTIONS:
            return False
        
        # Obtener contexto enriquecido
        conversation_history = self._format_conversation_history()
        extracted_info_summary = self._format_extracted_info()
        info_score = self._calculate_information_score()
        
        try:
            chain = self.analysis_prompt | self.llm
            result = chain.invoke({
                "conversation_history": conversation_history,
                "extracted_info_summary": extracted_info_summary,
                "information_score": info_score,
                "questions_count": self.conversation_context.current_question_number,
                "max_questions": self.MAX_QUESTIONS
            })
            
            analysis = result.content.strip()
            
            # Si el análisis indica CONTINUAR, seguimos
            should_continue = "CONTINUAR" in analysis.upper()
            
            # Log del análisis para debugging
            print(f"📊 Análisis LLM: {analysis[:100]}...")
            print(f"🎯 Decisión: {'Continuar' if should_continue else 'Suficiente información'}")
            
            return should_continue
            
        except Exception as e:
            print(f"⚠️  Error analizando contexto: {e}")
            # En caso de error, continuamos solo si el score es bajo
            return info_score < 60
    
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
    
    def _extract_information_with_llm(self, response: str):
        """
        Extrae información estructurada de la respuesta del usuario usando Gemini
        
        Args:
            response: Respuesta del usuario
        """
        try:
            # Formatear información previa
            previous_info = self._format_extracted_info()
            
            # Usar Gemini para extraer información estructurada
            chain = self.extraction_prompt | self.llm
            result = chain.invoke({
                "user_response": response,
                "previous_info": previous_info
            })
            
            # Parsear respuesta JSON
            extracted_text = result.content.strip()
            
            # Limpiar markdown si está presente
            if "```json" in extracted_text:
                extracted_text = extracted_text.split("```json")[1].split("```")[0].strip()
            elif "```" in extracted_text:
                extracted_text = extracted_text.split("```")[1].split("```")[0].strip()
            
            extracted_data = json.loads(extracted_text)
            
            # Actualizar información extraída (merge con info previa)
            info = self.conversation_context.extracted_info
            
            # Actualizar solo campos no nulos
            if extracted_data.get("categoria_producto"):
                info.categoria_producto = extracted_data["categoria_producto"]
            
            if extracted_data.get("presupuesto_min") is not None:
                info.presupuesto_min = float(extracted_data["presupuesto_min"])
                
            if extracted_data.get("presupuesto_max") is not None:
                info.presupuesto_max = float(extracted_data["presupuesto_max"])
            
            if extracted_data.get("uso_principal"):
                info.uso_principal = extracted_data["uso_principal"]
            
            if extracted_data.get("nivel_urgencia"):
                info.nivel_urgencia = extracted_data["nivel_urgencia"]
            
            if extracted_data.get("contexto_adicional"):
                # Combinar con contexto previo si existe
                if info.contexto_adicional:
                    info.contexto_adicional += f" | {extracted_data['contexto_adicional']}"
                else:
                    info.contexto_adicional = extracted_data["contexto_adicional"]
            
            # Para listas, hacer merge (no duplicar)
            for caracteristica in extracted_data.get("caracteristicas_clave", []):
                if caracteristica and caracteristica not in info.caracteristicas_clave:
                    info.caracteristicas_clave.append(caracteristica)
            
            for marca in extracted_data.get("preferencias_marca", []):
                if marca and marca not in info.preferencias_marca:
                    info.preferencias_marca.append(marca)
            
            for restriccion in extracted_data.get("restricciones", []):
                if restriccion and restriccion not in info.restricciones:
                    info.restricciones.append(restriccion)
            
            print(f"✅ Información extraída: {len(extracted_data)} campos procesados")
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Error parseando JSON de extracción: {e}")
            print(f"Respuesta recibida: {extracted_text[:200]}")
        except Exception as e:
            print(f"⚠️  Error extrayendo información con LLM: {e}")
    
    def _calculate_information_score(self) -> float:
        """
        Calcula un score (0-100) de cuánta información crítica hemos recopilado
        
        Returns:
            Score de 0 a 100
        """
        info = self.conversation_context.extracted_info
        score = 0.0
        
        # Información crítica (70% del score)
        if info.categoria_producto:
            score += 25
        
        if info.presupuesto_min or info.presupuesto_max:
            score += 25
        
        if info.uso_principal or len(info.caracteristicas_clave) > 0:
            score += 20
        
        # Información adicional útil (30% del score)
        if len(info.caracteristicas_clave) >= 2:
            score += 10
        
        if len(info.preferencias_marca) > 0:
            score += 5
        
        if len(info.restricciones) > 0:
            score += 5
        
        if info.nivel_urgencia:
            score += 5
        
        if info.contexto_adicional:
            score += 5
        
        self.conversation_context.information_score["total"] = score
        return score
    
    def _format_extracted_info(self) -> str:
        """
        Formatea la información extraída en un texto legible
        
        Returns:
            String formateado con la información extraída
        """
        info = self.conversation_context.extracted_info
        
        lines = []
        lines.append("📋 INFORMACIÓN EXTRAÍDA:")
        lines.append("")
        
        lines.append(f"🏷️  Categoría: {info.categoria_producto or '❌ No especificada'}")
        
        if info.presupuesto_min or info.presupuesto_max:
            presupuesto_str = ""
            if info.presupuesto_min and info.presupuesto_max:
                presupuesto_str = f"{info.presupuesto_min} - {info.presupuesto_max}€"
            elif info.presupuesto_min:
                presupuesto_str = f"Desde {info.presupuesto_min}€"
            elif info.presupuesto_max:
                presupuesto_str = f"Hasta {info.presupuesto_max}€"
            lines.append(f"💰 Presupuesto: {presupuesto_str}")
        else:
            lines.append("💰 Presupuesto: ❌ No especificado")
        
        lines.append(f"🎯 Uso principal: {info.uso_principal or '❌ No especificado'}")
        
        if info.caracteristicas_clave:
            lines.append(f"⚙️  Características: {', '.join(info.caracteristicas_clave)}")
        else:
            lines.append("⚙️  Características: ❌ No especificadas")
        
        if info.preferencias_marca:
            lines.append(f"🏢 Marcas: {', '.join(info.preferencias_marca)}")
        
        if info.restricciones:
            lines.append(f"⚠️  Restricciones: {', '.join(info.restricciones)}")
        
        if info.nivel_urgencia:
            lines.append(f"⏰ Urgencia: {info.nivel_urgencia}")
        
        if info.contexto_adicional:
            lines.append(f"📝 Contexto: {info.contexto_adicional[:100]}...")
        
        return "\n".join(lines)
    
    def _identify_missing_info(self) -> str:
        """
        Identifica qué información crítica aún falta
        
        Returns:
            String describiendo la información faltante
        """
        info = self.conversation_context.extracted_info
        missing = []
        
        if not info.categoria_producto:
            missing.append("❌ Categoría de producto (CRÍTICO)")
        
        if not info.presupuesto_min and not info.presupuesto_max:
            missing.append("❌ Presupuesto aproximado (CRÍTICO)")
        
        if not info.uso_principal and len(info.caracteristicas_clave) == 0:
            missing.append("❌ Uso principal o características clave (CRÍTICO)")
        
        if len(info.caracteristicas_clave) < 2:
            missing.append("⚠️  Características específicas (ÚTIL)")
        
        if len(info.preferencias_marca) == 0:
            missing.append("⚠️  Preferencias de marca (ÚTIL)")
        
        if not missing:
            return "✅ Tenemos toda la información esencial"
        
        return "\n".join(missing)
    
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
            
            # Obtener información extraída
            extracted_info_dict = self.get_extracted_info()
            
            # Guardar en memoria
            self.update_memory("conversation_history", conversation_history)
            self.update_memory("analysis", summary)
            self.update_memory("questions_asked", self.conversation_context.questions_asked)
            self.update_memory("user_answers", self.conversation_context.user_answers)
            self.update_memory("extracted_info", extracted_info_dict)
            
            return {
                "agent": self.name,
                "status": "completed",
                "questions_asked": len(self.conversation_context.questions_asked),
                "conversation_history": conversation_history,
                "structured_analysis": summary,
                "extracted_information": extracted_info_dict,
                "information_score": self._calculate_information_score(),
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
        Obtiene un resumen enriquecido de la información recopilada hasta ahora
        
        Returns:
            Resumen detallado de la información
        """
        if not self.conversation_context.user_answers:
            return "No se ha recopilado información aún."
        
        score = self._calculate_information_score()
        info_summary = self._format_extracted_info()
        
        return f"""
📊 RESUMEN DE INFORMACIÓN RECOPILADA

🎯 Score de completitud: {score:.0f}%
{'🟢' if score >= 70 else '🟡' if score >= 50 else '🔴'} {'Excelente' if score >= 70 else 'Buena' if score >= 50 else 'Necesita más información'}

{info_summary}

📝 Progreso:
- Preguntas realizadas: {len(self.conversation_context.questions_asked)}/{self.MAX_QUESTIONS}
- Respuestas obtenidas: {len(self.conversation_context.user_answers)}
- Temas cubiertos: {', '.join(self.conversation_context.topics_covered) if self.conversation_context.topics_covered else 'Ninguno específico'}
"""
    
    def get_extracted_info(self) -> Dict[str, Any]:
        """
        Obtiene la información extraída en formato de diccionario
        
        Returns:
            Diccionario con la información extraída
        """
        info = self.conversation_context.extracted_info
        return {
            "categoria_producto": info.categoria_producto,
            "presupuesto_min": info.presupuesto_min,
            "presupuesto_max": info.presupuesto_max,
            "uso_principal": info.uso_principal,
            "caracteristicas_clave": info.caracteristicas_clave,
            "preferencias_marca": info.preferencias_marca,
            "restricciones": info.restricciones,
            "nivel_urgencia": info.nivel_urgencia,
            "contexto_adicional": info.contexto_adicional,
            "information_score": self._calculate_information_score()
        }

