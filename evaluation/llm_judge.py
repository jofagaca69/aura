import json
import time
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from evaluation.config import LLM_JUDGE_CONFIG
from src.config import config
from src.utils.rate_limiter import RateLimiter


class LLMJudge:
    """
    Evaluador LLM-as-Judge para evaluar la calidad de las recomendaciones
    """

    def __init__(self):
        llm_kwargs = {
            "model": LLM_JUDGE_CONFIG["model"],
            "temperature": LLM_JUDGE_CONFIG["temperature"],
            "google_api_key": config.GOOGLE_API_KEY
        }

        timeout = LLM_JUDGE_CONFIG.get("timeout", 60)

        try:
            # Algunas versiones de langchain-google-genai soportan timeout
            if hasattr(ChatGoogleGenerativeAI, '__init__'):
                # Verificar si acepta timeout como parámetro
                import inspect
                sig = inspect.signature(ChatGoogleGenerativeAI.__init__)
                if 'timeout' in sig.parameters:
                    llm_kwargs['timeout'] = timeout
        except Exception:
            pass

        self.llm = ChatGoogleGenerativeAI(**llm_kwargs)
        self.timeout = timeout
        self.max_retries = LLM_JUDGE_CONFIG.get("max_retries", 3)
        self.rate_limiter = RateLimiter()

        """ 
        Prompt para evaluación de recomendaciones
        """
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Eres un evaluador experto de sistemas de recomendación de productos.
            Tu tarea es evaluar la calidad de las recomendaciones generadas por un asistente de IA.
            
            CRITERIOS DE EVALUACIÓN (escala 0-10):
            
            1. **RELEVANCIA** (0-10):
               - ¿Los productos recomendados coinciden con las necesidades del usuario?
               - ¿Se respetan las restricciones de presupuesto? (Si tiene restricción de presupuesto)
               - ¿Las características solicitadas están presentes?
               
            2. **DIVERSIDAD** (0-10):
               - ¿Hay variedad apropiada en las opciones presentadas?
               - ¿Se ofrecen diferentes rangos de precio dentro del presupuesto? (Si se presentan precios)
               - ¿Se consideran diferentes marcas o alternativas?
            
            3. **EXPLICACIÓN** (0-10):
               - ¿Las justificaciones son claras y comprensibles?
               - ¿Se explica POR QUÉ cada producto es adecuado?
               - ¿Se mencionan pros y contras relevantes?
               
            4. **PERSONALIZACIÓN** (0-10):
               - ¿Las recomendaciones se adaptaron al contexto del usuario?
               - ¿Se consideró el uso específico mencionado?
               - ¿Se reflejan las preferencias expresadas?
               
            5. **COMPLETITUD** (0-10):
               - ¿Se abordaron todos los criterios mencionados por el usuario?
               - ¿Se proporcionó información suficiente para tomar una decisión?
               - ¿Se incluyeron detalles técnicos relevantes?
               
            **Análisis del usuario:**
                {user_analysis}
                
            **Criterios de búsqueda aplicados:**
                {search_criteria}
                
            **Recomendaciones generadas:**
                {recommendations}
                
            **Número de productos encontrados:**
                {products_found}
                
            INSTRUCCIONES:
            1. Evalúa cada criterio objetivamente
            2. Proporciona un score numérico (0-10) para cada uno
            3. Calcula el score total como promedio de los 5 criterios
            4. Proporciona comentarios específicos y constructivos
            5. Sugiere mejoras concretas si aplica
            
            SÉ ESTRICTO PERO JUSTO:
            - Un 10 es excepcional, raramente otorgado
            - Un 7-8 es bueno/muy bueno
            - Un 5-6 es aceptable pero mejorable
            - Menos de 5 indica problemas serios
            
            RESPONDE EN FORMATO JSON VÁLIDO (sin markdown, sin comentarios):
            {{
                "relevancia": <número 0-10>,
                "diversidad": <número 0-10>,
                "explicacion": <número 0-10>,
                "personalizacion": <número 0-10>,
                "completitud": <número 0-10>,
                "score_total": <promedio de los 5 scores>,
                "comentarios": "<feedback específico sobre qué funcionó bien>",
                "areas_mejora": "<qué se podría mejorar>",
                "sugerencias": "<recomendaciones concretas para mejorar>",
                "veredicto": "<EXCELENTE|MUY_BUENO|BUENO|ACEPTABLE|DEFICIENTE>"
            }}
            """),
            ("user", "Evalúa estas recomendaciones:")
        ])

        """
        Prompt para evaluación de preguntas del QuestionerAgent
        """
        self.question_evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Eres un evaluador experto de sistemas de preguntas interactivas.
            Tu tarea es evaluar la calidad de las preguntas generadas por un agente de IA.
            
            CRITERIOS DE EVALUACIÓN (escala 0-10):
            
            1. **CONTEXTUALIDAD** (0-10):
               - ¿Las preguntas se adaptan al contexto de respuestas previas?
               - ¿Evitan repetir información ya obtenida?
               - ¿Profundizan en áreas relevantes?
               
            2. **RELEVANCIA** (0-10):
               - ¿Las preguntas buscan información crítica para hacer recomendaciones?
               - ¿Están enfocadas en necesidades del usuario?
               - ¿Ayudan a entender mejor las preferencias?
            
            3. **NATURALIDAD** (0-10):
               - ¿Las preguntas suenan naturales y conversacionales?
               - ¿Son fáciles de entender?
               - ¿Evitan ser robóticas o repetitivas?
               
            4. **EFICIENCIA** (0-10):
               - ¿Las preguntas obtienen información útil?
               - ¿Evitan preguntas redundantes o innecesarias?
               - ¿Maximizan el valor de cada pregunta?
               
            5. **COMPLETITUD** (0-10):
               - ¿Las preguntas cubren los aspectos esenciales?
               - ¿Se obtiene suficiente información para recomendar?
               - ¿Se identifican necesidades clave?
               
            **Historial de conversación:**
                {conversation_history}
                
            **Información extraída:**
                {extracted_info}
            
            INSTRUCCIONES:
            1. Evalúa cada criterio objetivamente
            2. Proporciona un score numérico (0-10) para cada uno
            3. Calcula el score total como promedio de los 5 criterios
            4. Identifica las mejores preguntas y las que se pueden mejorar
            5. Proporciona sugerencias concretas
            
            SÉ ESTRICTO PERO JUSTO:
            - Un 10 es excepcional, raramente otorgado
            - Un 7-8 es bueno/muy bueno
            - Un 5-6 es aceptable pero mejorable
            - Menos de 5 indica problemas serios
            
            RESPONDE EN FORMATO JSON VÁLIDO (sin markdown, sin comentarios):
            {{
                "contextualidad": <número 0-10>,
                "relevancia": <número 0-10>,
                "naturalidad": <número 0-10>,
                "eficiencia": <número 0-10>,
                "completitud": <número 0-10>,
                "score_total": <promedio de los 5 scores>,
                "comentarios": "<feedback específico sobre qué funcionó bien>",
                "mejores_preguntas": ["pregunta 1", "pregunta 2"],
                "preguntas_mejorables": ["pregunta que podría mejorarse"],
                "sugerencias": "<recomendaciones concretas para mejorar>"
            }}
            """),
            ("user", "Evalúa la calidad de las preguntas generadas:")
        ])

    def evaluate_recommendations(
            self,
            user_analysis: str,
            search_criteria: str,
            recommendations: str,
            products_found: int
    ) -> Dict[str, Any]:
        """
        Evalúa la calidad de las recomendaciones usando Gemini

        Args:
            user_analysis: Análisis de las necesidades del usuario
            search_criteria: Criterios de búsqueda aplicados
            recommendations: Texto de recomendaciones generadas
            products_found: Número de productos encontrados

        Returns:
            Diccionario con scores y feedback
        """
        try:
            chain = self.evaluation_prompt | self.llm

            inputs = {
                "user_analysis": user_analysis or "No disponible",
                "search_criteria": search_criteria or "No disponible",
                "recommendations": recommendations or "No se generaron recomendaciones",
                "products_found": products_found
            }

            result = self._invoke_with_timeout(chain, inputs)

            # Parsear JSON
            response_text = result.content.strip()

            # Limpiar markdown si está presente
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(response_text)

            return {
                "success": True,
                "evaluation": evaluation,
                "raw_response": result.content
            }

        except (TimeoutError, Exception) as e:
            error_msg = str(e)
            print(f"⚠️ Error evaluando recomendaciones: {error_msg}")

            # Retornar evaluación por defecto en caso de timeout
            if "timeout" in error_msg.lower():
                return {
                    "success": False,
                    "error": "timeout",
                    "error_message": f"Timeout después de {self.timeout}s y {self.max_retries} intentos",
                    "evaluation": {
                        "relevancia": 5.0,
                        "diversidad": 5.0,
                        "explicacion": 5.0,
                        "personalizacion": 5.0,
                        "completitud": 5.0,
                        "score_total": 5.0,
                        "comentarios": "Evaluación no disponible debido a timeout",
                        "areas_mejora": "Timeout en evaluación LLM",
                        "sugerencias": "Revisar configuración de timeout o conexión",
                        "veredicto": "NO_DISPONIBLE"
                    }
                }

            return {
                "success": False,
                "error": error_msg
            }

    def evaluate_questions(
            self,
            conversation_history: str,
            extracted_info: str
    ) -> Dict[str, Any]:
        """
        Evalúa la calidad de las preguntas generadas por el QuestionerAgent

        Args:
            conversation_history: Historial completo de preguntas y respuestas
            extracted_info: Información extraída de las respuestas

        Returns:
            Diccionario con scores y feedback
        """
        try:
            chain = self.question_evaluation_prompt | self.llm

            inputs = {
                "conversation_history": conversation_history or "No hay conversación",
                "extracted_info": extracted_info or "No se extrajo información"
            }

            result = self._invoke_with_timeout(chain, inputs)

            # Parsear JSON
            response_text = result.content.strip()

            # Limpiar markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(response_text)

            return {
                "success": True,
                "evaluation": evaluation,
                "raw_response": result.content
            }

        except json.JSONDecodeError as e:
            print(f"⚠️ Error parseando JSON: {e}")
            return {
                "success": False,
                "error": "JSON parsing error",
                "raw_response": response_text if 'response_text' in locals() else ""
            }
        except (TimeoutError, Exception) as e:
            error_msg = str(e)
            print(f"⚠️ Error evaluando preguntas: {error_msg}")

            # Retornar evaluación por defecto en caso de timeout
            if "timeout" in error_msg.lower():
                return {
                    "success": False,
                    "error": "timeout",
                    "error_message": f"Timeout después de {self.timeout}s y {self.max_retries} intentos",
                    "evaluation": {
                        "contextualidad": 5.0,
                        "relevancia": 5.0,
                        "naturalidad": 5.0,
                        "eficiencia": 5.0,
                        "completitud": 5.0,
                        "score_total": 5.0,
                        "comentarios": "Evaluación no disponible debido a timeout",
                        "mejores_preguntas": [],
                        "preguntas_mejorables": [],
                        "sugerencias": "Revisar configuración de timeout o conexión"
                    }
                }

            return {
                "success": False,
                "error": error_msg
            }

    def evaluate_information_extraction(
            self,
            conversation: list,
            extracted_info: Dict[str, Any],
            expected_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evalúa la precisión de la extracción de información

        Args:
            conversation: Lista de intercambios [pregunta, respuesta]
            extracted_info: Información extraída por el sistema
            expected_info: Información esperada (ground truth)

        Returns:
            Métricas de precisión
        """
        scores = {}
        total_score = 0
        fields_evaluated = 0

        # Evaluar cada campo
        for field, expected_value in expected_info.items():
            extracted_value = extracted_info.get(field)

            if expected_value is None:
                # Si no se esperaba valor, verificar que no se extrajo nada incorrecto
                scores[field] = 1.0 if extracted_value is None or extracted_value == [] else 0.5
            elif isinstance(expected_value, list):
                # Para listas, calcular overlap
                if not expected_value:  # Lista vacía esperada
                    scores[field] = 1.0 if not extracted_value else 0.5
                else:
                    # Convertir extracted_value a lista si es string (caso especial para categoria_producto)
                    if isinstance(extracted_value, str):
                        # Si el valor extraído es un string, compararlo con cada elemento de la lista esperada
                        extracted_lower = extracted_value.lower()
                        expected_lower_list = [str(v).lower() for v in expected_value]
                        
                        # Verificar si el string extraído contiene o coincide con algún valor esperado
                        matches = []
                        for expected_item in expected_lower_list:
                            # Verificar coincidencia exacta o si el string extraído contiene el valor esperado
                            if extracted_lower == expected_item or expected_item in extracted_lower or extracted_lower in expected_item:
                                matches.append(expected_item)
                        
                        if matches:
                            # Si hay coincidencias, calcular score basado en cuántos valores esperados coinciden
                            scores[field] = len(matches) / len(expected_value)
                        else:
                            # Verificar similitud por palabras
                            extracted_words = set(extracted_lower.split())
                            all_expected_words = set()
                            for exp_item in expected_lower_list:
                                all_expected_words.update(exp_item.split())
                            
                            if all_expected_words:
                                overlap = len(extracted_words & all_expected_words)
                                scores[field] = overlap / len(all_expected_words) if all_expected_words else 0.0
                            else:
                                scores[field] = 0.0
                    else:
                        # Si extracted_value ya es una lista, usar el método original
                        extracted_set = set(str(v).lower() for v in (extracted_value or []))
                        expected_set = set(str(v).lower() for v in expected_value)

                        if not expected_set:
                            scores[field] = 1.0
                        else:
                            # Jaccard similarity
                            intersection = len(extracted_set & expected_set)
                            union = len(extracted_set | expected_set)
                            scores[field] = intersection / union if union > 0 else 0.0
            elif isinstance(expected_value, bool):
                # Para booleanos, verificar coincidencia exacta
                if extracted_value is None:
                    scores[field] = 0.0
                else:
                    scores[field] = 1.0 if bool(extracted_value) == expected_value else 0.0
            elif isinstance(expected_value, (int, float)):
                # Para números, verificar si está dentro del rango razonable
                if extracted_value is None:
                    scores[field] = 0.0
                else:
                    try:
                        extracted_num = float(extracted_value)
                        diff = abs(extracted_num - expected_value)
                        # Tolerancia del 10%
                        tolerance = abs(expected_value) * 0.1 if expected_value != 0 else 0.1
                        scores[field] = 1.0 if diff <= tolerance else max(0, 1 - (diff / abs(expected_value)) if expected_value != 0 else 0.0)
                    except (ValueError, TypeError):
                        scores[field] = 0.0
            else:
                # Para strings, verificar similitud semántica simple
                if extracted_value is None:
                    scores[field] = 0.0
                else:
                    extracted_lower = str(extracted_value).lower()
                    expected_lower = str(expected_value).lower()

                    # Si hay coincidencia exacta, score perfecto
                    if extracted_lower == expected_lower:
                        scores[field] = 1.0
                    else:
                        # Similitud básica: palabras en común
                        extracted_words = set(extracted_lower.split())
                        expected_words = set(expected_lower.split())

                        if not expected_words:
                            scores[field] = 1.0
                        else:
                            overlap = len(extracted_words & expected_words)
                            scores[field] = overlap / len(expected_words)

            total_score += scores[field]
            fields_evaluated += 1

        # Calcular score promedio
        avg_score = total_score / fields_evaluated if fields_evaluated > 0 else 0.0

        return {
            "individual_scores": scores,
            "average_score": avg_score,
            "accuracy_percentage": avg_score * 100,
            "fields_evaluated": fields_evaluated,
            "perfect_matches": sum(1 for s in scores.values() if s >= 0.95),
            "partial_matches": sum(1 for s in scores.values() if 0.5 <= s < 0.95),
            "mismatches": sum(1 for s in scores.values() if s < 0.5)
        }

    def _invoke_with_timeout(self, chain, inputs: Dict[str, Any]) -> Any:
        """Invoca la cadena con timeout, retry y rate limiting usando threading"""
        import threading

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Aplicar rate limiting antes de cada intento
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed()

                result_container = [None]
                exception_container = [None]

                def invoke_chain():
                    """Función que ejecuta la invocación en un thread separado"""
                    try:
                        result_container[0] = chain.invoke(inputs)
                    except Exception as e:
                        exception_container[0] = e

                # Crear thread para ejecutar la invocación
                thread = threading.Thread(target=invoke_chain, daemon=True)
                thread.start()
                thread.join(timeout=self.timeout)

                # Verificar si el thread aún está corriendo (timeout)
                if thread.is_alive():
                    raise TimeoutError(
                        f"La operación excedió el timeout de {self.timeout}s "
                        f"(intento {attempt + 1}/{self.max_retries})"
                    )

                # Si hubo una excepción en el thread, relanzarla
                if exception_container[0]:
                    exception = exception_container[0]

                    # Manejar específicamente errores de rate limit (429)
                    error_str = str(exception).lower()
                    if "resourceexhausted" in error_str or "429" in error_str or "quota" in error_str:
                        # Extraer tiempo de espera del error si está disponible
                        retry_after = None
                        if "retry in" in error_str or "retry_delay" in error_str:
                            import re
                            numbers = re.findall(r'(\d+\.?\d*)', error_str)
                            if numbers:
                                retry_after = float(numbers[0])

                        if self.rate_limiter:
                            self.rate_limiter.handle_rate_limit_error(exception, retry_after)

                        # Reintentar después de manejar el rate limit
                        if attempt < self.max_retries - 1:
                            continue
                        else:
                            raise exception

                    raise exception

                # Si tenemos resultado, retornarlo
                if result_container[0] is not None:
                    return result_container[0]

                # Si llegamos aquí, algo salió mal
                raise Exception("No se obtuvo resultado ni excepción")

            except TimeoutError as e:
                last_exception = e
                print(f"⚠️ Timeout en intento {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** attempt, 5)  # Backoff exponencial, máximo 5 segundos
                    time.sleep(wait_time)
                continue
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Si es un error de rate limit, ya se manejó arriba
                if "resourceexhausted" in error_str or "429" in error_str or "quota" in error_str:
                    if attempt >= self.max_retries - 1:
                        raise
                    continue

                # Si es un error de red o API, reintentar
                if any(keyword in error_str for keyword in ["timeout", "connection", "network", "api"]):
                    print(f"⚠️ Error de conexión en intento {attempt + 1}/{self.max_retries}: {e}")
                    if attempt < self.max_retries - 1:
                        wait_time = min(2 ** attempt, 5)  # Backoff exponencial
                        time.sleep(wait_time)
                    continue
                else:
                    # Otros errores no se reintentan
                    raise

        # Si todos los intentos fallaron
        raise last_exception or TimeoutError(
            f"Todos los {self.max_retries} intentos fallaron después de {self.timeout}s cada uno"
        )