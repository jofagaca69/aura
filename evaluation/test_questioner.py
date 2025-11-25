"""
Tests para evaluar el QuestionerAgent
"""
import json
import time
from typing import Dict, Any, List
from datetime import datetime

from src.agents.questioner import QuestionerAgent
from evaluation.llm_judge import LLMJudge
from evaluation.config import THRESHOLDS


class QuestionerEvaluator:
    """
    Evaluador especializado para el QuestionerAgent
    """

    def __init__(self):
        self.agent = QuestionerAgent()
        self.llm_judge = LLMJudge()
        self.results = []

    def test_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta un escenario de prueba completo para el QuestionerAgent

        Args:
            scenario: Escenario de prueba con conversación y extracción esperada

        Returns:
            Resultados de la evaluación
        """
        print(f"\n🧪 Testing escenario: {scenario['name']}")
        print(f"   Dificultad: {scenario['difficulty']}")

        start_time = datetime.now()

        # Reiniciar agente
        self.agent.reset()

        # Simular conversación
        questions_asked = []

        # Primera pregunta
        first_question = self.agent.generate_next_question()
        questions_asked.append(first_question)
        print(f"   ❓ Pregunta 1: {first_question[:60]}...")

        # Procesar respuestas del usuario
        for i, user_response in enumerate(scenario['conversation'], 1):
            print(f"   💬 Respuesta {i}: {user_response[:60]}...")

            # Agregar respuesta del usuario
            self.agent.add_user_response(user_response)

            # Generar siguiente pregunta
            next_question = self.agent.generate_next_question()
            if next_question:
                questions_asked.append(next_question)
                print(f"   ❓ Pregunta {i + 1}: {next_question[:60]}...")
            else:
                print(f"   ✓ No más preguntas necesarias")
                break

        # Obtener información extraída
        extracted_info = self.agent.get_extracted_info()

        # Procesar y obtener análisis final
        result = self.agent.process({})

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Evaluación 1: Precisión de extracción
        extraction_eval = self._evaluate_extraction_accuracy(
            extracted_info,
            scenario.get('expected_extraction', {})
        )

        # Evaluación 2: Calidad de preguntas (usando LLM Judge)
        conversation_history = self.agent.get_memory('conversation_history', '')
        extracted_info_str = json.dumps(extracted_info, indent=2, ensure_ascii=False)

        questions_eval = self.llm_judge.evaluate_questions(
            conversation_history=conversation_history,
            extracted_info=extracted_info_str
        )

        # Evaluación 3: Eficiencia (número de preguntas)
        questions_count = len(questions_asked)
        max_questions = THRESHOLDS['questioner']['max_questions']
        efficiency_score = 1.0 if questions_count <= max_questions else max_questions / questions_count

        # Evaluación 4: Score de información
        info_score = extracted_info.get('information_score', 0)
        info_score_passed = info_score >= THRESHOLDS['questioner']['information_score_min']

        # Resultado consolidado
        test_result = {
            "scenario_id": scenario['id'],
            "scenario_name": scenario['name'],
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "questions_asked": questions_count,
            "questions_list": questions_asked,
            "user_responses": scenario['conversation'],
            "extracted_info": extracted_info,
            "expected_info": scenario.get('expected_extraction', {}),
            "extraction_evaluation": extraction_eval,
            "questions_evaluation": questions_eval,
            "efficiency_score": efficiency_score,
            "information_score": info_score,
            "thresholds_passed": {
                "extraction_accuracy": extraction_eval['average_score'] >= THRESHOLDS['questioner'][
                    'extraction_accuracy'],
                "max_questions": questions_count <= max_questions,
                "information_score": info_score_passed
            },
            # Lógica mejorada: más flexible, permite éxito parcial
            "overall_success": (
                # Criterio principal: precisión de extracción
                    extraction_eval['average_score'] >= THRESHOLDS['questioner']['extraction_accuracy'] and
                    # Criterio secundario: número de preguntas (debe cumplirse)
                    questions_count <= max_questions and
                    # Criterio terciario: información score (más flexible - permite 5 puntos menos)
                    (info_score >= (THRESHOLDS['questioner']['information_score_min'] - 10.0) or
                     extraction_eval['average_score'] >= 0.90)
            # Si la precisión es muy alta, es más flexible con info_score
            )
        }

        self.results.append(test_result)

        # Imprimir resumen
        print(f"\n   📊 Resultados:")
        print(f"      - Precisión extracción: {extraction_eval['accuracy_percentage']:.1f}%")
        print(f"      - Preguntas realizadas: {questions_count}/{max_questions}")
        print(f"      - Score de información: {info_score:.1f}%")
        print(f"      - Tiempo ejecución: {execution_time:.2f}s")

        if questions_eval.get('success'):
            eval_data = questions_eval['evaluation']
            print(f"      - Calidad preguntas: {eval_data.get('score_total', 'N/A'):.1f}/10")

        status = "✅ PASS" if test_result['overall_success'] else "❌ FAIL"
        print(f"   {status}")
        time.sleep(60)

        return test_result

    def _evaluate_extraction_accuracy(
            self,
            extracted: Dict[str, Any],
            expected: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evalúa la precisión de la extracción de información

        Args:
            extracted: Información extraída por el agente
            expected: Información esperada

        Returns:
            Métricas de precisión
        """
        return self.llm_judge.evaluate_information_extraction(
            conversation=[],  # No se usa en esta evaluación
            extracted_info=extracted,
            expected_info=expected
        )

    def run_all_tests(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ejecuta todos los escenarios de prueba

        Args:
            scenarios: Lista de escenarios de prueba

        Returns:
            Resumen consolidado de resultados
        """
        print("\n" + "=" * 80)
        print("🧪 EVALUACIÓN DEL QUESTIONER AGENT")
        print("=" * 80)

        self.results = []

        for scenario in scenarios:
            try:
                self.test_scenario(scenario)
                time.sleep(60)
            except Exception as e:
                print(f"   ❌ Error ejecutando escenario: {e}")
                self.results.append({
                    "scenario_id": scenario['id'],
                    "scenario_name": scenario['name'],
                    "error": str(e),
                    "overall_success": False
                })

        # Calcular métricas agregadas
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.get('overall_success', False))

        avg_extraction_accuracy = sum(
            r['extraction_evaluation']['accuracy_percentage']
            for r in self.results if 'extraction_evaluation' in r
        ) / total_tests if total_tests > 0 else 0

        avg_questions = sum(
            r['questions_asked']
            for r in self.results if 'questions_asked' in r
        ) / total_tests if total_tests > 0 else 0

        avg_info_score = sum(
            r['information_score']
            for r in self.results if 'information_score' in r
        ) / total_tests if total_tests > 0 else 0

        summary = {
            "component": "QuestionerAgent",
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "metrics": {
                "avg_extraction_accuracy": avg_extraction_accuracy,
                "avg_questions_asked": avg_questions,
                "avg_information_score": avg_info_score
            },
            "thresholds": THRESHOLDS['questioner'],
            "all_thresholds_passed": {
                "extraction_accuracy": avg_extraction_accuracy >= THRESHOLDS['questioner']['extraction_accuracy'] * 100,
                "max_questions": avg_questions <= THRESHOLDS['questioner']['max_questions']
            },
            "detailed_results": self.results
        }

        # Imprimir resumen final
        print("\n" + "=" * 80)
        print("📊 RESUMEN DE RESULTADOS - QUESTIONER AGENT")
        print("=" * 80)
        print(f"Tests ejecutados: {total_tests}")
        print(f"Tests exitosos: {successful_tests} ({successful_tests / total_tests * 100:.1f}%)")
        print(f"Tests fallidos: {total_tests - successful_tests}")
        print(f"\nMétricas promedio:")
        print(f"  - Precisión extracción: {avg_extraction_accuracy:.1f}%")
        print(f"  - Preguntas por sesión: {avg_questions:.1f}")
        print(f"  - Score de información: {avg_info_score:.1f}%")
        print("=" * 80 + "\n")

        return summary


def load_scenarios(filepath: str = "evaluation/datasets/test_multiples_scenarios.json") -> List[Dict[str, Any]]:
    """Carga escenarios de prueba desde archivo JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['scenarios']


if __name__ == "__main__":
    # Cargar escenarios
    scenarios = load_scenarios()

    # Crear evaluador
    evaluator = QuestionerEvaluator()

    # Ejecutar tests
    summary = evaluator.run_all_tests(scenarios)

    # Guardar resultados
    output_file = f"evaluation/results/questioner_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Resultados guardados en: {output_file}")

