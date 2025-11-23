"""
Script principal para ejecutar toda la suite de evaluación de AURA
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.vector_store import VectorStore
from src.config import config

from evaluation.test_questioner import QuestionerEvaluator, load_scenarios as load_scenarios_q
from evaluation.test_rag import RAGEvaluator, load_scenarios as load_scenarios_rag
from evaluation.test_orchestrator import OrchestratorEvaluator, load_scenarios as load_scenarios_orch
from evaluation.report_generator import ReportGenerator


class AURAEvaluationSuite:
    """
    Suite completa de evaluación para el sistema AURA
    """

    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store
        self.results = {}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def setup(self):
        """Configura el entorno de evaluación"""
        print("\n" + "=" * 80)
        print("🚀 SUITE DE EVALUACIÓN AURA")
        print("=" * 80)
        print(f"📅 Timestamp: {self.timestamp}")
        print(f"🔧 Configurando entorno...")

        # Validar configuración
        try:
            config.validate()
            print("   ✅ Configuración validada")
        except Exception as e:
            print(f"   ❌ Error en configuración: {e}")
            return False

        # Cargar VectorStore si no se proporcionó
        if self.vector_store is None:
            try:
                print("   📚 Cargando VectorStore...")
                self.vector_store = VectorStore()
                self.vector_store.load_vectorstore()
                print("   ✅ VectorStore cargado")
            except Exception as e:
                print(f"   ❌ Error cargando VectorStore: {e}")
                return False

        return True

    def run_questioner_tests(self) -> bool:
        """Ejecuta tests del QuestionerAgent"""
        print("\n" + "=" * 80)
        print("🧪 FASE 1: Evaluación del QuestionerAgent")
        print("=" * 80)

        try:
            scenarios = load_scenarios_q()
            evaluator = QuestionerEvaluator()
            self.results['questioner'] = evaluator.run_all_tests(scenarios)

            # Guardar resultados
            output_file = f"evaluation/results/questioner_{self.timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results['questioner'], f, indent=2, ensure_ascii=False)

            print(f"\n✅ Resultados guardados: {output_file}")
            return True

        except Exception as e:
            print(f"\n❌ Error en tests de QuestionerAgent: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_rag_tests(self) -> bool:
        """Ejecuta tests del sistema RAG"""
        print("\n" + "=" * 80)
        print("🧪 FASE 2: Evaluación del Sistema RAG")
        print("=" * 80)

        try:
            scenarios = load_scenarios_rag()
            evaluator = RAGEvaluator(self.vector_store)
            self.results['rag'] = evaluator.run_all_tests(scenarios)

            # Guardar resultados
            output_file = f"evaluation/results/rag_{self.timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results['rag'], f, indent=2, ensure_ascii=False)

            print(f"\n✅ Resultados guardados: {output_file}")
            return True

        except Exception as e:
            print(f"\n❌ Error en tests de RAG: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_orchestrator_tests(self) -> bool:
        """Ejecuta tests end-to-end del Orchestrator"""
        print("\n" + "=" * 80)
        print("🧪 FASE 3: Evaluación End-to-End del Orchestrator")
        print("=" * 80)

        try:
            scenarios = load_scenarios_orch()
            evaluator = OrchestratorEvaluator(self.vector_store)
            self.results['orchestrator'] = evaluator.run_all_tests(scenarios)

            # Guardar resultados
            output_file = f"evaluation/results/orchestrator_{self.timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results['orchestrator'], f, indent=2, ensure_ascii=False)

            print(f"\n✅ Resultados guardados: {output_file}")
            return True

        except Exception as e:
            print(f"\n❌ Error en tests de Orchestrator: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_report(self) -> str:
        """Genera reporte HTML consolidado"""
        print("\n" + "=" * 80)
        print("📊 Generando Reporte HTML")
        print("=" * 80)

        try:
            generator = ReportGenerator()
            output_file = f"evaluation/results/report_{self.timestamp}.html"

            report_path = generator.generate_full_report(
                questioner_results=self.results.get('questioner'),
                rag_results=self.results.get('rag'),
                orchestrator_results=self.results.get('orchestrator'),
                output_path=output_file
            )

            print(f"\n✅ Reporte HTML generado: {report_path}")
            return report_path

        except Exception as e:
            print(f"\n❌ Error generando reporte: {e}")
            import traceback
            traceback.print_exc()
            return None

    def print_summary(self):
        """Imprime resumen final de resultados"""
        print("\n" + "=" * 80)
        print("📊 RESUMEN FINAL DE EVALUACIÓN")
        print("=" * 80)

        total_components = len(self.results)

        print(f"\n🔍 Componentes evaluados: {total_components}")

        for component_name, component_results in self.results.items():
            total = component_results.get('total_tests', 0)
            success = component_results.get('successful_tests', 0)
            rate = (success / total * 100) if total > 0 else 0

            status = "✅" if rate >= 80 else "⚠️" if rate >= 60 else "❌"

            print(f"\n{status} {component_name.upper()}:")
            print(f"   Tests: {success}/{total} ({rate:.1f}%)")

            # Métricas específicas
            if component_name == 'questioner':
                metrics = component_results.get('metrics', {})
                print(f"   Precisión extracción: {metrics.get('avg_extraction_accuracy', 0):.1f}%")
                print(f"   Preguntas promedio: {metrics.get('avg_questions_asked', 0):.1f}")

            elif component_name == 'rag':
                metrics = component_results.get('metrics', {})
                print(f"   Tiempo búsqueda: {metrics.get('avg_search_time', 0):.3f}s")
                print(f"   Relevancia: {metrics.get('avg_relevance_score', 0):.2f}")

            elif component_name == 'orchestrator':
                metrics = component_results.get('metrics', {})
                exec_time = metrics.get('execution_time', {})
                llm_scores = metrics.get('llm_judge_scores', {})
                print(f"   Tiempo promedio: {exec_time.get('avg', 0):.2f}s")
                print(f"   Calidad LLM: {llm_scores.get('avg', 0):.1f}/10")

        # Calcular score general
        all_rates = []
        for component_results in self.results.values():
            total = component_results.get('total_tests', 0)
            success = component_results.get('successful_tests', 0)
            if total > 0:
                all_rates.append(success / total * 100)

        overall_rate = sum(all_rates) / len(all_rates) if all_rates else 0

        print(f"\n{'=' * 80}")
        print(f"🎯 SCORE GENERAL DEL SISTEMA: {overall_rate:.1f}%")

        if overall_rate >= 90:
            print("   🌟 EXCELENTE - Sistema funcionando óptimamente")
        elif overall_rate >= 75:
            print("   ✅ BUENO - Sistema funcionando correctamente")
        elif overall_rate >= 60:
            print("   ⚠️  ACEPTABLE - Algunas áreas necesitan mejora")
        else:
            print("   ❌ NECESITA MEJORAS - Revisar componentes fallidos")

        print(f"{'=' * 80}\n")

    def run_all(self, components: list = None) -> bool:
        """
        Ejecuta toda la suite de evaluación

        Args:
            components: Lista de componentes a evaluar ['questioner', 'rag', 'orchestrator']
                       Si es None, evalúa todos

        Returns:
            True si todos los tests se ejecutaron exitosamente
        """
        if components is None:
            components = ['questioner', 'rag', 'orchestrator']

        # Setup
        if not self.setup():
            print("\n❌ Error en configuración inicial. Abortando.")
            return False

        success = True

        # Ejecutar tests según componentes especificados
        if 'questioner' in components:
            if not self.run_questioner_tests():
                success = False

        if 'rag' in components:
            if not self.run_rag_tests():
                success = False

        if 'orchestrator' in components:
            if not self.run_orchestrator_tests():
                success = False

        # Generar reporte
        report_path = self.generate_report()

        # Imprimir resumen
        self.print_summary()

        # Información final
        print("\n📁 Archivos generados:")
        print(f"   - Resultados JSON: evaluation/results/*_{self.timestamp}.json")
        if report_path:
            print(f"   - Reporte HTML: {report_path}")

        print("\n" + "=" * 80)
        print("✅ Evaluación completada" if success else "⚠️  Evaluación completada con errores")
        print("=" * 80 + "\n")

        return success


def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Suite de Evaluación AURA - Sistema Multi-Agente',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python run_evaluation.py                    # Evaluar todos los componentes
  python run_evaluation.py -c questioner      # Solo QuestionerAgent
  python run_evaluation.py -c rag orchestrator # RAG y Orchestrator
  python run_evaluation.py --quick            # Evaluación rápida (menos tests)
        """
    )

    parser.add_argument(
        '-c', '--components',
        nargs='+',
        choices=['questioner', 'rag', 'orchestrator'],
        help='Componentes específicos a evaluar'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Evaluación rápida (reduce número de tests)'
    )

    parser.add_argument(
        '--no-report',
        action='store_true',
        help='No generar reporte HTML'
    )

    args = parser.parse_args()

    # Crear suite
    suite = AURAEvaluationSuite()

    # Ejecutar evaluación
    components = args.components if args.components else None
    success = suite.run_all(components=components)

    # Código de salida
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

