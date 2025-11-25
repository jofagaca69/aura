"""
Tests para evaluar el sistema RAG (VectorStore y recuperación)
"""
import json
import time
from typing import Dict, Any, List
from datetime import datetime

from src.rag.vector_store import VectorStore
from evaluation.config import THRESHOLDS


class RAGEvaluator:
    """
    Evaluador especializado para el sistema RAG
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.results = []

    def test_retrieval(
            self,
            query: str,
            expected_keywords: List[str] = None,
            k: int = 5
    ) -> Dict[str, Any]:
        """
        Prueba la recuperación de documentos para una consulta

        Args:
            query: Consulta de búsqueda
            expected_keywords: Palabras clave que deberían aparecer en resultados
            k: Número de resultados a recuperar

        Returns:
            Métricas de evaluación
        """
        print(f"\n   🔍 Consultando: '{query[:60]}...'")

        start_time = time.time()

        # Realizar búsqueda
        try:
            results = self.vector_store.search(query, k=k)
            search_time = time.time() - start_time

            # Analizar resultados
            num_results = len(results)

            # Verificar si contiene palabras clave esperadas
            keyword_matches = 0
            if expected_keywords:
                for result in results:
                    content = str(result.page_content).lower()
                    metadata = str(result.metadata).lower()
                    combined = content + " " + metadata

                    for keyword in expected_keywords:
                        if keyword.lower() in combined:
                            keyword_matches += 1
                            break

                relevance_score = keyword_matches / len(results) if results else 0
            else:
                relevance_score = 1.0 if results else 0.0

            # Calcular scores de similitud
            similarity_scores = []
            for result in results:
                # Si hay score en metadata
                score = result.metadata.get('score', 0.0) if hasattr(result, 'metadata') else 0.0
                similarity_scores.append(score)

            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

            result_data = {
                "query": query,
                "k": k,
                "num_results": num_results,
                "search_time": search_time,
                "relevance_score": relevance_score,
                "avg_similarity": avg_similarity,
                "expected_keywords": expected_keywords or [],
                "keyword_matches": keyword_matches,
                "passed_time_threshold": search_time <= THRESHOLDS['rag']['search_time_max'],
                "results_preview": [
                    {
                        "content": str(r.page_content)[:200] + "...",
                        "metadata": r.metadata if hasattr(r, 'metadata') else {}
                    }
                    for r in results[:3]  # Solo primeros 3 para preview
                ]
            }

            print(f"      - Resultados: {num_results}")
            print(f"      - Tiempo: {search_time:.3f}s")
            print(f"      - Relevancia: {relevance_score:.2f}")

            return result_data

        except Exception as e:
            print(f"      ❌ Error en búsqueda: {e}")
            return {
                "query": query,
                "error": str(e),
                "num_results": 0,
                "search_time": time.time() - start_time,
                "relevance_score": 0.0
            }

    def test_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta un escenario de prueba de búsqueda RAG

        Args:
            scenario: Escenario con consultas de prueba

        Returns:
            Resultados consolidados
        """
        print(f"\n🧪 Testing RAG - Escenario: {scenario['name']}")

        # Generar consulta de búsqueda basada en el escenario
        conversation_text = " ".join(scenario['conversation'])
        expected_keywords = scenario['expected_outcome'].get('relevance_keywords', [])

        # Test 1: Búsqueda con texto completo de conversación
        result_full = self.test_retrieval(
            query=conversation_text,
            expected_keywords=expected_keywords,
            k=10
        )

        # Test 2: Búsqueda con categoría específica
        category = scenario.get('category', '')
        if category and category != 'unclear':
            result_category = self.test_retrieval(
                query=category,
                expected_keywords=[category],
                k=5
            )
        else:
            result_category = None

        # Test 3: Búsqueda con características clave
        expected_extraction = scenario.get('expected_extraction', {})
        features = expected_extraction.get('caracteristicas_clave', [])
        if features:
            features_query = " ".join(features)
            result_features = self.test_retrieval(
                query=features_query,
                expected_keywords=features,
                k=5
            )
        else:
            result_features = None

        # Consolidar resultados
        test_result = {
            "scenario_id": scenario['id'],
            "scenario_name": scenario['name'],
            "timestamp": datetime.now().isoformat(),
            "full_query_test": result_full,
            "category_test": result_category,
            "features_test": result_features,
            "expected_outcome": scenario['expected_outcome'],
            "overall_success": (
                    result_full['num_results'] >= scenario['expected_outcome'].get('min_products', 1) and
                    result_full['search_time'] <= THRESHOLDS['rag']['search_time_max'] and
                    result_full['relevance_score'] >= 0.5
            )
        }

        self.results.append(test_result)

        status = "✅ PASS" if test_result['overall_success'] else "❌ FAIL"
        print(f"   {status}")

        return test_result

    def test_vectorstore_health(self) -> Dict[str, Any]:
        """
        Verifica el estado de salud del VectorStore

        Returns:
            Métricas de salud del sistema
        """
        print("\n🏥 Verificando salud del VectorStore...")

        health = {
            "timestamp": datetime.now().isoformat(),
            "vectorstore_loaded": self.vector_store.vectorstore is not None,
            "checks": {}
        }

        try:
            # Test 1: Verificar que puede realizar búsquedas básicas
            test_query = "test"
            start = time.time()
            results = self.vector_store.search(test_query, k=1)
            query_time = time.time() - start

            health["checks"]["basic_search"] = {
                "passed": True,
                "response_time": query_time,
                "results_returned": len(results)
            }

            print(f"   ✅ Búsqueda básica: OK ({query_time:.3f}s)")

        except Exception as e:
            health["checks"]["basic_search"] = {
                "passed": False,
                "error": str(e)
            }
            print(f"   ❌ Búsqueda básica: FAIL - {e}")

        try:
            # Test 2: Verificar diversidad de documentos
            results = self.vector_store.search("producto", k=10)
            unique_sources = set()

            for r in results:
                if hasattr(r, 'metadata') and 'source' in r.metadata:
                    unique_sources.add(r.metadata['source'])

            health["checks"]["document_diversity"] = {
                "passed": len(unique_sources) > 0,
                "unique_sources": len(unique_sources),
                "total_results": len(results)
            }

            print(f"   ✅ Diversidad: {len(unique_sources)} fuentes únicas")

        except Exception as e:
            health["checks"]["document_diversity"] = {
                "passed": False,
                "error": str(e)
            }
            print(f"   ❌ Diversidad: FAIL - {e}")

        # Overall health
        all_passed = all(
            check.get("passed", False)
            for check in health["checks"].values()
        )
        health["overall_health"] = "HEALTHY" if all_passed else "UNHEALTHY"

        return health

    def run_all_tests(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ejecuta todos los tests RAG

        Args:
            scenarios: Lista de escenarios

        Returns:
            Resumen consolidado
        """
        print("\n" + "=" * 80)
        print("🧪 EVALUACIÓN DEL SISTEMA RAG")
        print("=" * 80)

        # Verificar salud del sistema primero
        health = self.test_vectorstore_health()

        if health["overall_health"] != "HEALTHY":
            print("\n⚠️  WARNING: VectorStore no está saludable. Los tests pueden fallar.")

        # Ejecutar tests de escenarios
        self.results = []

        for scenario in scenarios:
            try:
                self.test_scenario(scenario)
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

        # Métricas de búsqueda
        all_search_times = []
        all_relevance_scores = []
        all_results_counts = []

        for result in self.results:
            if 'full_query_test' in result and 'error' not in result['full_query_test']:
                all_search_times.append(result['full_query_test']['search_time'])
                all_relevance_scores.append(result['full_query_test']['relevance_score'])
                all_results_counts.append(result['full_query_test']['num_results'])

        avg_search_time = sum(all_search_times) / len(all_search_times) if all_search_times else 0
        avg_relevance = sum(all_relevance_scores) / len(all_relevance_scores) if all_relevance_scores else 0
        avg_results = sum(all_results_counts) / len(all_results_counts) if all_results_counts else 0

        summary = {
            "component": "RAG",
            "timestamp": datetime.now().isoformat(),
            "vectorstore_health": health,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "metrics": {
                "avg_search_time": avg_search_time,
                "avg_relevance_score": avg_relevance,
                "avg_results_count": avg_results
            },
            "thresholds": THRESHOLDS['rag'],
            "all_thresholds_passed": {
                "search_time": avg_search_time <= THRESHOLDS['rag']['search_time_max'],
                "relevance": avg_relevance >= 0.5
            },
            "detailed_results": self.results
        }

        # Imprimir resumen
        print("\n" + "=" * 80)
        print("📊 RESUMEN DE RESULTADOS - RAG")
        print("=" * 80)
        print(f"Tests ejecutados: {total_tests}")
        print(f"Tests exitosos: {successful_tests} ({successful_tests / total_tests * 100:.1f}%)")
        print(f"Tests fallidos: {total_tests - successful_tests}")
        print(f"\nMétricas promedio:")
        print(f"  - Tiempo de búsqueda: {avg_search_time:.3f}s")
        print(f"  - Relevancia: {avg_relevance:.2f}")
        print(f"  - Resultados por consulta: {avg_results:.1f}")
        print("=" * 80 + "\n")

        return summary


def load_scenarios(filepath: str = "evaluation/datasets/test_multiples_scenarios.json") -> List[Dict[str, Any]]:
    """Carga escenarios de prueba"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['scenarios']


if __name__ == "__main__":
    # Cargar VectorStore
    print("🚀 Cargando VectorStore...")
    vector_store = VectorStore()
    vector_store.load_vectorstore()

    # Cargar escenarios
    scenarios = load_scenarios()

    # Crear evaluador
    evaluator = RAGEvaluator(vector_store)

    # Ejecutar tests
    summary = evaluator.run_all_tests(scenarios)

    # Guardar resultados
    output_file = f"evaluation/results/rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Resultados guardados en: {output_file}")

