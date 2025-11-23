"""
Generador de reportes HTML para los resultados de evaluación
"""
import json
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path


class ReportGenerator:
    """
    Genera reportes HTML profesionales con los resultados de evaluación
    """

    def __init__(self):
        self.template = self._load_template()

    def _load_template(self) -> str:
        """Carga plantilla HTML base"""
        return """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Evaluación AURA - {timestamp}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            color: #1f2937;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 1rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}

        .header .subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}

        .content {{
            padding: 2rem;
        }}

        .section {{
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: #f9fafb;
            border-radius: 0.75rem;
            border-left: 4px solid #667eea;
        }}

        .section h2 {{
            color: #667eea;
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }}

        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}

        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}

        .metric-label {{
            font-size: 0.875rem;
            color: #6b7280;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #1f2937;
        }}

        .metric-value.success {{
            color: #10b981;
        }}

        .metric-value.warning {{
            color: #f59e0b;
        }}

        .metric-value.error {{
            color: #ef4444;
        }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
        }}

        .test-list {{
            list-style: none;
        }}

        .test-item {{
            background: white;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-radius: 0.5rem;
            border-left: 4px solid #e5e7eb;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .test-item.pass {{
            border-left-color: #10b981;
        }}

        .test-item.fail {{
            border-left-color: #ef4444;
        }}

        .test-name {{
            font-weight: 500;
            flex: 1;
        }}

        .test-status {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
        }}

        .test-status.pass {{
            background: #d1fae5;
            color: #065f46;
        }}

        .test-status.fail {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .threshold-check {{
            display: flex;
            align-items: center;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            background: white;
            border-radius: 0.5rem;
        }}

        .threshold-check .icon {{
            font-size: 1.5rem;
            margin-right: 0.75rem;
        }}

        .threshold-check .label {{
            flex: 1;
            font-weight: 500;
        }}

        .summary-box {{
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }}

        .summary-box h3 {{
            color: #667eea;
            margin-bottom: 1rem;
        }}

        .summary-box p {{
            line-height: 1.6;
            color: #4b5563;
            margin-bottom: 0.5rem;
        }}

        .component-section {{
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 2px solid #e5e7eb;
        }}

        .component-section h3 {{
            color: #1f2937;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
        }}

        .badge.success {{
            background: #d1fae5;
            color: #065f46;
        }}

        .badge.warning {{
            background: #fef3c7;
            color: #92400e;
        }}

        .badge.error {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .footer {{
            background: #f9fafb;
            padding: 1.5rem;
            text-align: center;
            color: #6b7280;
            font-size: 0.875rem;
            border-top: 1px solid #e5e7eb;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}

        th {{
            background: #f3f4f6;
            padding: 0.75rem;
            text-align: left;
            font-weight: 600;
            color: #374151;
        }}

        td {{
            padding: 0.75rem;
            border-bottom: 1px solid #e5e7eb;
        }}

        tr:hover {{
            background: #f9fafb;
        }}

        .chart-container {{
            margin-top: 1rem;
            padding: 1rem;
            background: white;
            border-radius: 0.5rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AURA - Reporte de Evaluación</h1>
            <div class="subtitle">Sistema Multi-Agente de Recomendación de Productos</div>
            <div class="subtitle" style="margin-top: 0.5rem; font-size: 0.9rem;">
                Generado: {timestamp}
            </div>
        </div>

        <div class="content">
            {content}
        </div>

        <div class="footer">
            <p>🚀 AURA Evaluation System</p>
            <p>Powered by Google Gemini, LangChain & LangGraph</p>
            <p style="margin-top: 0.5rem;">© 2025 José Fabián García Camargo</p>
        </div>
    </div>
</body>
</html>
        """

    def generate_full_report(
            self,
            questioner_results: Dict[str, Any] = None,
            rag_results: Dict[str, Any] = None,
            orchestrator_results: Dict[str, Any] = None,
            output_path: str = None
    ) -> str:
        """
        Genera un reporte HTML completo con todos los resultados

        Args:
            questioner_results: Resultados del QuestionerAgent
            rag_results: Resultados del RAG
            orchestrator_results: Resultados del Orchestrator
            output_path: Ruta donde guardar el HTML

        Returns:
            Path del archivo generado
        """
        content_sections = []

        # Sección de resumen ejecutivo
        content_sections.append(self._generate_executive_summary(
            questioner_results, rag_results, orchestrator_results
        ))

        # Sección de QuestionerAgent
        if questioner_results:
            content_sections.append(self._generate_questioner_section(questioner_results))

        # Sección de RAG
        if rag_results:
            content_sections.append(self._generate_rag_section(rag_results))

        # Sección de Orchestrator
        if orchestrator_results:
            content_sections.append(self._generate_orchestrator_section(orchestrator_results))

        # Combinar todo
        content = "\n".join(content_sections)

        html = self.template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content=content
        )

        # Guardar archivo
        if output_path is None:
            output_path = f"evaluation/results/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return output_path

    def _generate_executive_summary(
            self,
            questioner_results: Dict[str, Any],
            rag_results: Dict[str, Any],
            orchestrator_results: Dict[str, Any]
    ) -> str:
        """Genera la sección de resumen ejecutivo"""

        # Calcular métricas generales
        all_results = [r for r in [questioner_results, rag_results, orchestrator_results] if r]

        total_tests = sum(r.get('total_tests', 0) for r in all_results)
        total_success = sum(r.get('successful_tests', 0) for r in all_results)
        overall_rate = (total_success / total_tests * 100) if total_tests > 0 else 0

        # Determinar estado general
        if overall_rate >= 90:
            status_class = "success"
            status_badge = "EXCELENTE"
        elif overall_rate >= 75:
            status_class = "success"
            status_badge = "BUENO"
        elif overall_rate >= 60:
            status_class = "warning"
            status_badge = "ACEPTABLE"
        else:
            status_class = "error"
            status_badge = "NECESITA MEJORAS"

        return f"""
        <div class="section">
            <h2>📊 Resumen Ejecutivo</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Componentes Evaluados</div>
                    <div class="metric-value">{len(all_results)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Tests Totales</div>
                    <div class="metric-value">{total_tests}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Tests Exitosos</div>
                    <div class="metric-value success">{total_success}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Tasa de Éxito</div>
                    <div class="metric-value {status_class}">{overall_rate:.1f}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {overall_rate}%"></div>
                    </div>
                </div>
            </div>

            <div class="summary-box">
                <h3>Estado General del Sistema: <span class="badge {status_class}">{status_badge}</span></h3>
                <p>
                    El sistema AURA ha completado <strong>{total_tests} tests de evaluación</strong> 
                    a través de sus {len(all_results)} componentes principales. 
                    Se alcanzó una tasa de éxito del <strong>{overall_rate:.1f}%</strong>.
                </p>
            </div>
        </div>
        """

    def _generate_questioner_section(self, results: Dict[str, Any]) -> str:
        """Genera sección de resultados del QuestionerAgent"""

        metrics = results.get('metrics', {})
        total = results.get('total_tests', 0)
        success = results.get('successful_tests', 0)
        rate = results.get('success_rate', 0) * 100

        # Tests individuales
        test_rows = []
        for test in results.get('detailed_results', [])[:10]:  # Primeros 10
            status_class = "pass" if test.get('overall_success') else "fail"
            status_text = "PASS" if test.get('overall_success') else "FAIL"

            extraction_acc = test.get('extraction_evaluation', {}).get('accuracy_percentage', 0)
            questions = test.get('questions_asked', 0)

            test_rows.append(f"""
                <tr>
                    <td>{test.get('scenario_name', 'N/A')}</td>
                    <td>{extraction_acc:.1f}%</td>
                    <td>{questions}</td>
                    <td><span class="test-status {status_class}">{status_text}</span></td>
                </tr>
            """)

        return f"""
        <div class="section">
            <h2>🔍 QuestionerAgent - Recopilación de Información</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Tasa de Éxito</div>
                    <div class="metric-value {'success' if rate >= 85 else 'warning'}">{rate:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Precisión Extracción</div>
                    <div class="metric-value">{metrics.get('avg_extraction_accuracy', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Preguntas Promedio</div>
                    <div class="metric-value">{metrics.get('avg_questions_asked', 0):.1f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Score Información</div>
                    <div class="metric-value">{metrics.get('avg_information_score', 0):.1f}%</div>
                </div>
            </div>

            <div class="component-section">
                <h3>📋 Resultados de Tests</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Escenario</th>
                            <th>Precisión</th>
                            <th>Preguntas</th>
                            <th>Estado</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(test_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        """

    def _generate_rag_section(self, results: Dict[str, Any]) -> str:
        """Genera sección de resultados del RAG"""

        metrics = results.get('metrics', {})
        health = results.get('vectorstore_health', {})
        rate = results.get('success_rate', 0) * 100

        health_status = health.get('overall_health', 'UNKNOWN')
        health_class = 'success' if health_status == 'HEALTHY' else 'error'

        return f"""
        <div class="section">
            <h2>📚 Sistema RAG - Recuperación de Información</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Estado VectorStore</div>
                    <div class="metric-value {health_class}">{health_status}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Tasa de Éxito</div>
                    <div class="metric-value {'success' if rate >= 80 else 'warning'}">{rate:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Tiempo Búsqueda</div>
                    <div class="metric-value">{metrics.get('avg_search_time', 0):.3f}s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Relevancia Promedio</div>
                    <div class="metric-value">{metrics.get('avg_relevance_score', 0):.2f}</div>
                </div>
            </div>
        </div>
        """

    def _generate_orchestrator_section(self, results: Dict[str, Any]) -> str:
        """Genera sección de resultados del Orchestrator"""

        metrics = results.get('metrics', {})
        rate = results.get('success_rate', 0) * 100
        by_difficulty = results.get('by_difficulty', {})

        # Métricas por dificultad
        difficulty_rows = []
        for diff, stats in by_difficulty.items():
            diff_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            difficulty_rows.append(f"""
                <tr>
                    <td style="text-transform: capitalize;">{diff}</td>
                    <td>{stats['total']}</td>
                    <td>{stats['passed']}</td>
                    <td>{diff_rate:.1f}%</td>
                </tr>
            """)

        exec_metrics = metrics.get('execution_time', {})
        llm_metrics = metrics.get('llm_judge_scores', {})

        return f"""
        <div class="section">
            <h2>🎯 Orchestrator - Flujo End-to-End</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Tasa de Éxito</div>
                    <div class="metric-value {'success' if rate >= 90 else 'warning'}">{rate:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Tiempo Promedio</div>
                    <div class="metric-value">{exec_metrics.get('avg', 0):.2f}s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Productos (Promedio)</div>
                    <div class="metric-value">{metrics.get('products_found', {}).get('avg', 0):.1f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Calidad LLM Judge</div>
                    <div class="metric-value {'success' if llm_metrics.get('avg', 0) >= 8 else 'warning'}">{llm_metrics.get('avg', 0):.1f}/10</div>
                </div>
            </div>

            <div class="component-section">
                <h3>📊 Rendimiento por Dificultad</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Dificultad</th>
                            <th>Total Tests</th>
                            <th>Exitosos</th>
                            <th>Tasa</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(difficulty_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        """


def generate_report_from_files(
        questioner_file: str = None,
        rag_file: str = None,
        orchestrator_file: str = None,
        output_file: str = None
) -> str:
    """
    Genera reporte HTML desde archivos JSON de resultados

    Args:
        questioner_file: Path al JSON de resultados del QuestionerAgent
        rag_file: Path al JSON de resultados del RAG
        orchestrator_file: Path al JSON de resultados del Orchestrator
        output_file: Path donde guardar el HTML

    Returns:
        Path del archivo HTML generado
    """
    generator = ReportGenerator()

    # Cargar resultados
    questioner_results = None
    rag_results = None
    orchestrator_results = None

    if questioner_file and Path(questioner_file).exists():
        with open(questioner_file, 'r', encoding='utf-8') as f:
            questioner_results = json.load(f)

    if rag_file and Path(rag_file).exists():
        with open(rag_file, 'r', encoding='utf-8') as f:
            rag_results = json.load(f)

    if orchestrator_file and Path(orchestrator_file).exists():
        with open(orchestrator_file, 'r', encoding='utf-8') as f:
            orchestrator_results = json.load(f)

    # Generar reporte
    return generator.generate_full_report(
        questioner_results=questioner_results,
        rag_results=rag_results,
        orchestrator_results=orchestrator_results,
        output_path=output_file
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Generar desde archivos específicos
        output = generate_report_from_files(
            questioner_file=sys.argv[1] if len(sys.argv) > 1 else None,
            rag_file=sys.argv[2] if len(sys.argv) > 2 else None,
            orchestrator_file=sys.argv[3] if len(sys.argv) > 3 else None
        )
        print(f"✅ Reporte generado: {output}")
    else:
        print("Uso: python report_generator.py [questioner.json] [rag.json] [orchestrator.json]")

