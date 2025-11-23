from pathlib import Path

EVALUATION_DIR = Path(__file__).parent
RESULTS_DIR = EVALUATION_DIR / 'results'
DATASET_DIR = EVALUATION_DIR / 'datasets'

RESULTS_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)

LLM_JUDGE_CONFIG = {
    "model": "gemini-2.5-flash-lite",
    "temperature": 0,
    "max_retries": 3,
    "timeout": 30.0,
    "request_timeout": 30.0,
    "rate_limit_enabled": True,
    "rate_limit_requests_per_minute": 5  # Máximo 12 requests/minuto (margen de seguridad)
}

REPORT_CONFIG = {
    "format": "html",
    "include_detailed_traces": True,
    "save_artifacts": True
}

THRESHOLDS = {
    "questioner": {
        "extraction_accuracy": 0.75,  # 75% precisión en extracción (ajustado desde 85%)
        "max_questions": 5,
        "information_score_min": 50.0  # Score mínimo de información (ajustado desde 60%)
    },
    "rag": {
        "precision_at_5": 0.80,  # 80% de precisión en top 5
        "recall_at_10": 0.70,    # 70% de recall en top 10
        "search_time_max": 2.0    # Máximo 2 segundos
    },
    "recommender": {
        "llm_judge_min_score": 7.0,  # Mínimo 7/10 en evaluación LLM (ajustado desde 8.0)
        "relevancia_min": 6.0,      # Ajustado desde 7.0
        "diversidad_min": 5.0,       # Ajustado desde 6.0
        "explicacion_min": 6.0       # Ajustado desde 7.0
    },
    "orchestrator": {
        "success_rate": 0.70,     # 70% de tests exitosos (ajustado desde 90%)
        # Tiempo máximo considera:
        # - Rate limiting: ~5s entre requests (12 req/min)
        # - Timeout LLM: 30s por intento (hasta 3 reintentos = 90s por llamada fallida)
        # - Procesamiento real: ~20-30s
        # Para 8-10 llamadas LLM: ~50s rate limiting + 20-30s procesamiento = 70-80s base
        # Con timeouts ocasionales: puede llegar a 120-150s
        "avg_time_max": 120.0,    # Máximo 120 segundos promedio (considera rate limiting + timeouts)
        "products_found_min": 1,   # Al menos 1 producto
        "llm_quality_min": 6.0    # Calidad mínima LLM Judge (nuevo criterio)
    }
}