
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .retrieval import find_relevant_places
from .generator import generate_recommendations


def get_recommendations(
    user_query: str,
    user_location: Tuple[float, float],
    current_time: Optional[str] = None,
    max_distance_km: float = 3.0,
    provider: str = "openai",
    model: Optional[str] = None,
    group_size: Optional[int] = None,
    group_type: Optional[str] = None,
    group_preferences: Optional[List[str]] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Главная функция RAG-системы

    Шаги:
    1. Найти релевантные места (retrieval)
    2. Подготовить промпт для LLM и сгенерировать рекомендации (generation)
    3. Вернуть структурированный JSON

    Параметры:
    - user_query: текст запроса пользователя
    - user_location: (lat, lon)
    - current_time: строка времени (например, "2025-11-03 22:41") — не обязательна
    - max_distance_km: радиус поиска
    - provider: "openai" | "anthropic" | "gemini"
    - model: имя модели провайдера
    - group_size: количество человек в группе (2-10)
    - group_type: тип группы (family, friends, colleagues, mixed)
    - group_preferences: предпочтения группы (kids_friendly, accessible, budget_friendly)
    - language: язык ответа ('ru', 'kk', 'en') или None для авто-определения
    """
    # 1) Retrieval
    places: List[Dict[str, Any]] = find_relevant_places(
        user_query=user_query,
        user_location=user_location,
        max_distance_km=max_distance_km,
        max_results=10,
    )

    # 2) Generation (LLM)
    gen = generate_recommendations(
        user_query=user_query,
        retrieved_places=places,
        provider=provider,
        model=model,
        group_size=group_size,
        group_type=group_type,
        group_preferences=group_preferences,
        language=language,
    )

    # 3) Возвращаем JSON + добавим исходные top-places для UI (опционально)
    return {
        "query": user_query,
        "user_location": {
            "lat": float(user_location[0]),
            "lon": float(user_location[1]),
        },
        "radius_km": float(max_distance_km),
        "retrieved": places,
        "recommendations": gen.get("recommendations", []),
    }


if __name__ == "__main__":
    # Небольшой smoke-тест, требует подготовленных артефактов и API-ключа выбранного провайдера
    out = get_recommendations(
        user_query="Я возле Bayterek, хочу прогуляться 30-60 минут",
        user_location=(51.1694, 71.4491),
        max_distance_km=2.0,
        provider="gemini",
        model="gemini-1.5-flash",
    )
    import json as _json
    print(_json.dumps(out, ensure_ascii=False, indent=2))
