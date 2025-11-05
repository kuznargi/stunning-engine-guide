from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


def build_system_prompt() -> str:
    return (
        "Вы — опытный местный гид Астаны, который помогает людям найти лучшие места поблизости. "
        "Вы даёте 1–3 конкретные рекомендации на основе входного списка релевантных мест и запроса пользователя. "
        "Говорите понятно, дружелюбно и по делу. Не придумывайте факты — если данных нет, честно укажите это. "
        "Учитывайте расстояние, категорию, возможные ограничения по времени работы и предпочтения из запроса. "
        "Отдавайте результат строго в формате JSON согласно схеме."
    )


def _compact_place_item(place: Dict[str, Any]) -> Dict[str, Any]:
    """Выбирает ключевые поля места для передачи в LLM, чтобы сократить токены."""
    return {
        "name": place.get("name"),
        "category": place.get("category"),
        "subcategory": place.get("subcategory"),
        "address": place.get("address"),
        "district": place.get("district"),
        "distance": place.get("distance_text"),
        "working_hours": place.get("working_hours") or "Рекомендуем уточнить время работы",
        "website": place.get("website"),
        "instagram": place.get("instagram"),
        "phone": place.get("phone"),
        "open_now": place.get("open_now"),
        "popularity_score": place.get("popularity_score"),
        # Для семантики можно добавить краткое описание
        "description": place.get("description"),
    }


def build_user_prompt(user_query: str, places: List[Dict[str, Any]]) -> str:
    """Формирует пользовательский промпт с контекстом релевантных мест."""
    header = (
        "Входные данные:\n"
        f"Запрос пользователя: {user_query}\n"
        "Релевантные места (5-10 шт):\n"
    )
    lines = [header]

    for i, p in enumerate(places, 1):
        pp = _compact_place_item(p)
        part = [
            f"{i}. {pp['name']} ({pp.get('category') or 'Категория не указана'})",
        ]
        if pp.get("subcategory"):
            part.append(f"   Подкатегория: {pp['subcategory']}")
        if pp.get("address"):
            part.append(f"   Адрес: {pp['address']}")
        if pp.get("district"):
            part.append(f"   Район: {pp['district']}")
        if pp.get("distance"):
            part.append(f"   Расстояние: {pp['distance']}")
        if pp.get("working_hours"):
            part.append(f"   Время работы: {pp['working_hours']}")
        if pp.get("description"):
            # Коротко, обрежем до 240 символов
            desc = str(pp.get("description"))[:240]
            part.append(f"   Описание: {desc}")
        lines.append("\n".join(part))

    # Инструкции к формату ответа
    lines.append(
        "\nЗадача:\n"
        "- Сформируй 1–3 рекомендации (в зависимости от уместности).\n"
        "- Для каждой рекомендации укажи: 'name', 'category', 'distance', 'why', 'action_plan', 'estimated_time', 'working_hours', 'confidence'.\n"
        "- Строго следуй JSON-схеме ниже и не добавляй поясняющий текст вне JSON.\n"
    )

    lines.append(
        "JSON-схема:\n"
        "{\n"
        "  \"recommendations\": [\n"
        "    {\n"
        "      \"name\": str,\n"
        "      \"category\": str,\n"
        "      \"distance\": str,\n"
        "      \"why\": str,\n"
        "      \"action_plan\": str,\n"
        "      \"estimated_time\": str,\n"
        "      \"working_hours\": str,\n"
        "      \"confidence\": float\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Где: distance — человеко‑читаемый формат (например, '800м, ~10 мин пешком' или '2.1 км, ~25 мин пешком').\n"
        "Если нет данных по времени работы — запиши 'Рекомендуем уточнить время работы'.\n"
        "Не выдумывай лишних деталей: если чего-то нет в списке мест — не придумывай.\n"
    )

    return "\n".join(lines)


# =============================
# Вызов LLM провайдера
# =============================

class LLMProviderError(RuntimeError):
    pass


def _call_openai(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
    """Вызывает OpenAI Chat Completions с JSON-режимом.
    Возвращает текст ответа (ожидается JSON-строка).
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise LLMProviderError("Пакет openai не установлен: pip install openai>=1.0.0") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMProviderError("Не найден OPENAI_API_KEY в окружении")

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},  # строгий JSON
        )
        content = resp.choices[0].message.content or "{}"
        return content
    except Exception as e:  # pragma: no cover
        raise LLMProviderError(f"Ошибка вызова OpenAI: {e}") from e


def _call_anthropic(messages: List[Dict[str, str]], model: str = "claude-3-5-sonnet-latest") -> str:
    """Вызывает Anthropic Messages API и старается получить ответ в JSON.
    Возвращает текст ответа.
    """
    try:
        import anthropic  # type: ignore
    except Exception as e:  # pragma: no cover
        raise LLMProviderError("Пакет anthropic не установлен: pip install anthropic") from e

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMProviderError("Не найден ANTHROPIC_API_KEY в окружении")

    client = anthropic.Anthropic(api_key=api_key)
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
    user_parts = [m["content"] for m in messages if m["role"] == "user"]

    try:
        resp = client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": "\n\n".join(user_parts)}],
            temperature=0.3,
            max_tokens=1200,
        )
        # Собираем текстовый контент
        out_text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                out_text += getattr(block, "text", "")
        return out_text.strip() or "{}"
    except Exception as e:  # pragma: no cover
        raise LLMProviderError(f"Ошибка вызова Anthropic: {e}") from e


def _call_gemini(messages: List[Dict[str, str]], model: str = "gemini-1.5-flash") -> str:
    """Вызывает Google Gemini (google-generativeai). Ожидается JSON-строка в ответе.
    Требуется переменная окружения GOOGLE_API_KEY. Установка пакета: pip install google-generativeai
    Проверяет доступность модели через list_models() и выбирает рабочую версию.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:  # pragma: no cover
        raise LLMProviderError("Пакет google-generativeai не установлен: pip install google-generativeai") from e

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMProviderError("Не найден GOOGLE_API_KEY (или GEMINI_API_KEY) в окружении")

    genai.configure(api_key=api_key)

    # Получаем список доступных моделей с поддержкой generateContent
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # Убираем префикс "models/" если он есть
                model_name = m.name.replace("models/", "")
                available_models.append(model_name)
    except Exception as e:  # pragma: no cover
        # Если не удалось получить список моделей, используем резервный список
        available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

    # Объединяем system и все user-сообщения в один промпт.
    system_text = "\n".join([m["content"] for m in messages if m["role"] == "system"]).strip()
    user_text = "\n\n".join([m["content"] for m in messages if m["role"] == "user"]).strip()

    prompt = (
        (system_text + "\n\n") if system_text else ""
        ) + user_text + "\n\nВыведи строго валидный JSON согласно указанной схеме без пояснений вне JSON."

    # Нормализуем имя запрошенной модели (убираем префикс models/ и суффиксы -latest)
    requested_model = model.replace("models/", "").replace("-latest", "")

    # Порядок попыток моделей: сначала запрошенная, затем доступные, затем резервные
    candidates = [requested_model]

    # Добавляем доступные модели (gemini-1.5-flash в приоритете)
    for am in available_models:
        if am not in candidates:
            if "1.5-flash" in am:
                candidates.insert(1, am)  # flash модели в начало
            else:
                candidates.append(am)

    # Добавляем резервные варианты на всякий случай
    for fallback in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
        if fallback not in candidates:
            candidates.append(fallback)

    last_error: Optional[Exception] = None
    tried_models = []

    for m in candidates:
        try:
            # Пробуем без префикса "models/"
            model_obj = genai.GenerativeModel(m)
            resp = model_obj.generate_content(prompt, generation_config={
                "temperature": 0.3,
            })
            text = (getattr(resp, "text", None) or "{}").strip()
            if text:
                return text
        except Exception as e:  # pragma: no cover
            last_error = e
            tried_models.append(m)
            continue

    # Если все попытки провалились — бросаем детальную ошибку
    error_msg = f"Ошибка вызова Gemini. Попробованы модели: {tried_models}. "
    error_msg += f"Доступные модели: {available_models[:5]}. "
    error_msg += f"Последняя ошибка: {last_error}"
    raise LLMProviderError(error_msg)


# =============================
# Публичная функция генерации
# =============================

def generate_recommendations(
    user_query: str,
    retrieved_places: List[Dict[str, Any]],
    provider: str = "openai",  # "openai" | "anthropic"
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Генерирует 1–3 рекомендации на основе списка релевантных мест и запроса пользователя.

    Параметры:
    - user_query: текст запроса пользователя
    - retrieved_places: список словарей (результат из retrieval.find_relevant_places)
    - provider: провайдер LLM ("openai" или "anthropic")
    - model: имя модели провайдера (по умолчанию разумные значения)

    Возвращает dict согласно схеме response_format.
    """
    load_dotenv()

    if not user_query or not isinstance(user_query, str):
        raise ValueError("user_query должен быть непустой строкой")
    if not isinstance(retrieved_places, list) or not retrieved_places:
        # Return friendly error message instead of raising exception
        return {
            "recommendations": [{
                "name": "Места не найдены",
                "category": "Информация",
                "distance": "—",
                "why": f"В радиусе {max_distance_km if 'max_distance_km' in locals() else 5}км от выбранной точки не найдено подходящих мест.",
                "action_plan": "Попробуйте: 1) Увеличить радиус поиска (передвиньте ползунок до 10км), 2) Выбрать другое 'точное место' ближе к центру города, 3) Изменить запрос (например, убрать слишком специфичные требования).",
                "estimated_time": "—",
                "working_hours": "—",
                "confidence": 0.0
            }]
        }

    system_prompt = build_system_prompt()

    # Передаём в LLM топ-5..10 мест (сократим до 8 для компактности)
    top_places = retrieved_places[:8]

    user_prompt = build_user_prompt(user_query, top_places)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Выбор провайдера
    provider = (provider or "openai").lower()
    if provider == "openai":
        model = model or "gpt-4o-mini"
        raw = _call_openai(messages, model=model)
    elif provider == "anthropic":
        model = model or "claude-3-5-sonnet-latest"
        raw = _call_anthropic(messages, model=model)
    elif provider == "gemini":
        # Используем стабильный алиас модели Gemini (без -latest суффикса)
        model = model or "gemini-1.5-flash"
        raw = _call_gemini(messages, model=model)
    else:
        raise ValueError("Недопустимый provider. Используйте 'openai', 'anthropic' или 'gemini'.")

    # Парсим JSON безопасно, с одной попыткой исправления
    def _parse_json(s: str) -> Dict[str, Any]:
        try:
            return json.loads(s)
        except Exception:
            # Попробуем вытащить JSON блок из текста
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start : end + 1])
                except Exception:
                    pass
            raise

    try:
        data = _parse_json(raw)
    except Exception as e:
        raise LLMProviderError(f"Не удалось распарсить JSON из ответа модели: {e}\nОтвет модели: {raw[:500]}")

    # Минимальная валидация схемы
    if not isinstance(data, dict) or "recommendations" not in data:
        raise LLMProviderError("Ответ модели не соответствует ожидаемой структуре (нет 'recommendations')")

    recs = data.get("recommendations")
    if not isinstance(recs, list):
        raise LLMProviderError("Поле 'recommendations' должно быть списком")

    # Нормализация полей и отсечение до 3
    out_recs: List[Dict[str, Any]] = []
    for r in recs[:3]:
        if not isinstance(r, dict):
            continue
        out_recs.append({
            "name": str(r.get("name", "Без названия")),
            "category": str(r.get("category", "Не указано")),
            "distance": str(r.get("distance", "Не указано")),
            "why": str(r.get("why", "")),
            "action_plan": str(r.get("action_plan", "")),
            "estimated_time": str(r.get("estimated_time", "Не указано")),
            "working_hours": str(r.get("working_hours", "Рекомендуем уточнить время работы")),
            "confidence": float(r.get("confidence", 0.5)),
        })

    return {"recommendations": out_recs}


# =============================
# Пример запуска
# =============================
if __name__ == "__main__":
    # Демонстрационный пример (потребует валидного OPENAI_API_KEY или ANTHROPIC_API_KEY)
    demo_places = [
        {
            "name": "Coffee Room",
            "category": "Кафе",
            "subcategory": "Кофейня",
            "address": "ул. Пример, 1",
            "district": "Есильский",
            "distance_text": "800м, ~10 мин пешком",
            "working_hours": "Ежедневно с 08:00 до 23:00",
            "instagram": "@coffeeroom",
            "website": "https://coffeeroom.example",
            "phone": "+7 701 000 00 00",
            "open_now": True,
            "popularity_score": 0.9,
            "description": "Тихая кофейня с Wi‑Fi и розетками у столиков."
        },
        {
            "name": "Городской парк",
            "category": "Парк",
            "address": "пр. Мира, 10",
            "district": "Алматинский",
            "distance_text": "1.6 км, ~20 мин пешком",
            "working_hours": "Круглосуточно",
            "open_now": True,
            "popularity_score": 0.6,
            "description": "Зелёная зона для прогулок, детские площадки, фонтаны."
        }
    ]

    try:
        out = generate_recommendations(
            user_query="Я возле Байтерека, хочу тихое место с Wi‑Fi на 1–2 часа",
            retrieved_places=demo_places,
            provider=os.getenv("ASTANA_GUIDE_PROVIDER", "openai"),
            model=os.getenv("ASTANA_GUIDE_MODEL"),
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Ошибка генерации:", e)
