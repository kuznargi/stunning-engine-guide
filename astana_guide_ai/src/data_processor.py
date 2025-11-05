"""
AstanaGuide AI — Data Processor

Задача:
- Загрузить CSV с POI (места Астаны)
- Очистить и нормализовать данные
- Сконструировать текстовые описания для каждого POI
- Построить эмбеддинги (sentence-transformers, мультиязычная модель)
- Сохранить:
  - processed_pois.json — очищенные записи + служебные поля
  - faiss.index — векторный индекс для быстрого поиска
  - embeddings.npy (необязательно, сохраняем для отладки)
  - id_map.json — соответствие row_id -> индекс в FAISS

Требования:
- Python 3.10+
- pandas>=2.0.0, sentence-transformers, faiss-cpu>=1.7.0, numpy, python-dotenv

ПРИМЕЧАНИЯ:
- Используется мультиязычная модель "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  хорошо подходит для русского/казахского/английского текстов.
- Если вы хотите использовать OpenAI Embeddings вместо локальной модели, смотрите комментарии в коде
  (замена функции build_embeddings()).

Пример запуска из терминала:
    python -m astana_guide_ai.src.data_processor \
        --csv_path "./gis.csv" \
        --output_dir "./astana_guide_ai/data" \
        --model_name "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
        --batch_size 64

После выполнения:
- Файлы будут сохранены в ./astana_guide_ai/data

Структура одной записи в processed_pois.json:
{
  "row_id": 0,
  "id": "...",                    # из CSV если есть колонка ID/Id/id
  "name": "...",
  "region": "...",
  "district": "...",              # район
  "city": "...",
  "address": "...",
  "phone": "...",
  "email": "...",
  "website": "...",
  "category": "...",              # Рубрика
  "subcategory": "...",           # Подрубрика
  "working_hours": "...",
  "instagram": "...",
  "lat": 51.1,
  "lon": 71.4,
  "description": "...",           # текст для семантического поиска
  "popularity_score": 0.5           # эвристический балл популярности
}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Векторная модель и индекс
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None  # type: ignore


# =============================
# Утилиты нормализации и парсинга
# =============================

def _norm_str(x: Any) -> str:
    """Безопасно приводит значение к нормализованной строке."""
    if pd.isna(x) or x is None:
        return ""
    s = str(x).strip()
    # Убираем дублирующиеся пробелы
    s = " ".join(s.split())
    return s


def _to_float(x: Any) -> Optional[float]:
    """Пытается преобразовать значение к float, иначе возвращает None."""
    try:
        if pd.isna(x) or x is None or str(x).strip() == "":
            return None
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def _make_description(row: Dict[str, Any]) -> str:
    """Формирует текстовое описание POI из полей записи.

    Ожидаемые ключи в row (не все обязательны): name, category, subcategory, address, working_hours, district
    """
    lines = []
    name = row.get("name") or "Без названия"
    lines.append(name)

    category = row.get("category") or ""
    subcategory = row.get("subcategory") or ""
    cat_line = "Категория: "
    if category:
        cat_line += category
    if subcategory:
        if category:
            cat_line += f", {subcategory}"
        else:
            cat_line += subcategory
    if cat_line != "Категория: ":
        lines.append(cat_line)

    address = row.get("address")
    if address:
        lines.append(f"Адрес: {address}")

    wh = row.get("working_hours")
    if wh:
        lines.append(f"Время работы: {wh}")

    district = row.get("district")
    if district:
        lines.append(f"Район: {district}")

    # Добавим контактные детали для качества семантики
    extras = []
    if row.get("website"):
        extras.append("есть сайт")
    if row.get("instagram"):
        extras.append("есть Instagram")
    if row.get("phone"):
        extras.append("есть телефон")
    if extras:
        lines.append("Особенности: " + ", ".join(extras))

    return "\n".join(lines)


def _popularity_heuristic(row: Dict[str, Any]) -> float:
    """Простая эвристика популярности (0..1), если нет реального рейтинга в CSV.

    - +0.25 за сайт
    - +0.25 за Instagram
    - +0.25 за телефон
    - +0.25 за заполненный адрес
    """
    score = 0.0
    if row.get("website"):
        score += 0.25
    if row.get("instagram"):
        score += 0.25
    if row.get("phone"):
        score += 0.25
    if row.get("address"):
        score += 0.25
    return max(0.0, min(1.0, score))


# =============================
# Основная логика обработки
# =============================

@dataclass
class ProcessConfig:
    csv_path: str
    output_dir: str
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 64
    normalize_text: bool = True
    save_embeddings: bool = True


def load_csv(csv_path: str) -> pd.DataFrame:
    """Загружает CSV в DataFrame с безопасной обработкой ошибок.

    Поддерживает большие файлы, автоматически определяет разделитель (по умолчанию запятая).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")

    try:
        # encoding: пытаемся cp1251 затем utf-8 для совместимости
        try:
            df = pd.read_csv(csv_path, encoding="utf-8",sep=";",quotechar='"',
    on_bad_lines="skip",
    engine="python"      )
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="cp1251",sep=";",quotechar='"',
    on_bad_lines="skip",
    engine="python" )
    except Exception as e:
        raise RuntimeError(f"Не удалось прочитать CSV: {e}")

    if df.empty:
        raise ValueError("CSV загружен, но не содержит данных")

    return df


def clean_and_structure(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Очищает DataFrame и приводит к унифицированному списку словарей."""

    # 1️⃣ Приводим названия столбцов к нижнему регистру и убираем пробелы
    df.columns = df.columns.str.strip().str.lower()

    # 2️⃣ Маппинг возможных названий к унифицированным
    col_map_candidates = {
        "id": ["id"],
        "name": ["название", "наименование", "name"],
        "region": ["регион", "region"],
        "district": ["район", "district"],
        "city": ["город", "city"],
        "address": ["адрес", "address"],
        "phone": ["телефон", "phone"],
        "email": ["email", "e-mail"],
        "website": ["сайт", "site", "website", "web"],
        "category": ["рубрика", "category"],
        "subcategory": ["подрубрика", "subcategory"],
        "working_hours": ["время работы", "график", "режим работы", "working hours"],
        "instagram": ["instagram", "insta"],
        "lat": ["широта", "lat", "latitude"],
        "lon": ["долгота", "lon", "long", "lng", "longitude"],
    }

    # 3️⃣ Строим reverse_map
    reverse_map: Dict[str, str] = {}
    for unified, variants in col_map_candidates.items():
        for v in variants:
            if v in df.columns:
                reverse_map[v] = unified
                break

    # 4️⃣ Переименовываем
    df_renamed = df.rename(columns=reverse_map).reset_index(drop=True)

    # 5️⃣ Попытка найти координаты под любыми доступными именами
    if "lat" not in df_renamed.columns or "lon" not in df_renamed.columns:
        raise ValueError(f"⚠️ Не найдены столбцы координат. Найдены: {df_renamed.columns.tolist()}")

    # 6️⃣ Преобразуем координаты в float (включая замену запятых на точки)
    df_renamed["lat"] = (
        df_renamed["lat"].astype(str).str.replace(",", ".", regex=False)
    )
    df_renamed["lon"] = (
        df_renamed["lon"].astype(str).str.replace(",", ".", regex=False)
    )
    df_renamed["lat"] = pd.to_numeric(df_renamed["lat"], errors="coerce")
    df_renamed["lon"] = pd.to_numeric(df_renamed["lon"], errors="coerce")

    # 7️⃣ Фильтрация пустых координат
    df_renamed = df_renamed.dropna(subset=["lat", "lon"])

    records: List[Dict[str, Any]] = []
    for i, row in df_renamed.iterrows():
        rec: Dict[str, Any] = {}
        rec["row_id"] = int(i)
        rec["id"] = _norm_str(row.get("id", ""))
        rec["name"] = _norm_str(row.get("name", "")) or f"Объект #{i}"
        rec["region"] = _norm_str(row.get("region", ""))
        rec["district"] = _norm_str(row.get("district", ""))
        rec["city"] = _norm_str(row.get("city", ""))
        rec["address"] = _norm_str(row.get("address", ""))
        rec["phone"] = _norm_str(row.get("phone", ""))
        rec["email"] = _norm_str(row.get("email", ""))
        rec["website"] = _norm_str(row.get("website", ""))
        rec["category"] = _norm_str(row.get("category", ""))
        rec["subcategory"] = _norm_str(row.get("subcategory", ""))
        rec["working_hours"] = _norm_str(row.get("working_hours", ""))
        rec["instagram"] = _norm_str(row.get("instagram", ""))
        rec["lat"] = float(row.get("lat"))
        rec["lon"] = float(row.get("lon"))

        rec["description"] = _make_description(rec)
        rec["popularity_score"] = _popularity_heuristic(rec)
        records.append(rec)

    if not records:
        raise ValueError("После очистки не осталось ни одной валидной записи с координатами")

    return records

def build_embeddings(
    records: List[Dict[str, Any]],
    model_name: str,
    batch_size: int = 64,
) -> Tuple[np.ndarray, int]:
    """Строит эмбеддинги для списка записей.

    Возвращает кортеж (embeddings, dim), где
    - embeddings: np.ndarray формы (N, D)
    - dim: размерность эмбеддинга

    Примечание: Используется локальная мультиязычная модель.
    Если хотите использовать OpenAI, замените на вызов openai embeddings.
    """
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers не установлен. Установите пакет: pip install sentence-transformers"
        )

    model = SentenceTransformer(model_name)
    texts = [r.get("description", "") for r in records]

    # Пакетная обработка для памяти
    all_vecs: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch = texts[start:end]
        vecs = model.encode(batch, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        all_vecs.append(vecs)

    embeddings = np.vstack(all_vecs)
    dim = embeddings.shape[1]
    return embeddings, dim


def build_faiss_index(embeddings: np.ndarray) -> Any:
    """Создает FAISS индекс (IndexFlatIP для косинусного сходства при нормализованных векторах).

    Важно: Мы нормализуем эмбеддинги при построении, поэтому скалярное произведение = косинусное сходство.
    """
    if faiss is None:
        raise ImportError("faiss-cpu не установлен. Установите пакет: pip install faiss-cpu")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_artifacts(
    records: List[Dict[str, Any]],
    embeddings: np.ndarray,
    index: Any,
    output_dir: str,
    model_name: str,
) -> Dict[str, str]:
    """Сохраняет обработанные данные и артефакты на диск.

    Возвращает словарь с путями к сохраненным файлам.
    """
    ensure_dir(output_dir)

    processed_path = os.path.join(output_dir, "processed_pois.json")
    id_map_path = os.path.join(output_dir, "id_map.json")
    emb_path = os.path.join(output_dir, "embeddings.npy")
    index_path = os.path.join(output_dir, "faiss.index")
    meta_path = os.path.join(output_dir, "meta.json")

    # processed_pois.json — список словарей
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # id_map.json — row_id -> позиция в FAISS (здесь 1:1 по порядку records)
    id_map = {str(r["row_id"]): i for i, r in enumerate(records)}
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    # embeddings.npy — для отладки (по желанию)
    np.save(emb_path, embeddings.astype(np.float32))

    # faiss.index — бинарный индекс
    if faiss is None:
        raise ImportError("faiss-cpu не установлен. Установите пакет: pip install faiss-cpu")
    faiss.write_index(index, index_path)

    # meta.json — метаданные набора
    meta = {
        "model_name": model_name,
        "num_records": len(records),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "files": {
            "processed_pois": processed_path,
            "id_map": id_map_path,
            "embeddings": emb_path,
            "faiss_index": index_path,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "processed_pois": processed_path,
        "id_map": id_map_path,
        "embeddings": emb_path,
        "faiss_index": index_path,
        "meta": meta_path,
    }


def process_csv_to_vectors(config: ProcessConfig) -> Dict[str, Any]:
    """Полный конвейер: загрузка CSV -> очистка -> эмбеддинги -> FAISS -> сохранение.

    Возвращает словарь со статистикой и путями к артефактам.
    """
    df = load_csv(config.csv_path)
    records = clean_and_structure(df)

    # Лог для пользователя
    print(f"Загружено валидных записей: {len(records)}")

    embeddings, dim = build_embeddings(
        records=records,
        model_name=config.model_name,
        batch_size=config.batch_size,
    )

    index = build_faiss_index(embeddings)

    artifacts = save_artifacts(
        records=records,
        embeddings=embeddings,
        index=index,
        output_dir=config.output_dir,
        model_name=config.model_name,
    )

    return {
        "num_records": len(records),
        "embedding_dim": dim,
        "artifacts": artifacts,
    }


# =============================
# CLI
# =============================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AstanaGuide AI — обработка CSV и построение векторной базы (FAISS)",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./gis.csv",
        help="Путь к исходному CSV (по умолчанию ./gis.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./astana_guide_ai/data",
        help="Каталог для сохранения артефактов (по умолчанию ./astana_guide_ai/data)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Имя модели sentence-transformers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Размер батча при кодировании",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    # Загружаем .env (на будущее — если позже решите использовать OpenAI)
    load_dotenv()

    args = parse_args(argv)
    config = ProcessConfig(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    try:
        result = process_csv_to_vectors(config)
        print("\nУспех! Построены эмбеддинги и создан FAISS индекс.")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        return 2
    except ImportError as e:
        print(f"Ошибка зависимостей: {e}")
        print("Убедитесь, что установлены: pandas, numpy, sentence-transformers, faiss-cpu, python-dotenv")
        return 3
    except ValueError as e:
        print(f"Проблема с данными: {e}")
        return 4
    except Exception as e:
        print("Непредвиденная ошибка:", repr(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
