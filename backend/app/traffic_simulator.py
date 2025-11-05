"""
Smart Astana Traffic Simulator
Реалистичная симуляция трафика для 15-20 дорог Астаны, 6 пригородов и 3 мостов
"""
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from enum import Enum


class TrafficStatus(str, Enum):
    FREE = "free"  # 0-40% загрузка
    MODERATE = "moderate"  # 40-70% загрузка
    HEAVY = "heavy"  # 70-90% загрузка
    JAM = "jam"  # 90-100% загрузка


class RoadType(str, Enum):
    HIGHWAY = "highway"  # Крупные магистрали
    ARTERIAL = "arterial"  # Основные дороги
    BRIDGE = "bridge"  # Мосты
    ENTRANCE = "entrance"  # Въезды из пригородов


# =============================
# Основные дороги Астаны
# =============================

ASTANA_ROADS = {
    "kabanbay_batyr": {
        "id": "kabanbay_batyr",
        "name": "пр. Кабанбай батыра",
        "name_en": "Kabanbay Batyr Ave",
        "type": RoadType.HIGHWAY,
        "capacity": 8000,  # машин/час
        "speed_limit": 80,  # км/ч
        "lanes": 6,
        "length_km": 12.5,
        "coordinates": [[51.1694, 71.4491], [51.0894, 71.4291]],
        "description": "Главная магистраль левого берега, соединяет север и юг города"
    },
    "turan_ave": {
        "id": "turan_ave",
        "name": "пр. Туран",
        "name_en": "Turan Avenue",
        "type": RoadType.HIGHWAY,
        "capacity": 7500,
        "speed_limit": 80,
        "lanes": 6,
        "length_km": 11.2,
        "coordinates": [[51.1294, 71.4091], [51.1894, 71.4891]],
        "description": "Параллельная магистраль, альтернатива Кабанбай батыра"
    },
    "mangilik_el": {
        "id": "mangilik_el",
        "name": "пр. Мәңгілік Ел",
        "name_en": "Mangilik El Avenue",
        "type": RoadType.HIGHWAY,
        "capacity": 9000,
        "speed_limit": 90,
        "lanes": 8,
        "length_km": 8.7,
        "coordinates": [[51.0994, 71.4191], [51.1494, 71.4791]],
        "description": "Новая широкая магистраль с высокой пропускной способностью"
    },
    "abay_ave": {
        "id": "abay_ave",
        "name": "пр. Абая",
        "name_en": "Abay Avenue",
        "type": RoadType.ARTERIAL,
        "capacity": 5500,
        "speed_limit": 60,
        "lanes": 4,
        "length_km": 9.3,
        "coordinates": [[51.1594, 71.4291], [51.1794, 71.4691]],
        "description": "Основная артерия с торговыми центрами и офисами"
    },
    "syganak_st": {
        "id": "syganak_st",
        "name": "ул. Сыганак",
        "name_en": "Syganak Street",
        "type": RoadType.ARTERIAL,
        "capacity": 4500,
        "speed_limit": 60,
        "lanes": 4,
        "length_km": 7.8,
        "coordinates": [[51.1394, 71.4391], [51.1694, 71.4791]],
        "description": "Популярная улица с множеством магазинов"
    },
    "respublika_ave": {
        "id": "respublika_ave",
        "name": "пр. Республики",
        "name_en": "Respublika Avenue",
        "type": RoadType.ARTERIAL,
        "capacity": 6000,
        "speed_limit": 70,
        "lanes": 4,
        "length_km": 10.5,
        "coordinates": [[51.1194, 71.4091], [51.1694, 71.4891]],
        "description": "Важная дорога правого берега"
    },
    "kenesary_st": {
        "id": "kenesary_st",
        "name": "ул. Кенесары",
        "name_en": "Kenesary Street",
        "type": RoadType.ARTERIAL,
        "capacity": 3500,
        "speed_limit": 50,
        "lanes": 3,
        "length_km": 6.2,
        "coordinates": [[51.1494, 71.4391], [51.1694, 71.4591]],
        "description": "Центральная улица старого города"
    },
    "beibitshilik_st": {
        "id": "beibitshilik_st",
        "name": "ул. Бейбітшілік",
        "name_en": "Beibitshilik Street",
        "type": RoadType.ARTERIAL,
        "capacity": 4000,
        "speed_limit": 60,
        "lanes": 4,
        "length_km": 8.1,
        "coordinates": [[51.1594, 71.4191], [51.1894, 71.4591]],
        "description": "Деловой район с офисными зданиями"
    },
    "zheltoksan_st": {
        "id": "zheltoksan_st",
        "name": "ул. Желтоқсан",
        "name_en": "Zheltoksan Street",
        "type": RoadType.ARTERIAL,
        "capacity": 3800,
        "speed_limit": 50,
        "lanes": 3,
        "length_km": 5.9,
        "coordinates": [[51.1694, 71.4291], [51.1894, 71.4491]],
        "description": "Жилой район с школами и детсадами"
    },
    "pobedy_ave": {
        "id": "pobedy_ave",
        "name": "пр. Победы",
        "name_en": "Pobedy Avenue",
        "type": RoadType.ARTERIAL,
        "capacity": 5000,
        "speed_limit": 70,
        "lanes": 4,
        "length_km": 11.2,
        "coordinates": [[51.0994, 71.3891], [51.1694, 71.4691]],
        "description": "Длинная магистраль западной части города"
    },
    "auezov_st": {
        "id": "auezov_st",
        "name": "ул. Әуезова",
        "name_en": "Auezov Street",
        "type": RoadType.ARTERIAL,
        "capacity": 4200,
        "speed_limit": 60,
        "lanes": 4,
        "length_km": 7.5,
        "coordinates": [[51.1394, 71.4091], [51.1694, 71.4591]],
        "description": "Улица с университетами и библиотеками"
    },
    "saryarka_ave": {
        "id": "saryarka_ave",
        "name": "пр. Сарыарка",
        "name_en": "Saryarka Avenue",
        "type": RoadType.HIGHWAY,
        "capacity": 7000,
        "speed_limit": 80,
        "lanes": 6,
        "length_km": 13.8,
        "coordinates": [[51.0794, 71.3791], [51.1894, 71.5091]],
        "description": "Длинная магистраль через весь город"
    },
    "zhenis_ave": {
        "id": "zhenis_ave",
        "name": "пр. Жеңіс",
        "name_en": "Zhenis Avenue",
        "type": RoadType.ARTERIAL,
        "capacity": 4800,
        "speed_limit": 60,
        "lanes": 4,
        "length_km": 8.9,
        "coordinates": [[51.1394, 71.3891], [51.1794, 71.4591]],
        "description": "Проспект с торговыми комплексами"
    },
    "korgalzhyn_road": {
        "id": "korgalzhyn_road",
        "name": "Коргалжынское шоссе",
        "name_en": "Korgalzhyn Highway",
        "type": RoadType.HIGHWAY,
        "capacity": 6000,
        "speed_limit": 90,
        "lanes": 4,
        "length_km": 15.2,
        "coordinates": [[51.1694, 71.4491], [51.2194, 71.3291]],
        "description": "Шоссе на запад к аэропорту и Коргалжыну"
    },
    "kosshy_road": {
        "id": "kosshy_road",
        "name": "Дорога на Коссы",
        "name_en": "Kosshy Road",
        "type": RoadType.HIGHWAY,
        "capacity": 5500,
        "speed_limit": 90,
        "lanes": 4,
        "length_km": 12.7,
        "coordinates": [[51.1694, 71.4491], [51.2394, 71.5191]],
        "description": "Северный въезд из крупного пригорода Коссы"
    }
}

# =============================
# 6 пригородов с въездами
# =============================

SUBURBS = {
    "kosshy": {
        "id": "kosshy",
        "name": "Коссы",
        "name_en": "Kosshy",
        "population": 25000,
        "daily_inflow": 18000,  # машин в день
        "distance_km": 18,
        "entry_road": "kosshy_road",
        "coordinates": [51.2594, 71.5491],
        "description": "Крупнейший пригород, много жителей работают в Астане"
    },
    "korgalzhyn": {
        "id": "korgalzhyn",
        "name": "Коргалжын",
        "name_en": "Korgalzhyn",
        "population": 6000,
        "daily_inflow": 3500,
        "distance_km": 130,
        "entry_road": "korgalzhyn_road",
        "coordinates": [50.6194, 70.3691],
        "description": "Районный центр на западе"
    },
    "akmol": {
        "id": "akmol",
        "name": "Акмол",
        "name_en": "Akmol",
        "population": 12000,
        "daily_inflow": 8500,
        "distance_km": 45,
        "entry_road": "saryarka_ave",
        "coordinates": [51.0294, 71.0991],
        "description": "Западный пригород с дачами"
    },
    "shortandy": {
        "id": "shortandy",
        "name": "Шортанды",
        "name_en": "Shortandy",
        "population": 18000,
        "daily_inflow": 12000,
        "distance_km": 35,
        "entry_road": "saryarka_ave",
        "coordinates": [51.0194, 71.9091],
        "description": "Крупный поселок на востоке"
    },
    "arshaly": {
        "id": "arshaly",
        "name": "Аршалы",
        "name_en": "Arshaly",
        "population": 8000,
        "daily_inflow": 5500,
        "distance_km": 80,
        "entry_road": "kabanbay_batyr",
        "coordinates": [51.3794, 72.1891],
        "description": "Районный центр на юго-востоке"
    }
}

# =============================
# 3 моста между берегами
# =============================

BRIDGES = {
    "atyrau_bridge": {
        "id": "atyrau_bridge",
        "name": "Мост Атырау",
        "name_en": "Atyrau Bridge",
        "type": RoadType.BRIDGE,
        "capacity": 4500,
        "speed_limit": 60,
        "lanes": 4,
        "length_km": 0.85,
        "coordinates": [[51.1594, 71.4391], [51.1614, 71.4411]],
        "status": "operational",
        "year_built": 2008,
        "description": "Южный мост, соединяет старый город и правый берег"
    },
    "central_bridge": {
        "id": "central_bridge",
        "name": "Центральный мост",
        "name_en": "Central Bridge",
        "type": RoadType.BRIDGE,
        "capacity": 5000,
        "speed_limit": 70,
        "lanes": 6,
        "length_km": 1.2,
        "coordinates": [[51.1694, 71.4491], [51.1714, 71.4511]],
        "status": "operational",
        "year_built": 2012,
        "description": "Главный мост города, самый загруженный"
    },
    "northern_bridge": {
        "id": "northern_bridge",
        "name": "Северный мост",
        "name_en": "Northern Bridge",
        "type": RoadType.BRIDGE,
        "capacity": 4000,
        "speed_limit": 60,
        "lanes": 4,
        "length_km": 0.95,
        "coordinates": [[51.1794, 71.4591], [51.1814, 71.4611]],
        "status": "operational",
        "year_built": 2015,
        "description": "Новый мост на севере, разгружает центр"
    }
}


# =============================
# Функции расчета трафика
# =============================

def get_time_coefficient(hour: int) -> float:
    """
    Коэффициент загрузки дорог в зависимости от времени суток.

    Args:
        hour: Час суток (0-23)

    Returns:
        Коэффициент загрузки (0.3-1.8)
    """
    if 7 <= hour < 10:
        # Утренний пик: 7-10 утра
        return 1.8
    elif 17 <= hour < 20:
        # Вечерний пик: 17-20 вечера
        return 1.6
    elif 12 <= hour < 14:
        # Обеденный пик: 12-14
        return 1.2
    elif 0 <= hour < 6:
        # Ночь: 0-6
        return 0.3
    elif 6 <= hour < 7:
        # Раннее утро: 6-7
        return 0.7
    elif 10 <= hour < 12:
        # До обеда: 10-12
        return 0.9
    elif 14 <= hour < 17:
        # После обеда: 14-17
        return 1.0
    elif 20 <= hour < 22:
        # Вечер: 20-22
        return 0.8
    else:
        # Поздний вечер: 22-24
        return 0.5


def get_suburb_flow_coefficient(hour: int, suburb_id: str) -> float:
    """
    Коэффициент потока из пригорода в зависимости от времени.
    Утром люди едут В город, вечером ИЗ города.

    Args:
        hour: Час суток (0-23)
        suburb_id: ID пригорода

    Returns:
        Коэффициент потока (0.1-2.0)
    """
    if 6 <= hour < 9:
        # Утро: поток В город (максимум)
        return 2.0
    elif 17 <= hour < 20:
        # Вечер: поток ИЗ города (максимум)
        return 1.8
    elif 9 <= hour < 17:
        # День: небольшой поток
        return 0.5
    elif 20 <= hour < 23:
        # Вечер: средний поток
        return 0.7
    else:
        # Ночь: минимальный поток
        return 0.1


def calculate_traffic_load(
    capacity: int,
    hour: int,
    road_type: RoadType,
    random_factor: bool = True
) -> Dict:
    """
    Расчет текущей загрузки дороги.

    Args:
        capacity: Пропускная способность (машин/час)
        hour: Текущий час суток
        road_type: Тип дороги
        random_factor: Добавить случайность (±15%)

    Returns:
        Dict с данными о загрузке
    """
    # Базовый коэффициент времени
    time_coef = get_time_coefficient(hour)

    # Дополнительные модификаторы по типу дороги
    if road_type == RoadType.BRIDGE:
        # Мосты всегда более загружены (+20%)
        time_coef *= 1.2
    elif road_type == RoadType.ENTRANCE:
        # Въезды зависят от пригородных потоков
        time_coef *= 1.1

    # Случайность ±15%
    if random_factor:
        randomness = random.uniform(0.85, 1.15)
        time_coef *= randomness

    # Расчет текущей нагрузки
    current_vehicles = int(capacity * time_coef)
    current_vehicles = min(current_vehicles, int(capacity * 1.05))  # Максимум 105%

    # Процент загрузки
    load_percent = (current_vehicles / capacity) * 100

    # Статус трафика
    if load_percent < 40:
        status = TrafficStatus.FREE
        color = "green"
    elif load_percent < 70:
        status = TrafficStatus.MODERATE
        color = "yellow"
    elif load_percent < 90:
        status = TrafficStatus.HEAVY
        color = "orange"
    else:
        status = TrafficStatus.JAM
        color = "red"

    # Расчет средней скорости
    speed_factor = max(0.3, 1 - (load_percent / 150))  # При 100% загрузке скорость падает до 33%

    return {
        "capacity": capacity,
        "current_vehicles": current_vehicles,
        "load_percent": round(load_percent, 1),
        "status": status,
        "color": color,
        "time_coefficient": round(time_coef, 2),
        "speed_factor": round(speed_factor, 2)
    }


def calculate_eco_impact(traffic_data: Dict) -> Dict:
    """
    Расчет экологического воздействия трафика.

    Args:
        traffic_data: Данные о трафике со всех дорог

    Returns:
        Dict с экологическими метриками
    """
    total_vehicles = sum(
        road["traffic"]["current_vehicles"]
        for road in traffic_data.get("roads", [])
    )

    # Средний расход топлива: 8 л/100км в свободном режиме
    # При пробках расход увеличивается до 12-15 л/100км
    avg_distance_km = 10  # Средняя поездка по городу

    # CO2: 2.3 кг на литр бензина
    base_fuel_per_vehicle = (8 / 100) * avg_distance_km  # литров
    jam_fuel_per_vehicle = (14 / 100) * avg_distance_km

    # Процент машин в пробках
    jam_vehicles = sum(
        road["traffic"]["current_vehicles"]
        for road in traffic_data.get("roads", [])
        if road["traffic"]["status"] in ["heavy", "jam"]
    )
    free_vehicles = total_vehicles - jam_vehicles

    # Расчет топлива и CO2
    total_fuel_liters = (free_vehicles * base_fuel_per_vehicle +
                        jam_vehicles * jam_fuel_per_vehicle)
    total_co2_kg = total_fuel_liters * 2.3
    total_co2_tons = total_co2_kg / 1000

    # Экономические потери
    # При скорости 20 км/ч вместо 60 км/ч: потеря времени = 2x
    # Средняя зарплата в Астане: 350,000 тенге/месяц = ~14,600 тенге/день
    # Рабочий день: 8 часов = 1,825 тенге/час
    avg_time_in_traffic_hours = 1.5  # Средняя поездка
    time_loss_factor = jam_vehicles / max(total_vehicles, 1)
    economic_loss_per_hour = 1825
    total_economic_loss = (jam_vehicles * avg_time_in_traffic_hours *
                          time_loss_factor * economic_loss_per_hour)

    return {
        "total_vehicles": total_vehicles,
        "vehicles_in_jams": jam_vehicles,
        "jam_percentage": round((jam_vehicles / max(total_vehicles, 1)) * 100, 1),
        "co2_emissions_kg_per_hour": round(total_co2_kg, 1),
        "co2_emissions_tons_per_day": round(total_co2_tons * 24, 2),
        "fuel_consumption_liters_per_hour": round(total_fuel_liters, 1),
        "economic_loss_tenge_per_hour": round(total_economic_loss, 0),
        "economic_loss_tenge_per_day": round(total_economic_loss * 24, 0),
        "estimated_time_loss_hours": round(time_loss_factor * avg_time_in_traffic_hours, 2)
    }


def get_current_traffic(target_time: Optional[datetime] = None) -> Dict:
    """
    Получить текущее состояние трафика на всех дорогах.

    Args:
        target_time: Целевое время (по умолчанию сейчас)

    Returns:
        Полные данные о трафике
    """
    if target_time is None:
        target_time = datetime.now()

    hour = target_time.hour

    # Расчет трафика для всех дорог
    roads_traffic = []
    for road_id, road_data in ASTANA_ROADS.items():
        traffic = calculate_traffic_load(
            capacity=road_data["capacity"],
            hour=hour,
            road_type=road_data["type"]
        )

        roads_traffic.append({
            **road_data,
            "traffic": traffic
        })

    # Расчет трафика для мостов
    bridges_traffic = []
    for bridge_id, bridge_data in BRIDGES.items():
        traffic = calculate_traffic_load(
            capacity=bridge_data["capacity"],
            hour=hour,
            road_type=bridge_data["type"]
        )

        bridges_traffic.append({
            **bridge_data,
            "traffic": traffic
        })

    # Расчет потоков из пригородов
    suburbs_traffic = []
    for suburb_id, suburb_data in SUBURBS.items():
        flow_coef = get_suburb_flow_coefficient(hour, suburb_id)
        current_flow = int(suburb_data["daily_inflow"] * flow_coef / 24)

        # Процент от дневной нормы
        flow_percent = (current_flow / (suburb_data["daily_inflow"] / 24)) * 100

        suburbs_traffic.append({
            **suburb_data,
            "current_flow": current_flow,
            "flow_coefficient": round(flow_coef, 2),
            "flow_percent": round(flow_percent, 1),
            "status": "high" if flow_percent > 150 else "normal" if flow_percent > 50 else "low"
        })

    # Агрегированная статистика
    total_capacity = sum(r["capacity"] for r in ASTANA_ROADS.values())
    total_vehicles = sum(r["traffic"]["current_vehicles"] for r in roads_traffic)
    avg_load = (total_vehicles / total_capacity) * 100

    result = {
        "timestamp": target_time.isoformat(),
        "hour": hour,
        "time_coefficient": get_time_coefficient(hour),
        "roads": roads_traffic,
        "bridges": bridges_traffic,
        "suburbs": suburbs_traffic,
        "total_roads": len(roads_traffic),
        "total_bridges": len(bridges_traffic),
        "total_suburbs": len(suburbs_traffic),
        "avg_city_load_percent": round(avg_load, 1),
        "total_vehicles_on_roads": total_vehicles,
        "total_capacity": total_capacity
    }

    # Добавляем эко-импакт
    result["eco_impact"] = calculate_eco_impact(result)

    return result


def predict_traffic(hours_ahead: int = 4) -> List[Dict]:
    """
    Предсказание трафика на N часов вперед.

    Args:
        hours_ahead: Количество часов для предсказания

    Returns:
        Список прогнозов трафика
    """
    predictions = []
    now = datetime.now()

    for i in range(1, hours_ahead + 1):
        future_time = now + timedelta(hours=i)
        # Предсказание без случайности для стабильности
        traffic_data = get_current_traffic(future_time)
        predictions.append(traffic_data)

    return predictions
