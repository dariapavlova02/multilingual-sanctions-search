#!/usr/bin/env python3
"""
Скрипт для развертывания конфигурации на продакшен сервере
"""
import requests
import time
import json

def check_production_status():
    """Проверить текущее состояние продакшен сервера"""
    print("🔍 Проверяем текущее состояние продакшен сервера...")

    base_url = "http://95.217.84.234:8000"

    # 1. Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Сервер доступен")
        else:
            print(f"⚠️  Health check: {response.status_code}")
    except Exception as e:
        print(f"❌ Сервер недоступен: {e}")
        return False

    # 2. Тест с проблемным случаем
    test_data = {
        "text": "Ковриков Роман Валерійович",
        "options": {
            "enable_advanced_features": True,
            "generate_variants": False,
            "generate_embeddings": False,
            "enable_search": True
        }
    }

    print(f"\n🧪 Тестируем с '{test_data['text']}'...")

    try:
        response = requests.post(
            f"{base_url}/process",
            json=test_data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()

            # Анализ результата
            decision = result.get("decision", {})
            search_results = result.get("search_results", {})

            print(f"Status: {response.status_code}")
            print(f"Risk level: {decision.get('risk_level', 'N/A')}")
            print(f"Decision reasons: {decision.get('decision_reasons', [])}")

            details = decision.get('decision_details', {})
            smartfilter_should_process = details.get('smartfilter_should_process', False)
            print(f"SmartFilter should process: {smartfilter_should_process}")

            total_hits = search_results.get('total_hits', 0)
            print(f"Search hits: {total_hits}")

            # Определяем проблему
            if not smartfilter_should_process:
                print("🔥 ПРОБЛЕМА: SmartFilter блокирует обработку!")
                print("   Причина: Скорее всего не установлены переменные окружения")
                return False
            elif total_hits == 0:
                print("🔥 ПРОБЛЕМА: SmartFilter разрешает, но поиск не находит результаты!")
                print("   Причина: Проблема с индексом AC паттернов или гомоглифами")
                return False
            else:
                print("✅ Система работает правильно!")
                return True

        else:
            print(f"❌ API ошибка: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Ошибка запроса: {e}")
        return False

def show_required_environment():
    """Показать необходимые переменные окружения"""
    print("\n📋 НЕОБХОДИМЫЕ ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ:")
    print("=====================================")

    required_vars = [
        ("ENABLE_AHO_CORASICK", "true", "Включить AC поиск паттернов"),
        ("AHO_CORASICK_CONFIDENCE_BONUS", "0.3", "Бонус доверия для AC совпадений"),
        ("ENABLE_SMART_FILTER", "true", "Включить SmartFilter"),
        ("ALLOW_SMART_FILTER_SKIP", "true", "Разрешить пропуск SmartFilter"),
        ("ENABLE_SEARCH", "true", "Включить поиск в ES"),
        ("ENABLE_VECTOR_FALLBACK", "true", "Включить fallback на векторный поиск"),
        ("ENABLE_VARIANTS", "true", "Включить генерацию вариантов"),
        ("ENABLE_EMBEDDINGS", "true", "Включить генерацию векторов"),
        ("ENABLE_DECISION_ENGINE", "true", "Включить движок принятия решений"),
        ("ENABLE_METRICS", "true", "Включить сбор метрик"),
        ("PRIORITIZE_QUALITY", "true", "Приоритет качества"),
        ("ENABLE_FAISS_INDEX", "true", "Включить FAISS индекс"),
        ("ENABLE_AC_TIER0", "true", "Включить AC Tier 0 паттерны")
    ]

    for var_name, var_value, description in required_vars:
        print(f"{var_name}={var_value}  # {description}")

    print("\n📄 КОМАНДЫ ДЛЯ УСТАНОВКИ НА СЕРВЕРЕ:")
    print("====================================")
    print("# Создать .env файл:")
    print("cat > .env << 'EOF'")
    for var_name, var_value, _ in required_vars:
        print(f"{var_name}={var_value}")
    print("EOF")

    print("\n# Или экспортировать переменные:")
    for var_name, var_value, _ in required_vars:
        print(f"export {var_name}={var_value}")

    print("\n# Перезапустить сервис:")
    print("# sudo systemctl restart ai-service")
    print("# или")
    print("# pkill -f 'python.*main.py' && python -m src.ai_service.main &")

def test_elasticsearch_connectivity():
    """Проверить подключение к Elasticsearch"""
    print("\n🔍 Проверяем подключение к Elasticsearch...")

    ES_HOST = "95.217.84.234"
    ES_PORT = 9200
    ES_USER = "elastic"
    ES_PASSWORD = "[REDACTED]"

    try:
        from requests.auth import HTTPBasicAuth

        # Health check ES
        response = requests.get(
            f"http://{ES_HOST}:{ES_PORT}/_cluster/health",
            auth=HTTPBasicAuth(ES_USER, ES_PASSWORD),
            timeout=10
        )

        if response.status_code == 200:
            health = response.json()
            print(f"✅ ES доступен: {health.get('status', 'unknown')} статус")
        else:
            print(f"⚠️  ES health check: {response.status_code}")

        # Проверить индекс AC паттернов
        response = requests.get(
            f"http://{ES_HOST}:{ES_PORT}/ai_service_ac_patterns/_count",
            auth=HTTPBasicAuth(ES_USER, ES_PASSWORD),
            timeout=10
        )

        if response.status_code == 200:
            count_data = response.json()
            pattern_count = count_data.get('count', 0)
            print(f"✅ AC паттерны в индексе: {pattern_count:,}")

            if pattern_count == 0:
                print("🔥 ПРОБЛЕМА: Нет AC паттернов в ES!")
                return False
            elif pattern_count < 900000:
                print("⚠️  Мало AC паттернов, ожидаем ~942K")
            else:
                print("✅ Количество AC паттернов корректное")
        else:
            print(f"❌ Ошибка проверки AC паттернов: {response.status_code}")
            return False

        # Проверить индекс векторов
        response = requests.get(
            f"http://{ES_HOST}:{ES_PORT}/ai_service_vectors/_count",
            auth=HTTPBasicAuth(ES_USER, ES_PASSWORD),
            timeout=10
        )

        if response.status_code == 200:
            count_data = response.json()
            vector_count = count_data.get('count', 0)
            print(f"✅ Векторы в индексе: {vector_count:,}")

            if vector_count == 0:
                print("⚠️  Нет векторов в ES (не критично, но желательно)")
            elif vector_count < 25000:
                print("⚠️  Мало векторов, ожидаем ~26K")
            else:
                print("✅ Количество векторов корректное")
        else:
            print(f"⚠️  Ошибка проверки векторов: {response.status_code}")

        return True

    except Exception as e:
        print(f"❌ Ошибка подключения к ES: {e}")
        return False

def main():
    print("🚀 КОНФИГУРАЦИЯ ПРОДАКШЕН СЕРВЕРА")
    print("=" * 50)

    # 1. Проверить ES
    es_ok = test_elasticsearch_connectivity()

    # 2. Проверить текущее состояние
    server_ok = check_production_status()

    # 3. Показать требуемую конфигурацию
    show_required_environment()

    # 4. Заключение
    print("\n🎯 ЗАКЛЮЧЕНИЕ:")
    print("=" * 20)

    if es_ok and server_ok:
        print("✅ Система полностью рабочая!")
    elif es_ok and not server_ok:
        print("🔧 ES работает, но нужно настроить переменные окружения на сервере")
        print("   Выполните команды выше для установки переменных")
    elif not es_ok:
        print("🔥 Проблема с Elasticsearch - проверьте индексы AC паттернов")

    print("\n📞 СЛЕДУЮЩИЕ ШАГИ:")
    print("1. Установить переменные окружения на сервере")
    print("2. Перезапустить AI сервис")
    print("3. Протестировать API снова")

if __name__ == "__main__":
    main()