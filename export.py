"""
Модуль для експорту YOLO моделей у різні формати.
Підтримує: ONNX, TensorRT, TFLite, OpenVINO, CoreML та інші.
Усі параметри конфігурації знаходяться на початку файлу.
Документація: https://docs.ultralytics.com/modes/export/#arguments
"""

import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ
# =============================================================================
PROJECT_NAME = "yolov8x-p2_for_autolabelling"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, PROJECT_NAME)
DATASET_ROOT = os.path.join(BASE_DIR, "dataset_split")
YAML_PATH = os.path.join(DATASET_ROOT, "yolo.yaml")

# Шлях до моделі для експорту
MODEL_PATH = os.path.join(PROJECT_DIR, "baseline", "weights", "best.pt")


# =============================================================================
# КОНФІГУРАЦІЯ ЕКСПОРТУ
# Документація: https://docs.ultralytics.com/modes/export/#arguments
# =============================================================================
EXPORT_CONFIG = {
    "format": "onnx",
    

    "imgsz": 960,
    "half": False,
    "int8": False,
    "optimize": False,
    "half": False,
    "dynamic": True,
    "simplify": True,
    "opset": None,
    "workspace": None,
    "nms": False,
    "batch": 1,
    "device": None,
    "data": None,
    "fraction": 1.0,
    "keras": False,
    "end2end": None,
}


# =============================================================================
# ФУНКЦІЇ
# =============================================================================
def print_header(model_path: str, config: dict) -> None:
    """Виведення заголовку експорту."""
    model_name = Path(model_path).name
    print()
    print("=" * 70)
    print("YOLO MODEL EXPORT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Format: {config['format']}")
    print(f"Image Size: {config['imgsz']}")
    print(f"Dynamic: {config.get('dynamic', False)}")
    print(f"Half (FP16): {config.get('half', False)}")
    print(f"INT8: {config.get('int8', False)}")
    if config['format'] == 'onnx':
        print(f"Simplify: {config.get('simplify', True)}")
        print(f"Opset: {config.get('opset', 'auto')}")
    if config['format'] == 'engine':
        print(f"Workspace: {config.get('workspace', 'auto')} GiB")
    if config.get('int8') and config.get('data'):
        print(f"Calibration data: {config['data']}")
    print("=" * 70)
    print()


def export_model(
    model_path: str = MODEL_PATH,
    **kwargs
) -> str | None:
    """
    Експорт YOLO моделі.
    
    Args:
        model_path: Шлях до моделі (.pt файл)
        **kwargs: Параметри експорту (перезаписують EXPORT_CONFIG)
    
    Returns:
        str | None: Шлях до експортованої моделі або None при помилці
    """
    # Перевірка існування моделі
    if not os.path.exists(model_path):
        print(f"Помилка: Модель не знайдена: {model_path}")
        return None
    
    # Об'єднуємо конфігурацію з переданими параметрами
    config = {**EXPORT_CONFIG, **kwargs}
    
    # Фільтруємо None значення (ultralytics не любить None)
    export_params = {k: v for k, v in config.items() if v is not None}
    
    # Виводимо заголовок
    print_header(model_path, config)
    
    # Завантаження та експорт
    start_time = datetime.now()
    print(f"[Export] Завантаження моделі: {Path(model_path).name}")
    
    try:
        model = YOLO(model_path)
        print(f"[Export] Експорт у формат '{config['format']}'...")
        
        exported_path = model.export(**export_params)
        
        export_time = (datetime.now() - start_time).total_seconds()
        
        # Розмір файлу
        if exported_path and os.path.exists(exported_path):
            if os.path.isfile(exported_path):
                size_mb = os.path.getsize(exported_path) / (1024 * 1024)
            else:
                # Для директорій
                total_size = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, files in os.walk(exported_path)
                    for f in files
                )
                size_mb = total_size / (1024 * 1024)
        else:
            size_mb = 0
        
        print()
        print("=" * 70)
        print("EXPORT COMPLETED")
        print("=" * 70)
        print(f"Output: {exported_path}")
        print(f"Size: {size_mb:.2f} MB")
        print(f"Time: {export_time:.2f} seconds")
        print("=" * 70)
        
        return exported_path
        
    except Exception as e:
        print(f"\nПомилка під час експорту: {e}")
        return None


def main():
    """Головна функція для запуску експорту."""
    result = export_model()
    return result


if __name__ == "__main__":
    main()
