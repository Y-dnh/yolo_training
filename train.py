"""
Модуль для навчання YOLO моделі детекції на IR (інфрачервоних) зображеннях.
Класи: person, car, truck
Усі параметри конфігурації знаходяться на початку файлу.
"""

import os
import sys

# Фікс для правильного відображення tqdm у Windows PowerShell
if sys.platform == 'win32':
    os.system('')  # Включає ANSI escape sequences підтримку
    # Альтернативно можна вимкнути кольори якщо не допомагає:
    # os.environ['NO_COLOR'] = '1'

import random
import numpy as np
import torch
from ultralytics import YOLO
import ultralytics


# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ
# =============================================================================
SEED = 42
PROJECT_NAME = "yolov8x-p2_for_autolabelling"
PRETRAINED_MODEL = "yolov8x-p2.yaml"
# Шляхи до даних
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset_split")
PROJECT_DIR = os.path.join(BASE_DIR, PROJECT_NAME)
YAML_PATH = os.path.join(DATASET_ROOT, "yolo.yaml")


# ============================================================================
# ПАРАМЕТРИ НАВЧАННЯ (передаються як **kwargs до model.train())
# =============================================================================
TRAINING_CONFIG = {
    # Модель та дані
    "data": YAML_PATH,
    "project": PROJECT_DIR,
    "name": "baseline",
    "exist_ok": False,

    # ==========================================================================
    # ЗАГАЛЬНІ ПАРАМЕТРИ НАВЧАННЯ ДЛЯ IR (ТЕПЛОВІЗІЙНИХ) ЗОБРАЖЕНЬ
    # ==========================================================================
    "epochs": 100,          # Більше епох для кращої збіжності на IR даних
    "time": None,
    "patience": 10,        # Більше терпіння - IR дані можуть потребувати більше часу
    "batch": 4,            # Менший batch для YOLO11x (велика модель)
    "imgsz": 960,         # Стандартний розмір
    "save": True,
    "save_period": -1,
    "cache": False,
    "device": 0,
    "workers": 8,
    "seed": SEED,
    "deterministic": True,
    "single_cls": False,
    "classes": None,
    "rect": False,          # Rectangular training - зберігає aspect ratio
    "multi_scale": False,    # Multi-scale для кращої генералізації
    "cos_lr": False,        # Cosine LR scheduler - плавніше навчання
    "close_mosaic": 10,     # Вимкнути mosaic пізніше для fine-tuning
    "resume": False,
    "amp": False,
    "fraction": 1.0,
    "profile": False,
    "freeze": None,
    "val": True,
    "plots": True,
    "compile": False,

# ==========================================================================
    # ОПТИМІЗАТОР ТА LEARNING RATE ДЛЯ IR ЗОБРАЖЕНЬ
    # ==========================================================================
    "pretrained": False,
    "optimizer": "AdamW",
    "lr0": 0.001,           
    "lrf": 0.01,
    "momentum": 0.937,        # AdamW зазвичай має beta1=0.9
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,   
    "warmup_momentum": 0.5, 
    "warmup_bias_lr": 0.01, 

    # ==========================================================================
    # ВАГИ ФУНКЦІЙ ВТРАТ ДЛЯ IR ДЕТЕКЦІЇ
    # ==========================================================================
    "box": 7.5,             # Вага box loss
    "cls": 1.0,             # Збільшено - важливо розрізняти person/car/truck на IR
    "dfl": 1.5,             # Distribution focal loss
    "pose": 12.0,
    "kobj": 1.0,
    "nbs": 64,
    "overlap_mask": True,
    "mask_ratio": 4,
    "dropout": 0.0,
    "label_smoothing": 0.0,

    # ==========================================================================
    # АУГМЕНТАЦІЯ ДЛЯ ІНФРАЧЕРВОНИХ (ТЕПЛОВІЗІЙНИХ) ЗОБРАЖЕНЬ
    # ==========================================================================
    # HSV аугментації вимкнені - зображення вже grayscale
    "hsv_h": 0.0,           # Вимкнено - немає кольору
    "hsv_s": 0.0,           # Вимкнено - немає насиченості
    "hsv_v": 0.3,           # Невелика варіація яскравості (контраст тепловізора може змінюватись)
    
    # Геометричні трансформації
    "degrees": 5.0,         # Мінімальний поворот - камера на мачті стабільна
    "translate": 0.15,      # Зсув зображення
    "scale": 0.4,           # Масштабування (об'єкти на різних відстанях)
    "shear": 2.0,           # Невеликий зсув перспективи
    "perspective": 0.0,     # Вимкнено - фіксована висота камери
    
    # Відображення
    "flipud": 0.0,          # Вимкнено - камера завжди зверху
    "fliplr": 0.5,          # Горизонтальне відображення
    "bgr": 0.0,             # Вимкнено - grayscale
    
    # Композитні аугментації
    "mosaic": 1.0,          # Mosaic аугментація
    "mixup": 0.0,           # Невеликий mixup для регуляризації
    "cutmix": 0.0,          # Вимкнено
    "copy_paste": 0.0,      # Невеликий copy-paste для малих об'єктів (person)
    "copy_paste_mode": "flip",
    
    # Інші аугментації
    "auto_augment": "",     # Вимкнено - стандартні аугментації не для IR
    "erasing": 0.3,         # Random erasing для регуляризації
}


def setup_seed(seed: int) -> None:
    """Ініціалізація seed для відтворюваності результатів."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_environment() -> str:
    """Налаштування середовища та створення директорій."""
    ultralytics.checks()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(PROJECT_DIR, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Project Directory: {PROJECT_DIR}")
    
    return device


def train_model(
    model_path: str = PRETRAINED_MODEL,
    **kwargs
) -> tuple:
    """
    Навчання YOLO моделі.
    
    Args:
        model_path: Шлях до попередньо навченої моделі
        **kwargs: Параметри навчання (перезаписують TRAINING_CONFIG)
    
    Returns:
        tuple: (results, trained_model_path)
    """
    # Об'єднуємо конфігурацію з переданими параметрами
    config = {**TRAINING_CONFIG, **kwargs}
    
    print(f"Завантаження моделі: {model_path}")
    model = YOLO(model_path)
    
    print("Початок навчання...")
    print(f"Конфігурація: epochs={config['epochs']}, batch={config['batch']}, imgsz={config['imgsz']}")
    
    # Навчання моделі
    results = model.train(**config)
    
    # Шлях до найкращої моделі
    trained_model_path = os.path.join(
        config["project"],
        config["name"],
        "weights",
        "best.pt"
    )
    
    print("Навчання завершено!")
    print(f"Найкраща модель збережена: {trained_model_path}")
    
    return results, trained_model_path


def main():
    """Головна функція для запуску навчання."""
    # Налаштування seed
    setup_seed(SEED)
    
    # Налаштування середовища
    device = setup_environment()
    
    # Оновлення device в конфігурації якщо потрібно
    training_kwargs = {"device": 0 if device == "cuda" else "cpu"}
    
    # Запуск навчання
    results, model_path = train_model(**training_kwargs)
    
    return results, model_path


if __name__ == "__main__":
    main()
