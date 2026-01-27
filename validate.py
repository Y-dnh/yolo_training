"""
Модуль для тестування/валідації YOLO моделі детекції на IR зображеннях.
Усі параметри конфігурації знаходяться на початку файлу.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO


# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ
# =============================================================================
PROJECT_NAME = "yolov8x-p2_for_autolabelling"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, PROJECT_NAME)
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")
YAML_PATH = os.path.join(DATASET_ROOT, "yolo.yaml")
EXPERIMENT_NAME = "validation_test"

TRAINED_MODEL_PATH = os.path.join(PROJECT_DIR, "baseline", "weights", "best.pt")

# Класи датасету
CLASSES = {
    0: "person",
    1: "car",
    2: "truck",
}


# =============================================================================
# ПАРАМЕТРИ ВАЛІДАЦІЇ (передаються як **kwargs до model.val())
# =============================================================================
VALIDATION_CONFIG = {
    # Параметри датасету
    "data": YAML_PATH,
    "split": "test",
    
    # Параметри детекції
    "conf": 0.5,
    "iou": 0.5,
    "imgsz": 1920,
    "device": None,  # Автоматичне визначення
    "batch": 2,  # Зменшено для економії пам'яті
    "max_det": 300,
    
    # Параметри обробки
    "rect": True,
    "half": True,
    "augment": False,
    "agnostic_nms": False,
    "classes": None,
    "single_cls": False,
    "dnn": False,
    
    # Параметри виводу
    "save_json": True,
    "save_txt": False,
    "save_conf": True,
    "plots": True,
    "verbose": False,
    "workers": 8,  # 0 щоб уникнути multiprocessing та проблем з пам'яттю
    
    # Візуалізація
    "visualize": True,
    
    # Налаштування проекту
    "project": PROJECT_DIR,
    "name": EXPERIMENT_NAME,
}


def setup_device() -> str:
    """Визначення доступного пристрою."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    return device


def validate_model(
    model_path: str = TRAINED_MODEL_PATH,
    **kwargs
) -> object:
    """
    Валідація YOLO моделі на тестовому датасеті.
    
    Args:
        model_path: Шлях до навченої моделі
        **kwargs: Параметри валідації (перезаписують VALIDATION_CONFIG)
    
    Returns:
        object: Результати валідації
    """
    # Об'єднуємо конфігурацію з переданими параметрами
    config = {**VALIDATION_CONFIG, **kwargs}
    
    # Автоматичне визначення device якщо не вказано
    if config["device"] is None:
        config["device"] = setup_device()
    
    print(f"Завантаження моделі: {model_path}")
    model = YOLO(model_path)
    
    print(f"Запуск валідації на split='{config['split']}'...")
    print(f"Конфігурація: conf={config['conf']}, iou={config['iou']}, imgsz={config['imgsz']}")
    
    # Запуск валідації
    results = model.val(**config)
    
    print("Валідація завершена!")
    
    return results


def extract_metrics(validation_results: object) -> dict:
    """
    Витягування метрик з результатів валідації.
    
    Args:
        validation_results: Результати валідації від model.val()
    
    Returns:
        dict: Словник з метриками
    """
    metrics = {
        # Основні метрики
        "mAP50": float(validation_results.box.map50) if hasattr(validation_results.box, "map50") else 0.0,
        "mAP50-95": float(validation_results.box.map) if hasattr(validation_results.box, "map") else 0.0,
        "precision": float(validation_results.box.mp) if hasattr(validation_results.box, "mp") else 0.0,
        "recall": float(validation_results.box.mr) if hasattr(validation_results.box, "mr") else 0.0,
        
        # Метадані
        "validation_date": datetime.now().isoformat(),
        "conf_threshold": VALIDATION_CONFIG["conf"],
        "iou_threshold": VALIDATION_CONFIG["iou"],
        "image_size": VALIDATION_CONFIG["imgsz"],
    }
    
    # Per-class метрики якщо доступні
    if hasattr(validation_results.box, "maps") and validation_results.box.maps is not None:
        metrics["per_class_mAP"] = [float(m) for m in validation_results.box.maps]
    
    return metrics


def save_results(
    validation_results: object,
    metrics: dict,
    model_path: str = TRAINED_MODEL_PATH,
    output_dir: str = None
) -> dict:
    """
    Збереження результатів валідації.
    
    Args:
        validation_results: Результати валідації
        metrics: Витягнуті метрики
        model_path: Шлях до моделі (для звіту)
        output_dir: Директорія для збереження (за замовчуванням PROJECT_DIR/results)
    
    Returns:
        dict: Словник зі шляхами до збережених файлів
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_DIR, EXPERIMENT_NAME)
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    # Збереження метрик у JSON
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    saved_files["metrics_json"] = metrics_path
    
    # Збереження результатів валідації у JSON
    validation_json_path = os.path.join(output_dir, "validation_results.json")
    validation_json = validation_results.to_json()
    with open(validation_json_path, "w", encoding="utf-8") as f:
        f.write(validation_json)
    saved_files["validation_json"] = validation_json_path
    
    # Збереження результатів валідації у CSV
    validation_csv_path = os.path.join(output_dir, "validation_results.csv")
    validation_csv = validation_results.to_csv()
    with open(validation_csv_path, "w", encoding="utf-8") as f:
        f.write(validation_csv)
    saved_files["validation_csv"] = validation_csv_path
    
    # Генерація markdown звіту
    report_path = generate_markdown_report(
        validation_results=validation_results,
        metrics=metrics,
        model_path=model_path,
        output_dir=output_dir
    )
    saved_files["markdown_report"] = report_path
    
    print(f"\nРезультати збережено:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")
    
    return saved_files


def print_metrics(metrics: dict) -> None:
    """Виведення метрик у консоль."""
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТИ ТЕСТУВАННЯ")
    print("=" * 50)
    print(f"  mAP@0.5:      {metrics.get('mAP50', 0):.4f}")
    print(f"  mAP@0.5:0.95: {metrics.get('mAP50-95', 0):.4f}")
    print(f"  Precision:    {metrics.get('precision', 0):.4f}")
    print(f"  Recall:       {metrics.get('recall', 0):.4f}")
    print("=" * 50)


def generate_markdown_report(
    validation_results: object,
    metrics: dict,
    model_path: str,
    output_dir: str = None
) -> str:
    """
    Генерація детального markdown звіту.
    
    Args:
        validation_results: Результати валідації від model.val()
        metrics: Словник з метриками
        model_path: Шлях до моделі
        output_dir: Директорія для збереження звіту
    
    Returns:
        str: Шлях до збереженого звіту
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_DIR, EXPERIMENT_NAME)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Отримуємо додаткову інформацію
    model_name = Path(model_path).stem
    
    # Per-class метрики
    per_class_metrics = []
    if hasattr(validation_results.box, "maps") and validation_results.box.maps is not None:
        maps = validation_results.box.maps
        for i, (class_id, class_name) in enumerate(CLASSES.items()):
            if i < len(maps):
                per_class_metrics.append({
                    "id": class_id,
                    "name": class_name,
                    "mAP50": float(maps[i]) if maps[i] is not None else 0.0
                })
    
    # Отримуємо precision/recall per class якщо доступні
    if hasattr(validation_results.box, "p") and validation_results.box.p is not None:
        for i, pc in enumerate(per_class_metrics):
            if i < len(validation_results.box.p):
                pc["precision"] = float(validation_results.box.p[i])
    
    if hasattr(validation_results.box, "r") and validation_results.box.r is not None:
        for i, pc in enumerate(per_class_metrics):
            if i < len(validation_results.box.r):
                pc["recall"] = float(validation_results.box.r[i])
    
    # Швидкість інференсу
    speed_info = {}
    if hasattr(validation_results, "speed"):
        speed = validation_results.speed
        speed_info = {
            "preprocess": speed.get("preprocess", 0),
            "inference": speed.get("inference", 0),
            "postprocess": speed.get("postprocess", 0),
        }
        total_time = sum(speed_info.values())
        speed_info["total"] = total_time
        speed_info["fps"] = 1000 / total_time if total_time > 0 else 0
    
    # Формуємо звіт
    report_content = f"""# YOLO Model Evaluation Report

## Experiment Overview

| **Parameter** | **Value** |
|---------------|-----------|
| **Model** | `{model_name}` |
| **Model Path** | `{model_path}` |
| **Date & Time** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
| **Dataset** | IR Thermal Images |
| **Dataset Config** | `{YAML_PATH}` |
| **Split** | {VALIDATION_CONFIG['split']} |

## Configuration Settings

| **Setting** | **Value** |
|-------------|-----------|
| **Confidence Threshold** | {VALIDATION_CONFIG['conf']} |
| **IoU Threshold** | {VALIDATION_CONFIG['iou']} |
| **Image Size** | {VALIDATION_CONFIG['imgsz']} |
| **Batch Size** | {VALIDATION_CONFIG['batch']} |
| **Max Detections** | {VALIDATION_CONFIG['max_det']} |
| **Half Precision** | {VALIDATION_CONFIG['half']} |
| **Augmentation** | {VALIDATION_CONFIG['augment']} |

## Overall Performance

| **Metric** | **Value** |
|------------|-----------|
| **mAP@0.5** | {metrics.get('mAP50', 0):.4f} |
| **mAP@0.5:0.95** | {metrics.get('mAP50-95', 0):.4f} |
| **Precision** | {metrics.get('precision', 0):.4f} |
| **Recall** | {metrics.get('recall', 0):.4f} |
| **F1-Score** | {2 * metrics.get('precision', 0) * metrics.get('recall', 0) / (metrics.get('precision', 0) + metrics.get('recall', 0) + 1e-6):.4f} |

"""
    
    # Per-class performance
    if per_class_metrics:
        report_content += """## Per-Class Performance

| **Class** | **mAP@0.5** | **Precision** | **Recall** |
|-----------|-------------|---------------|------------|
"""
        for pc in per_class_metrics:
            prec = pc.get('precision', 0)
            rec = pc.get('recall', 0)
            report_content += f"| {pc['name']} | {pc['mAP50']:.4f} | {prec:.4f} | {rec:.4f} |\n"
        
        # Найкращий та найгірший клас
        best_class = max(per_class_metrics, key=lambda x: x['mAP50'])
        worst_class = min(per_class_metrics, key=lambda x: x['mAP50'])
        
        report_content += f"""
### Class Analysis
- **Best performing class**: {best_class['name']} (mAP@0.5 = {best_class['mAP50']:.4f})
- **Worst performing class**: {worst_class['name']} (mAP@0.5 = {worst_class['mAP50']:.4f})

"""
    
    # Speed metrics
    if speed_info:
        report_content += f"""## Inference Speed

| **Stage** | **Time (ms)** |
|-----------|---------------|
| Preprocess | {speed_info.get('preprocess', 0):.2f} |
| Inference | {speed_info.get('inference', 0):.2f} |
| Postprocess | {speed_info.get('postprocess', 0):.2f} |
| **Total** | **{speed_info.get('total', 0):.2f}** |
| **FPS** | **{speed_info.get('fps', 0):.1f}** |

"""
    
    # Performance interpretation
    map50 = metrics.get('mAP50', 0)
    if map50 >= 0.9:
        performance_level = "Excellent"
        performance_emoji = "🥇"
    elif map50 >= 0.7:
        performance_level = "Good"
        performance_emoji = "🥈"
    elif map50 >= 0.5:
        performance_level = "Moderate"
        performance_emoji = "🥉"
    else:
        performance_level = "Needs Improvement"
        performance_emoji = "⚠️"
    
    report_content += f"""## Performance Summary

{performance_emoji} **Overall Performance Level**: {performance_level}

### Key Findings
- Model achieves **{map50:.1%}** mAP@0.5 on the test set
- Precision of **{metrics.get('precision', 0):.1%}** indicates {"low" if metrics.get('precision', 0) < 0.7 else "acceptable" if metrics.get('precision', 0) < 0.85 else "good"} false positive rate
- Recall of **{metrics.get('recall', 0):.1%}** indicates {"many missed" if metrics.get('recall', 0) < 0.7 else "some missed" if metrics.get('recall', 0) < 0.85 else "few missed"} detections

### Recommendations
"""
    
    if metrics.get('precision', 0) < 0.7:
        report_content += "- Consider increasing confidence threshold to reduce false positives\n"
    if metrics.get('recall', 0) < 0.7:
        report_content += "- Consider decreasing confidence threshold to catch more objects\n"
    if map50 < 0.7:
        report_content += "- Model may benefit from more training data or longer training\n"
        report_content += "- Check for class imbalance in training data\n"
    
    report_content += f"""
## Output Files

The following files were generated during validation:
- `test_metrics.json` - Metrics in JSON format
- `validation_results.json` - Full validation results
- `validation_results.csv` - Results in CSV format
- `evaluation_report.md` - This report

### Visualization Outputs
Located in: `{VALIDATION_CONFIG['project']}/{VALIDATION_CONFIG['name']}/`
- `confusion_matrix.png` - Confusion matrix
- `confusion_matrix_normalized.png` - Normalized confusion matrix
- `F1_curve.png` - F1 score curve
- `P_curve.png` - Precision curve
- `R_curve.png` - Recall curve
- `PR_curve.png` - Precision-Recall curve
- `val_batch*_labels.jpg` - Ground truth visualizations
- `val_batch*_pred.jpg` - Prediction visualizations

---
*Report generated automatically by YOLO Validation System*  
*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Зберігаємо звіт
    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\nMarkdown report generated: {report_path}")
    
    return report_path


def main(
    model_path: str = TRAINED_MODEL_PATH,
    save_results_flag: bool = True,
    **kwargs
):
    """
    Головна функція для запуску валідації.
    
    Args:
        model_path: Шлях до навченої моделі
        save_results_flag: Чи зберігати результати у файли
        **kwargs: Додаткові параметри валідації
    """
    # Запуск валідації
    results = validate_model(model_path=model_path, **kwargs)
    
    # Витягування метрик
    metrics = extract_metrics(results)
    
    # Виведення метрик
    print_metrics(metrics)
    
    # Збереження результатів
    if save_results_flag:
        save_results(results, metrics, model_path=model_path)
    
    return results, metrics


if __name__ == "__main__":
    main()
