"""
Модуль для конвертації анотацій COCO формату в YOLO формат.
Конвертує тільки лейбли без створення структури проекту.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm


# =============================================================================
# КОНФІГУРАЦІЯ
# =============================================================================
CONFIG = {
    # Вхідні дані
    "labels_dir": "D:/yolo_training/train_dpsu_data_rfdetr/val",      # Папка з JSON файлами COCO анотацій
    "json_file": None,                         # Конкретний JSON файл (якщо None - всі файли в папці)
    
    # Вихідні дані
    "save_dir": "D:/yolo_training/train_dpsu_data_rfdetr/labels",             # Папка для збереження YOLO лейблів
    
    # Параметри конвертації
    "use_segments": False,                     # Чи включати сегментаційні маски
    "use_keypoints": False,                    # Чи включати keypoints
    "cls91to80": False,                        # Чи мапити 91 COCO класів до 80
    "class_offset": 0,                         # Зсув ID класів (1 для COCO, 0 якщо класи вже з 0)
}


def coco91_to_coco80_class() -> list:
    """
    Конвертація 91 COCO class IDs до 80 COCO class IDs.
    
    Returns:
        list: Список маппінгу класів
    """
    return [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, None, 24, 25, None, None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67,
        68, 69, 70, 71, 72, None, 73, 74, 75, 76, 77, 78, 79, None,
    ]


def merge_multi_segment(segments: list) -> list:
    """
    Об'єднання кількох сегментів в один.
    
    Args:
        segments: Список сегментів
        
    Returns:
        list: Об'єднані сегменти
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Знаходимо найближчі точки між сегментами
    for i in range(1, len(segments)):
        idx1, idx2 = min(
            [(idx1, idx2) for idx1, p1 in enumerate(segments[i - 1]) for idx2, p2 in enumerate(segments[i])],
            key=lambda x: np.linalg.norm(segments[i - 1][x[0]] - segments[i][x[1]])
        )
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Об'єднуємо сегменти
    for idx, seg in enumerate(segments):
        if len(idx_list[idx]) == 2:
            idx1, idx2 = idx_list[idx]
            s.append(seg[idx2:idx1 + 1][::-1] if idx1 >= idx2 else np.concatenate([seg[idx2:], seg[:idx1 + 1]])[::-1])
        elif len(idx_list[idx]) == 1:
            s.append(seg[idx_list[idx][0]:] if idx == 0 else seg[:idx_list[idx][0] + 1][::-1])
        else:
            s.append(seg)
            
    return s


def convert_coco_to_yolo(
    labels_dir: str = None,
    json_file: str = None,
    save_dir: str = None,
    use_segments: bool = None,
    use_keypoints: bool = None,
    cls91to80: bool = None,
    class_offset: int = None,
    **kwargs
) -> None:
    """
    Конвертація анотацій COCO у формат YOLO.
    
    Args:
        labels_dir: Папка з JSON файлами COCO анотацій
        json_file: Конкретний JSON файл (якщо None - всі файли в папці)
        save_dir: Папка для збереження YOLO лейблів
        use_segments: Чи включати сегментаційні маски
        use_keypoints: Чи включати keypoints
        cls91to80: Чи мапити 91 COCO класів до 80
        class_offset: Зсув ID класів (1 для COCO)
    """
    # Застосовуємо значення за замовчуванням з CONFIG
    labels_dir = labels_dir or CONFIG["labels_dir"]
    json_file = json_file or CONFIG["json_file"]
    save_dir = save_dir or CONFIG["save_dir"]
    use_segments = use_segments if use_segments is not None else CONFIG["use_segments"]
    use_keypoints = use_keypoints if use_keypoints is not None else CONFIG["use_keypoints"]
    cls91to80 = cls91to80 if cls91to80 is not None else CONFIG["cls91to80"]
    class_offset = class_offset if class_offset is not None else CONFIG["class_offset"]
    
    labels_dir = Path(labels_dir)
    save_dir = Path(save_dir)
    
    # Створюємо папку для лейблів
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Маппінг класів
    coco80 = coco91_to_coco80_class()
    
    # Визначаємо файли для обробки
    if json_file:
        json_files = [Path(json_file)]
    else:
        json_files = sorted(labels_dir.resolve().glob("*.json"))
    
    if not json_files:
        print(f"Не знайдено JSON файлів у {labels_dir}")
        return
    
    total_images = 0
    total_annotations = 0
    
    for jf in json_files:
        print(f"\nОбробка: {jf.name}")
        
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        
        # Словник зображень
        images = {x["id"]: x for x in data["images"]}
        
        # Групуємо анотації по зображеннях
        annotations = defaultdict(list)
        for ann in data["annotations"]:
            annotations[ann["image_id"]].append(ann)
        
        # Обробляємо кожне зображення
        for img_id, anns in tqdm(annotations.items(), desc="Конвертація"):
            img = images.get(img_id)
            if img is None:
                continue
                
            h, w = img["height"], img["width"]
            filename = Path(img["file_name"]).stem
            
            bboxes = []
            segments = []
            keypoints_list = []
            
            for ann in anns:
                # Пропускаємо crowd анотації
                if ann.get("iscrowd", False):
                    continue
                
                # COCO bbox формат: [x, y, width, height] (top-left corner)
                box = np.array(ann["bbox"], dtype=np.float64)
                
                # Конвертуємо в центр
                box[:2] += box[2:] / 2  # xy top-left -> center
                
                # Нормалізуємо
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                
                # Перевіряємо валідність
                if box[2] <= 0 or box[3] <= 0:
                    continue
                
                # Визначаємо клас
                if cls91to80:
                    cls = coco80[ann["category_id"] - 1]
                    if cls is None:
                        continue
                else:
                    cls = ann["category_id"] - class_offset
                
                box_line = [cls, *box.tolist()]
                
                if box_line not in bboxes:
                    bboxes.append(box_line)
                    
                    # Сегментація
                    if use_segments and ann.get("segmentation"):
                        if len(ann["segmentation"]) == 0:
                            segments.append([])
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        segments.append([cls, *s])
                    
                    # Keypoints
                    if use_keypoints and ann.get("keypoints"):
                        kpts = (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                        keypoints_list.append(box_line + kpts)
            
            # Записуємо у файл
            if bboxes:
                output_file = save_dir / f"{filename}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    for i in range(len(bboxes)):
                        if use_keypoints and i < len(keypoints_list):
                            line = keypoints_list[i]
                        elif use_segments and i < len(segments) and len(segments[i]) > 0:
                            line = segments[i]
                        else:
                            line = bboxes[i]
                        
                        f.write(" ".join(f"{x:.6f}" if isinstance(x, float) else str(int(x)) for x in line) + "\n")
                
                total_annotations += len(bboxes)
            
            total_images += 1
    
    print(f"\n{'=' * 50}")
    print(f"Конвертація завершена!")
    print(f"Оброблено зображень: {total_images}")
    print(f"Всього анотацій: {total_annotations}")
    print(f"Результати збережено: {save_dir.resolve()}")
    print(f"{'=' * 50}")


def main(**kwargs):
    """Головна функція для запуску конвертації."""
    convert_coco_to_yolo(**kwargs)


if __name__ == "__main__":
    main()
