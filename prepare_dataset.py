"""
Скрипт для підготовки датасету YOLO:
1. Спліт на train/val/test
2. Створення yolo.yaml конфігурації
"""

import os
import shutil
import random
from pathlib import Path


# =============================================================================
# КОНФІГУРАЦІЯ
# =============================================================================
SEED = 42

# Пропорції спліту
TRAIN_RATIO = 0.8
VAL_RATIO = 0.15
TEST_RATIO = 0.05  # Якщо не потрібен test - постав 0.0

# Шляхи
BASE_DIR = Path(__file__).parent
SOURCE_DIR = BASE_DIR / "dataset"
SOURCE_IMAGES = SOURCE_DIR / "images"
SOURCE_LABELS = SOURCE_DIR / "labels"

# Класи датасету
CLASSES = {
    0: "person",
    1: "car",
    2: "truck",
}

# Розширення файлів
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =============================================================================
# ФУНКЦІЇ
# =============================================================================
def get_image_files(images_dir: Path) -> list[Path]:
    """Отримує список всіх зображень з директорії."""
    files = []
    for f in images_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(f)
    return sorted(files)


def split_dataset(
    image_files: list[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42
) -> tuple[list[Path], list[Path], list[Path]]:
    """Розбиває датасет на train/val/test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Сума пропорцій має дорівнювати 1.0, отримано: {train_ratio + val_ratio + test_ratio}"
    
    random.seed(seed)
    shuffled = image_files.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]
    
    return train_files, val_files, test_files


def copy_files(
    image_files: list[Path],
    source_labels_dir: Path,
    dest_images_dir: Path,
    dest_labels_dir: Path,
    split_name: str
) -> int:
    """Копіює зображення та відповідні лейбли в цільову директорію."""
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    missing_labels = []
    
    for img_path in image_files:
        # Шлях до лейбла
        label_name = img_path.stem + ".txt"
        label_path = source_labels_dir / label_name
        
        # Копіюємо зображення
        shutil.copy2(img_path, dest_images_dir / img_path.name)
        
        # Копіюємо лейбл (якщо існує)
        if label_path.exists():
            shutil.copy2(label_path, dest_labels_dir / label_name)
            copied += 1
        else:
            missing_labels.append(img_path.name)
    
    if missing_labels:
        print(f"  [!] {split_name}: {len(missing_labels)} зображень без лейблів")
    
    return copied


def create_yaml(
    output_path: Path,
    dataset_path: Path,
    classes: dict[int, str],
    train_path: str = "train/images",
    val_path: str = "val/images",
    test_path: str = "test/images",
    include_test: bool = True
) -> None:
    """Створює YAML конфігурацію для YOLO."""
    # Перетворюємо шлях на формат з прямими слешами
    dataset_path_str = str(dataset_path.resolve()).replace("\\", "/")
    
    lines = [
        "# YOLO Dataset Configuration",
        f"path: {dataset_path_str}",
        f"train: {train_path}",
        f"val: {val_path}",
    ]
    
    if include_test:
        lines.append(f"test: {test_path}")
    
    lines.extend([
        "",
        "# Класи",
        f"nc: {len(classes)}",
        "",
        "names:",
    ])
    
    for idx in sorted(classes.keys()):
        lines.append(f"  {idx}: {classes[idx]}")
    
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_dataset(
    source_images: Path,
    source_labels: Path,
    output_dir: Path,
    classes: dict[int, str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42
) -> dict:
    """Головна функція підготовки датасету."""
    print(f"Джерело зображень: {source_images}")
    print(f"Джерело лейблів: {source_labels}")
    print(f"Вихідна директорія: {output_dir}")
    print()
    
    # Отримуємо список зображень
    image_files = get_image_files(source_images)
    print(f"Знайдено {len(image_files)} зображень")
    
    if not image_files:
        raise ValueError("Не знайдено жодного зображення!")
    
    # Спліт
    train_files, val_files, test_files = split_dataset(
        image_files, train_ratio, val_ratio, test_ratio, seed
    )
    
    print(f"Спліт: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    print()
    
    # Копіюємо файли
    splits = [
        ("train", train_files),
        ("val", val_files),
    ]
    if test_ratio > 0:
        splits.append(("test", test_files))
    
    stats = {}
    for split_name, files in splits:
        print(f"Копіювання {split_name}...")
        dest_images = output_dir / split_name / "images"
        dest_labels = output_dir / split_name / "labels"
        
        copied = copy_files(files, source_labels, dest_images, dest_labels, split_name)
        stats[split_name] = {"total": len(files), "with_labels": copied}
        print(f"  {split_name}: {len(files)} зображень, {copied} з лейблами")
    
    print()
    
    # Створюємо YAML
    yaml_path = output_dir / "yolo.yaml"
    create_yaml(
        yaml_path,
        output_dir,
        classes,
        include_test=(test_ratio > 0)
    )
    print(f"YAML конфігурація: {yaml_path}")
    
    return stats


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Директорія для готового датасету
    OUTPUT_DIR = BASE_DIR / "dataset_split"
    
    # Видаляємо стару директорію якщо існує
    if OUTPUT_DIR.exists():
        response = input(f"Директорія {OUTPUT_DIR} існує. Видалити? [y/N]: ")
        if response.lower() == "y":
            shutil.rmtree(OUTPUT_DIR)
            print("Видалено.")
        else:
            print("Скасовано.")
            exit(0)
    
    # Запускаємо підготовку
    stats = prepare_dataset(
        source_images=SOURCE_IMAGES,
        source_labels=SOURCE_LABELS,
        output_dir=OUTPUT_DIR,
        classes=CLASSES,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )
    
    print()
    print("=" * 50)
    print("Готово!")
    print(f"Датасет збережено в: {OUTPUT_DIR}")
    print()
    print("Для використання в train.py оновіть шлях:")
    print(f'  DATASET_ROOT = os.path.join(BASE_DIR, "dataset_split")')
