import pytest
import pandas as pd
from pathlib import Path


def test_load_data_with_valid_file():
    root = Path(__file__).parent.parent
    raw_folder = root / 'data' / 'raw'
    
    assert raw_folder.exists(), f"Папка {raw_folder} не существует"
    assert raw_folder.is_dir(), f"{raw_folder} не является папкой"
    
    csv_files = list(raw_folder.glob('*.csv'))
    
    # Проверяем что есть хотя бы один CSV файл
    assert len(csv_files) > 0, f"В папке {raw_folder} нет CSV файлов"
    
    print(f"\nНайдено CSV файлов: {len(csv_files)}")
    for file in csv_files:
        print(f"   - {file.name}")
    