import sys
import pandas as pd
from pathlib import Path
import re


def test_artifacts_exist_and_not_empty():
    root = Path(__file__).parent.parent
    artifacts_dir = root / 'artifacts'

    expected_files = [
            'recall_score.txt',
            'confusion_matrix.txt',
            'feature_importance.txt'
        ]
    for filename in expected_files:
        file_path = artifacts_dir / filename
        assert file_path.exists(), f"Файл {filename} не существует"
    
    # Проверка что файл не пустой
        file_size = file_path.stat().st_size
        assert file_size > 0, f"Файл {filename} пустой (0 байт)"
        # Проверка что файл можно прочитать
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 0, f"Файл {filename} не содержит данных"
        except Exception as e:
            assert False, f"Не удалось прочитать {filename}: {e}"
            
        print(f"{filename}: {file_size} байт")

        if filename == 'recall_score.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            match = re.search(r'Recall:\s*([0-9.]+)', content)

            assert match is not None, f"Не найден Recall в файле. Содержимое: {content}"
            recall_value = float(match.group(1))

            assert 0 <= recall_value <= 1, f"Recall = {recall_value} вне диапазона [0, 1]"
            assert recall_value > 0.8, f"Recall = {recall_value}, ожидается > 0.8"
        
            print(f"Recall = {recall_value} > 0.8")