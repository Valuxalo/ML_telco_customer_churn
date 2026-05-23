import os
import sys
import pickle
import joblib
from pathlib import Path

def test_pkl_file_exists_in_artifacts():
    root = Path(__file__).parent.parent
    artifacts_dir = root / 'artifacts'

    pkl_files = list(artifacts_dir.glob('*.pkl'))

    assert len(pkl_files) > 0, f"В папке {artifacts_dir} нет .pkl файлов"

    print(f"Найдено .pkl файлов: {len(pkl_files)}")
    for pkl_file in pkl_files:
        print(f"   - {pkl_file.name}")

    for pkl_file in pkl_files:
        file_size = pkl_file.stat().st_size
        assert file_size > 0, f"Файл {pkl_file.name} пустой (0 байт)"

        print(f"{pkl_file.name}: {file_size} байт")

    try:
        # Пробуем загрузить модель
        model = joblib.load(pkl_file)
        assert model is not None, f"Не удалось загрузить {pkl_file.name}"
                
        # Проверяем что загруженный объект имеет метод predict
        assert hasattr(model, 'predict'), f"Загруженный объект не является моделью (нет метода predict)"
                
        print(f"{pkl_file.name}: успешно загружен, тип {type(model).__name__}")
    except Exception as e:
        assert False, f"Не удалось загрузить {pkl_file.name}: {e}"
