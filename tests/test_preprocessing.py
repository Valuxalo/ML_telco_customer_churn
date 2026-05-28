import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_preprocessing():
    root = Path(__file__).parent.parent
    processed_folder = root / 'data' / 'processed'
    
    assert processed_folder.exists(), f"Папка {processed_folder} не существует"
    
    csv_files = list(processed_folder.glob('*.csv'))
    assert len(csv_files) > 0, f"Нет CSV файлов в {processed_folder}"
    print('\n')
    for file in csv_files:
        df = pd.read_csv(file)
        total_nan = df.isnull().sum().sum()
        
        assert total_nan == 0, f"Файл {file.name} содержит {total_nan} NaN значений"
        print(f"{file.name}: NaN = {total_nan}")
        
        num_col = df.shape[1]
        assert num_col == 24, f"Файл {file.name} содержит {num_col} столбцов"
        print(f"{file.name}: columns = {num_col}")

