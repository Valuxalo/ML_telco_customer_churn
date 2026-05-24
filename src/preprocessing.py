import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

def drop_columns(X):
    X = X.copy()
    columns_to_drop = ['Churn Category', 'Churn Reason', 'Churn Score', 'Customer ID', 'Customer Status', 'Dependents', 'Under 30', 
                       'Senior Citizen', 'Lat Long', 'Latitude', 'Longitude', 'City', 'Population', 'Zip Code',
                       'Referred a Friend', 'Married', 'Number of Referrals',  'Unlimited Data', 'Internet Type', 'Phone Service', 
                       'Streaming Music', 'Streaming Movies', 'Total Revenue', 'Total Charges', 'Monthly Charge', 'Country', 'Quarter', 'State']
    columns_to_drop = [col for col in columns_to_drop if col in X.columns]
    if columns_to_drop:
        X = X.drop(columns_to_drop, axis=1)
    return X

def fill_na(X):
    X = X.copy() 
    object_columns = X.select_dtypes(include=['object', 'category']).columns
    for col in object_columns:
        X[col] = X[col].fillna('0')
    
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mean())
    return X

def to_bin(X):
    X = X.copy()
    columns_to_bin = ['Offer', 'Number of Dependents', 'Total Extra Data Charges', 'Total Refunds']
    
    for col in columns_to_bin:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = (X[col] > 0).astype(int)
        else:
            X[col] = (
                X[col]
                .astype(str)
                .ne('0')
            ).astype(int)
    return X


def preprocessing():
    drop_transformer = FunctionTransformer(drop_columns)
    fill_na_transformer = FunctionTransformer(fill_na)
    to_bin_transformer = FunctionTransformer(to_bin)

    pipeline_data = Pipeline([
        ('drop_columns', drop_transformer),
        ('fill_na_transform', fill_na_transformer),
        ('create_features', to_bin_transformer),
    ])
    
    
    return pipeline_data

def process_all_files(raw_folder='raw', processed_folder='processed'):   
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_path = os.path.join(root_dir, 'data', raw_folder)
    save_path = os.path.join(root_dir, 'data', processed_folder)
    os.makedirs(save_path, exist_ok=True)
    # Получаем все файлы из папки raw
    all_files = [f for f in os.listdir(load_path) 
                 if os.path.isfile(os.path.join(load_path, f)) and f.endswith(".csv")]
    
    print(f"Найдено файлов: {len(all_files)}")
    
    pipeline_data = preprocessing()  
    for filename in all_files:
        print(f"Обработка: {filename}")
        # Полный путь к файлу
        file_path = os.path.join(load_path, filename)
            
        data = pd.read_csv(file_path)
        if 'train' in filename:
            X_train_processed = pipeline_data.fit_transform(data)
            X_train_processed.to_csv(os.path.join(save_path, 'eda_telco_customer_churn_train.csv'), index=False)
        elif 'validation' in filename:
            X_val_processed = pipeline_data.transform(data)
            X_val_processed.to_csv(os.path.join(save_path, 'eda_telco_customer_churn_val.csv'), index=False)
        elif 'test' in filename:
            X_test_processed = pipeline_data.transform(data)
            X_test_processed.to_csv(os.path.join(save_path, 'eda_telco_customer_churn_test.csv'), index=False)
    return X_train_processed, X_val_processed, X_test_processed


        

        
