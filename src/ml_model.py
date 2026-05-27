import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

import mlflow
import mlflow.catboost
import os
import time

from dotenv import load_dotenv

load_dotenv()

def train_model(df_train, df_val):
    model_name = os.getenv("MODEL_NAME")
    
    X_train = df_train.drop(columns=['Churn'])
    y_train = df_train['Churn']
    X_val = df_val.drop(columns=['Churn'])
    y_val = df_val['Churn']

    categorical = ['Contract', 'Device Protection Plan', 'Gender', 'Internet Service', 'Multiple Lines', 'Offer', 
              'Online Backup', 'Online Security', 'Paperless Billing', 'Partner',
              'Payment Method', 'Premium Tech Support', 'Streaming TV', 'Total Refunds',
              'Number of Dependents', 'Total Extra Data Charges']

    numeric = [col for col in X_train.columns if col not in categorical]
    num_transformer = Pipeline([
            ('scaler', StandardScaler())])
    cat_transformer = Pipeline([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numeric),
        ('cat', cat_transformer, categorical)
    ])
   
    models = {
        "CatBoost": CatBoostClassifier(
                cat_features=tuple(categorical),
                random_seed=42,
                verbose=0,
                thread_count=-1,
                auto_class_weights='Balanced',
                early_stopping_rounds=50,
                eval_metric='Recall',
                allow_writing_files=False
        ),

        "RandomForest": Pipeline([
            ("preprocessing", preprocessor),
            ("model", RandomForestClassifier(
            class_weight='balanced',
            max_depth=10, 
            max_features='log2',
            min_samples_leaf=1,
            min_samples_split=5,
            n_estimators=50,
            random_state=42
        ))
        ]),

        "LogisticRegression": Pipeline([
            ("preprocessing", preprocessor),
            ("model", LogisticRegression(
                C=1,
                l1_ratio=1,
                max_iter=1000,
                solver='saga',
                random_state=42
        ))
        ]),
    }
    start = time.time()

    model = models.get(model_name)
    if model_name != 'CatBoost':
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train,
                            eval_set=(X_val, y_val))
    try: 
        mlflow.log_params(model.get_all_params())
    except Exception:
        mlflow.log_params(model.get_params())

    train_time = time.time() - start

    if model_name == 'CatBoost':
        mlflow.catboost.log_model(
            model,
            name="model"
        )
    else:
        mlflow.sklearn.log_model(
            model, 
            name="model")
        
    mlflow.log_metric("train_time_sec", train_time)
    
    return model