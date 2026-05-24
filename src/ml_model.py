import pandas as pd
import numpy as np

from catboost import CatBoostClassifier

def train_model(df_train, df_val):
    categorical = ['Contract', 'Device Protection Plan', 'Gender', 'Internet Service', 'Multiple Lines', 'Offer', 
              'Online Backup', 'Online Security', 'Paperless Billing', 'Partner',
              'Payment Method', 'Premium Tech Support', 'Streaming TV', 'Total Refunds',
              'Number of Dependents', 'Total Extra Data Charges']
    X_train = df_train.drop(columns=['Churn'])
    y_train = df_train['Churn']
    X_val = df_val.drop(columns=['Churn'])
    y_val = df_val['Churn']
    catboost_model = CatBoostClassifier(
            cat_features=tuple(categorical),
            random_seed=42,
            verbose=0,
            thread_count=-1,
            auto_class_weights='Balanced',
            early_stopping_rounds=50,
            eval_metric='Recall',
            allow_writing_files=False
        )

    catboost_model.fit(X_train, y_train, 
                        eval_set=(X_val, y_val))
    
    return catboost_model