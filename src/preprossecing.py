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
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = (X[col] != '0').astype(int)
        else:
            X[col] = (X[col] > 0).astype(int)
    return X


def preprossecing(X):
    drop_transformer = FunctionTransformer(drop_columns)
    fill_na_transformer = FunctionTransformer(fill_na)
    to_bin_transformer = FunctionTransformer(to_bin)

    pipeline_data = Pipeline([
        ('drop_columns', drop_transformer),
        ('log_transform', fill_na_transformer),
        ('create_features', to_bin_transformer),
    ])
    X_processed = pipeline_data.fit_transform(X)
    
    return X_processed
