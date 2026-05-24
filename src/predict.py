import pandas as pd
import numpy as np
import os
import mlflow
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from dotenv import load_dotenv

load_dotenv()

def predict(model, df_true):
    model_name = os.getenv("MODEL_NAME")
    X_true = df_true.drop(columns=['Churn'])
    y_true = df_true['Churn']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = model.predict(X_true)

    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    y_pred = cross_val_predict(model, X_true, y_true, cv=cv)
    report = classification_report(y_true, y_pred)
    if model_name == 'CatBoost':
        feature_importance = model.get_feature_importance()
        feature_names = model.feature_names_

    elif model_name == 'LogisticRegression':
        lr = model.named_steps["model"]
        feature_importance = lr.coef_[0]
        feature_names = model.named_steps["preprocessing"].get_feature_names_out()

    elif model_name == 'RandomForest':
        rf = model.named_steps["model"]
        feature_names = model.named_steps["preprocessing"].get_feature_names_out()
        feature_importance = rf.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
        }).sort_values('importance', ascending=False)
    
    #сохранение артефактов
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f'{root_dir}/artifacts/recall_score.txt')
    with open(f'{root_dir}/artifacts/recall_score.txt', 'w', encoding='utf-8') as f:
        f.write(f"Recall: {recall}")
    with open(f'{root_dir}/artifacts/confusion_matrix.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    with open(f'{root_dir}/artifacts/feature_importance.txt', 'w', encoding='utf-8') as f:
        f.write("Топ-10 важных признаков:\n")
        f.write(importance_df.head(10).to_string(index=False))

    mlflow.log_metric(
        "recall",
        recall
    )
    mlflow.log_metric(
        "accuracy",
        accuracy
    )
    mlflow.log_metric(
        "f1",
        f1
    )
    mlflow.log_text(report, "classification_report.txt")
    mlflow.log_artifact(
    "artifacts/feature_importance.txt"
)