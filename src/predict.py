import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def predict(model, X_true, y_true):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = model.predict(X_true)
    catboost_recall = recall_score(y_true, y_pred)
    y_pred = cross_val_predict(model, X_true, y_true, cv=cv)
    report = classification_report(y_true, y_pred)
    feature_importance = model.get_feature_importance()
    feature_names = model.feature_names_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
        }).sort_values('importance', ascending=False)

    #сохранение артефактов
    with open('../artifacts/recall_score.txt', 'w', encoding='utf-8') as f:
        f.write(f"Recall: {catboost_recall}")
    with open('../artifacts/confusion_matrix.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    with open('../artifacts/feature_importance.txt', 'w', encoding='utf-8') as f:
        f.write("Топ-10 важных признаков (CatBoost):\n")
        f.write(importance_df.head(10).to_string(index=False))
