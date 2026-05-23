[![CI Pipeline](https://github.com/Valuxalo/ML_telco_customer_churn/actions/workflows/ci.yml/badge.svg)](https://github.com/Valuxalo/ML_telco_customer_churn/actions/workflows/ci.yml)

## Тема
Прогнозирование оттока клиентов телекоммуникационных компаний

## Описание задачи
Для телекоммуникационных компаний очень важно удержать клиента, так как в условиях высокой конкуренции клиент легко может уйти к другому оператору связи или провайдеру. Для этого необходимо выявлять на раннем сроке клиентов, кто находится в группе риска расторжения договора с компанией, чтобы предложить им персонализированные условия (скидки, бонусы) и тем самым снизить потери абонентской базы.

## Данные
Ссылка на датасет: https://huggingface.co/datasets/aai510-group1/telco-customer-churn

Это объединённые набор данных об оттоке клиентов телекоммуникационных компаний, который имеет 51 признак, описывающие разные параметры абонентов, такие как демографические и финансовые факторы, использование услуг. Объём датасета - 7000 строк. 

## Инструменты

## Структура проекта

```telco_project/
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── EDA.ipynb
│   ├── load_data.ipynb
│   └── ML.ipynb
├── artifacts/
├── src/
│   ├── __init__.py
│   ├── main_pipeline.py
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── ml_model.py
│   ├── predict.py
│   └── save_model.py
├── tests/
│   ├── __init__.py
│   ├── test_load_data.py
│   ├── test_preprocessing.py
│   ├── test_save_model.py
│   └── test_predict.py
```

## Автор
Кулакова Валентина Валерьевна - студентка УрФУ направления "Инженерия машинного обучения"
