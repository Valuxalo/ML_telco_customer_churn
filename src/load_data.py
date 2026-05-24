import pandas as pd
import os

def load_data(load_path=None, raw_folder=None):
    splits = {'train': 'train.csv', 'validation': 'validation.csv', 'test': 'test.csv'}
    if load_path is None:
        load_path = "hf://datasets/aai510-group1/telco-customer-churn/"
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(root_dir, 'data', raw_folder)
    os.makedirs(save_path, exist_ok=True)
    print(save_path)
    name_file = 'raw_eda_telco_customer_churn_'

    for key in splits.keys():
        df = pd.read_csv(load_path + splits[key])
        f = name_file + key + '.csv'
        df.to_csv(os.path.join(save_path, f), index=False)
    return True