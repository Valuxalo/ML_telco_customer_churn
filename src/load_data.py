import pandas as pd
import os

def load_data():
    splits = {'train': 'train.csv', 'validation': 'validation.csv', 'test': 'test.csv'}
    load_path = "hf://datasets/aai510-group1/telco-customer-churn/"
    save_path = os.path.join('..', 'data', 'raw')
    name_file = 'raw_eda_telco_customer_churn_'
    try:
        for key in splits.keys():
            df = pd.read_csv(load_path + splits[key])
            f = name_file + key + '.csv'
            df.to_csv(os.path.join(save_path, f), index=False)
        return True
    except Exception as e:
        return e

    
