from load_data import load_data
from ml_model import train_model
from predict import predict
from save_model import save_model
from preprocessing import process_all_files

class MLPipeline:
    def __init__(self):
        self.model = None
        self.data = None
        self.train = None
        self.val = None
        self.test = None
        self.raw_folder = 'raw'
        self.processed_folder = 'processed'
        self.name_model = 'CatBoost'
    def run_full_pipeline(self):
        
        self.data = load_data(load_path=None, raw_folder=self.raw_folder)
        print('загрузили данные')
        self.train, self.val, self.test = process_all_files(self.raw_folder, self.processed_folder)
        print('обработали данные')
        self.model = train_model(self.train, self.val)
        print('обучили модель')
        if self.model is not None:
            predict(self.model, self.test)
            print('предсказали на тесте')
        save_model(model=self.model, name=self.name_model)
        print('сохранили модель')    
        print("✅ Pipeline выполнен успешно!")
    

# Использование одной командой
if __name__ == "__main__":
    pipeline = MLPipeline()
    model = pipeline.run_full_pipeline()