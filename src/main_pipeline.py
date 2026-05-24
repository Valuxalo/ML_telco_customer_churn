from load_data import load_data
from ml_model import train_model
from predict import predict
from save_model import save_model
from preprocessing import process_all_files
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

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
        self.model_name = os.getenv("MODEL_NAME")
    def run_full_pipeline(self):
        mlflow.set_experiment("telco-churn-exp")
        mlflow.enable_system_metrics_logging()
        mlflow.set_tracking_uri("file:./mlruns")
        print('Выбрана модель:', self.model_name)
        with mlflow.start_run(run_name=self.model_name):
            self.data = load_data(load_path=None, raw_folder=self.raw_folder)
            if self.data:
                self.train, self.val, self.test = process_all_files(self.raw_folder, self.processed_folder)
                self.model = train_model(self.train, self.val)
                if self.model is not None:
                    predict(self.model, self.test)
                save_model(model=self.model, name=self.name_model)
                print("Pipeline выполнен успешно!")
    

# Использование одной командой
if __name__ == "__main__":
    pipeline = MLPipeline()
    model = pipeline.run_full_pipeline()