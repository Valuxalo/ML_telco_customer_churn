from load_data import load_data
from ml_model import train_model
from predict import predict
from save_model import save_model
from preprocessing import process_all_files
import mlflow
import os
from dotenv import load_dotenv
from pathlib import Path
import shutil
import requests
from mlflow.tracking import MlflowClient

load_dotenv()


REMOTE_URI = "http://127.0.0.1:5000"
LOCAL_URI = "./mlruns"

def setup_mlflow():
    try:
        # Проверяем доступность сервера
        response = requests.get(f"{REMOTE_URI}/health", timeout=2)

        if response.status_code == 200:
            mlflow.set_tracking_uri(REMOTE_URI)

            # дополнительная проверка API
            client = MlflowClient()
            client.search_experiments()

            print(f"MLflow server найден: {REMOTE_URI}")
            return "remote"

    except Exception as e:
        print(f"MLflow server недоступен: {e}")

    # fallback на локальный storage
    mlflow.set_tracking_uri(LOCAL_URI)

    print("Используется локальный MLflow")
    return "local"

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
        mode = setup_mlflow()
        mlruns_path = Path("./mlruns")
        if mlruns_path.exists():
            shutil.rmtree(mlruns_path)

        mlflow.set_experiment("telco-churn-exp")
        mlflow.enable_system_metrics_logging()

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