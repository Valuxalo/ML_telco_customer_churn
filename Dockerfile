FROM python:3.12.3

WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего проекта
COPY . .

# Команда по умолчанию
CMD ["python", "./src/main_pipeline.py"]