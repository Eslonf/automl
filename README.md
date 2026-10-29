Репозиторий практических работ по дисциплине **«Автоматизация машинного обучения»**.

## Запуск инфраструктуры


### 1. Переход в директорию деплоя
```bash
cd deploy
```


### 2. Настройка окружения для Kaggle

```bash
cp .kaggle.env.example .kaggle.env
```

Откройте файл `.kaggle.env` и укажите свои учетные данные


### 3. Запуск сервисов

```bash
docker compose up -d
```


## Доступ к сервисам

| Сервис | URL (Traefik) | Альтернативный URL (прямой порт) | Логин / Пароль |
| :--- | :--- | :--- | :--- |
| **Apache Airflow** | [http://airflow.localhost](http://airflow.localhost) | [http://localhost:8080](http://localhost:8080) | `admin` / `admin` |
| **MLflow** | [http://mlflow.localhost](http://mlflow.localhost) | [http://localhost:5000](http://localhost:5000) | — |
| **Grafana** | [http://grafana.localhost](http://grafana.localhost) | [http://localhost:3000](http://localhost:3000) | `admin` / `admin` |
| **Prometheus** | [http://prometheus.localhost](http://prometheus.localhost) | [http://localhost:9090](http://localhost:9090) | — |



## Остановка инфраструктуры

```bash
cd deploy
docker compose down
```
