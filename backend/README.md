# Superstore ML API

REST API для прогнозування відповіді клієнта на маркетингову кампанію.  
Побудовано на **FastAPI** + **scikit-learn** + **PostgreSQL** (Supabase).

---

## Швидкий старт

### 1. Встановити залежності

```bash
cd backend
pip install -r requirements.txt
```

### 2. Налаштувати `.env`

```ini
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

### 3. Запустити міграції

```bash
# Створити нову міграцію (після зміни моделей)
alembic revision --autogenerate -m "add feature and prediction tables"

# Застосувати
alembic upgrade head
```

### 4. Заповнити дані

```bash
# Сирі дані (customers, education_levels, marital_statuses)
python scripts/seed_data.py --csv ../ml_project/superstore_data.csv

# Препроцесовані фічі (customer_features)
python scripts/seed_features.py
```

### 5. Запустити сервер

```bash
uvicorn app.main:app --reload
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Ендпоінти

### `POST /api/v1/train-model`

Тренує модель на даних з БД.

```bash
curl -X POST http://localhost:8000/api/v1/train-model
```

**Відповідь:**

```json
{
  "train_size": 1582,
  "test_size": 396,
  "accuracy": 0.8712,
  "balanced_accuracy": 0.5841,
  "precision": 0.5432,
  "recall": 0.6154,
  "f1_score": 0.5769,
  "roc_auc": 0.8923,
  "pr_auc": 0.6511,
  "mcc": 0.5123,
  "true_negatives": 315,
  "false_positives": 42,
  "false_negatives": 21,
  "true_positives": 18,
  "model_version": "logistic_regression_weighted_20260408_163000",
  "message": "Model successfully trained and artifacts saved."
}
```

### `POST /api/v1/predict`

Приймає сирі дані клієнта (без `Id` та `Response`), прогнозує відповідь.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "year_birth": 1970,
    "education": "Graduation",
    "marital_status": "Single",
    "income": 58138,
    "kidhome": 0,
    "teenhome": 0,
    "dt_customer": "6/16/2014",
    "recency": 58,
    "mnt_wines": 635,
    "mnt_fruits": 88,
    "mnt_meat_products": 546,
    "mnt_fish_products": 172,
    "mnt_sweet_products": 88,
    "mnt_gold_prods": 88,
    "num_deals_purchases": 3,
    "num_web_purchases": 8,
    "num_catalog_purchases": 10,
    "num_store_purchases": 4,
    "num_web_visits_month": 7,
    "complain": 0
  }'
```

**Відповідь:**

```json
{
  "prediction": 1,
  "prediction_proba": 0.7312,
  "model_version": "logistic_regression_weighted_20260408_163000"
}
```

---

## Структура БД

### `customers` — сирі дані
Оригінальні дані з CSV. FK на таблиці `education_levels` та `marital_statuses`.

### `customer_features` — препроцесовані фічі
1:1 зв'язок з `customers` через `customer_id`. Містить усі фічі після feature engineering (log-трансформації, ратіо, Age, is_alone тощо).

### `training_results` — метрики тренування
Зберігає результати кожного тренувального циклу моделі, включаючи `accuracy`, `balanced_accuracy`, `roc_auc`, `pr_auc`, `mcc` та інше.

### `predictions` — результати прогнозування
| Колонка | Опис |
|---|---|
| prediction | Передбачений клас (0/1) |
| prediction_proba | Ймовірність класу 1 |
| source | `"train"` — при тренуванні, `"inference"` — при прогнозі |
| model_version | Версія моделі (назва + timestamp) |

### `inference_inputs` — вхідні дані запиту
Зберігає повний JSON-запит від користувача. Зв'язок 1:1 з `predictions` через `prediction_id`.

---

## Приклади збережених передбачень

При виклику `/predict`, у БД записується наступне:

**Таблиця `predictions`:**
| id | prediction | prediction_proba | source | model_version | created_at |
|----|------------|------------------|-----------|----------------------------------|---------------------|
| 42 | 1 | 0.7312 | inference | logistic_regression_20260408 | 2026-04-13 10:42:01 |

**Таблиця `inference_inputs` (для prediction_id=42):**
| id | prediction_id | year_birth | education | marital_status | income | ... |
|----|---------------|------------|------------|----------------|---------|-----|
| 10 | 42 | 1970 | Graduation | Single | 58138.0 | ... |

---

## Логіка розподілу даних

- Сирі дані зчитуються з таблиці `customers`
- Очищення: дедублікація, видалення NaN, невалідних Marital_Status, аутлаєрів (IQR)
- **Спочатку** розбиття 80% train / 20% test (стратифіковано по `Response`)
- **Потім** feature engineering застосовується окремо до кожної частини
- Скейлер (`StandardScaler`) тренується **тільки** на train-даних (всередині sklearn Pipeline)

## Збережені моделі

Моделі зберігаються у папці `models/` у форматі `.joblib`.  
Кожна модель — це повний sklearn Pipeline (StandardScaler + LogisticRegression).  
Назва: `logistic_regression_weighted_{YYYYMMDD_HHMMSS}.joblib`

---

## Структура проєкту

```
backend/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── database.py          # SQLAlchemy setup
│   ├── core/config.py       # Pydantic Settings
│   ├── api/
│   │   ├── deps.py          # Dependency injection
│   │   └── v1/
│   │       ├── router.py    # v1 router aggregator
│   │       └── endpoints/
│   │           ├── training.py
│   │           └── inference.py
│   ├── models/              # SQLAlchemy ORM models
│   ├── schemas/             # Pydantic request/response
│   ├── services/            # Business logic
│   ├── ml/                  # Pipeline + model registry
│   └── utils/
├── models/                  # Trained .joblib files
├── scripts/                 # Seed scripts
├── alembic/                 # DB migrations
└── requirements.txt
```
