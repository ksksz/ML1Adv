# Проект ML1Adv: Прогнозирование выживаемости пациентов с циррозом

## Автор

- ФИО: Ксения Захарова
- Группа: 972403

## Цель Проекта

Репозиторий содержит решение лабораторной работы по задаче прогнозирования статуса выживаемости пациентов с циррозом печени. В проект входят:

- baseline-модель `RandomForest` для сравнения
- финальная модель `CatBoostClassifier` с подобранными гиперпараметрами
- поддержка подбора гиперпараметров через `Optuna`
- production-style CLI в [model.py](/Users/kseniazaharova/Desktop/ML1Adv/model.py)
- логирование в `./data/log_file.log`
- ноутбук с экспериментами и созданием submission-файла в [notebooks/homework_1_solution.ipynb](/Users/kseniazaharova/Desktop/ML1Adv/notebooks/homework_1_solution.ipynb)

## Текущие локальные метрики

- `log_loss` baseline-модели на 5-fold CV: `0.431983`
- `log_loss` финальной CatBoost-модели на 5-fold CV: `0.376121`

## Подтверждение Результата На Kaggle

- Финальный результат на Kaggle: `0.37874`
- Требование задания `log_loss < 0.55` выполнено
- Подтверждение результата приложено в проект: [kaggle_score_confirmation.png](/Users/kseniazaharova/Desktop/ML1Adv/docs/kaggle/kaggle_score_confirmation.png)

![Подтверждение оценки Kaggle](/Users/kseniazaharova/Desktop/ML1Adv/docs/kaggle/kaggle_score_confirmation.png)

## Структура репозитория

- [model.py](/Users/kseniazaharova/Desktop/ML1Adv/model.py): основной артефакт проекта с классом `My_Classifier_Model` и методами `train` и `predict`
- [ml1adv_cirrhosis/pipeline.py](/Users/kseniazaharova/Desktop/ML1Adv/ml1adv_cirrhosis/pipeline.py): подготовка данных, оценка baseline, кросс-валидация CatBoost, поиск через Optuna
- [ml1adv_cirrhosis/logging_utils.py](/Users/kseniazaharova/Desktop/ML1Adv/ml1adv_cirrhosis/logging_utils.py): настройка singleton-логгера
- [ml1adv_cirrhosis/tracking.py](/Users/kseniazaharova/Desktop/ML1Adv/ml1adv_cirrhosis/tracking.py): опциональная интеграция с ClearML
- [notebooks/homework_1_solution.ipynb](/Users/kseniazaharova/Desktop/ML1Adv/notebooks/homework_1_solution.ipynb): финальный ноутбук с Optuna и генерацией submission-файла
- [pyproject.toml](/Users/kseniazaharova/Desktop/ML1Adv/pyproject.toml): управление зависимостями через Poetry

## Запуск без Docker

1. Установить зависимости:

```bash
poetry install
```

2. Обучить финальную модель:

```bash
poetry run python model.py train --dataset=/абсолютный/путь/до/train.csv
```

3. При необходимости повторно обучить модель с подбором гиперпараметров через Optuna:

```bash
poetry run python model.py train --dataset=/абсолютный/путь/до/train.csv --optuna-trials=12
```

4. Получить предсказания для тестового набора:

```bash
poetry run python model.py predict --dataset=/абсолютный/путь/до/test.csv
```

5. Проверить созданные артефакты:

- обученная модель: `./model/catboost_model.cbm`
- метаданные: `./model/metadata.json`
- сводка по обучению: `./model/training_summary.json`
- история Optuna: `./model/optuna_study.json`, если обучение запускалось с `--optuna-trials`
- предсказания: `./data/results.csv`
- лог-файл: `./data/log_file.log`

## Запуск с Docker

1. Собрать Docker-образ:

```bash
docker build -t ml1adv-cirrhosis .
```

2. Обучить модель:

```bash
docker run --rm \
  -v "$(pwd)/model:/app/model" \
  -v "$(pwd)/data:/app/data" \
  -v /абсолютный/путь/до/датасетов:/datasets \
  ml1adv-cirrhosis train --dataset=/datasets/train.csv
```

3. Запустить инференс:

```bash
docker run --rm \
  -v "$(pwd)/model:/app/model" \
  -v "$(pwd)/data:/app/data" \
  -v /абсолютный/путь/до/датасетов:/datasets \
  ml1adv-cirrhosis predict --dataset=/datasets/test.csv
```

## Замечания по оценке и развертыванию

- Целевая метрика: многоклассовый `log_loss`
- Baseline-модель оценивается с помощью 5-fold stratified cross-validation
- Финальная CatBoost-модель обучается на полном тренировочном наборе и сохраняется для повторного использования
- Формат выходного файла соответствует требованиям Kaggle: `id`, `Status_C`, `Status_CL`, `Status_D`

## Опциональный трекинг через ClearML

- Зависимость уже добавлена в Poetry.
- Чтобы включить логирование экспериментов, нужно настроить доступ к своему ClearML-серверу и запустить:

```bash
ENABLE_CLEARML=1 poetry run python model.py train --dataset=/абсолютный/путь/до/train.csv --optuna-trials=12
```

- При включенном ClearML команда обучения логирует гиперпараметры, метрики, артефакт модели, итоговую сводку обучения и snapshot датасета

## Использованные ресурсы

- датасет из Kaggle Playground Series S3E26
- документация CatBoost
- документация Optuna
- документация scikit-learn
- документация Poetry

## Ограничения

- Kaggle submission уже выполнен, а подтверждение итоговой оценки добавлено в репозиторий.
- Скриншоты ClearML и удаленный хостинг экспериментов требуют запущенного ClearML-инстанса, который по умолчанию не входит в это локальное окружение.
