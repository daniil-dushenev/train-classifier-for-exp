# Train Classifier PoC

PoC для эксперимента: как синтетические данные класса `1` влияют на бинарную классификацию изображений.

## Что делает проект

1. Собирает два варианта датасета:
- `no_synth`: train на `class0 + class1(real)`
- `with_synth`: тот же train + `class1_synth`

2. Обучает классификатор и сравнивает метрики на одинаковом test.

3. Сохраняет отчеты:
- `metrics.csv`
- `metrics.json`
- `metrics.md` (markdown-таблица)

## Структура входных данных

```text
your_data/
  class0/        # негативы
  class1/        # реальные позитивы
  class1_synth/  # синтетические позитивы
```

## Скрипты

- `build_datasets.py`
  - Сборка `no_synth` / `with_synth`
  - Поддерживает два режима:
    - классический split по `test_ratio`
    - фиксированное число реальных позитивов в train (`--train-pos-count`)

- `run_experiment.py`
  - Полный pipeline: сборка датасета + обучение + оценка + сохранение метрик
  - Сейчас используется `resnet18`
  - Поддерживает class imbalance через `pos_weight` (авто или вручную)
  - Автоматически отбрасывает битые изображения

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Базовый запуск (split по ratio)

```bash
python run_experiment.py \
  --input-root your_data \
  --datasets-root data_out \
  --reports-dir reports \
  --test-ratio 0.2 \
  --epochs 3 \
  --num-workers 0 \
  --device cpu
```

### Запуск для сценария с фиксированными 50 позитивами

Пример конфигурации:
- train: `class0 = 70%`, `class1(real) = 50`
- test: `class0 = 30%`, `class1(real) = остальные`
- `class1_synth`: все в train только для `with_synth`

```bash
python run_experiment.py \
  --input-root datasets_med \
  --datasets-root data_out_med \
  --reports-dir reports_med \
  --train-pos-count 50 \
  --test-ratio 0.3 \
  --epochs 3 \
  --num-workers 0 \
  --device cpu \
  --no-pretrained \
  --seed 42
```

## Метрики

Сохраняются:
- `accuracy`
- `precision`
- `recall`
- `f1_class1`
- `specificity`
- `roc_auc`
- `pr_auc`
- `tn`, `fp`, `fn`, `tp`

## Выходные артефакты

```text
data_out*/, data_out_med*/         # собранные train/test выборки
reports*/                          # metrics.csv, metrics.json, metrics.md
```
