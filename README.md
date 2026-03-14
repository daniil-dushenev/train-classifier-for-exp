# Train Classifier PoC

PoC для эксперимента: как синтетические данные класса `1` влияют на бинарную классификацию изображений.

## Что делает проект

1. Собирает 4 варианта датасета:
- `no_synth`: train на `class0 + class1(real)`
- `synth_0.5x`: тот же train + 0.5× синтетики относительно реальных позитивов
- `synth_1x`: тот же train + 1× синтетики
- `synth_2x`: тот же train + 2× синтетики

2. Обучает ResNet18 и сравнивает метрики на одинаковом test-сете.

3. Поддерживает два режима оценки:
   - **Single split** — обычный train/test, быстро
   - **K-Fold CV** — k-кратная кросс-валидация на `your_data` (MT_Free/MT_Crack всегда только в тесте), результат — mean ± std по фолдам

4. Сохраняет отчеты:
- `metrics.csv` / `metrics.json` — single split
- `metrics_cv.csv` / `metrics_cv.json` — средние по фолдам (при `--n-folds > 1`)
- `metrics_cv_per_fold.csv` / `metrics_cv_per_fold.json` — метрики каждого фолда

## Структура входных данных

```text
your_data/
  class0/        # негативы
  class1/        # реальные позитивы
  class1_synth/  # синтетические позитивы
```

## Скрипты

- `build_datasets.py`
  - Собирает `no_synth` и `synth_{ratio}x` варианты
  - Поддерживает два режима split:
    - классический по `test_ratio`
    - фиксированное число реальных позитивов в train (`--train-pos-count`)
  - Поддерживает дополнительные папки для test-сета (`--extra-test-class0`, `--extra-test-class1`)

- `run_experiment.py`
  - Полный pipeline: сборка датасета + обучение + оценка + сохранение метрик
  - Модель: ResNet18 (pretrained ImageNet по умолчанию)
  - Поддерживает class imbalance через `pos_weight` (авто или вручную)
  - Автоматически отбрасывает битые изображения
  - `--n-folds N` — запустить K-Fold CV (N фолдов на `your_data`, extra-test всегда в тесте)

- `delete_png.py`
  - Удаляет `.png`-дубликаты из папок датасета
  - По умолчанию чистит `class0/` и `class1/` относительно `--root`
  - Поддерживает произвольные папки через `--dirs`
  - `--dry-run` для превью без удаления

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

### С дополнительным test-сетом

```bash
python run_experiment.py \
  --input-root your_data \
  --datasets-root data_out \
  --reports-dir reports \
  --test-ratio 0.2 \
  --epochs 10 \
  --num-workers 0 \
  --device auto \
  --extra-test-class0 ../MT_Free/Imgs \
  --extra-test-class1 ../MT_Crack/Imgs
```

### K-Fold кросс-валидация

`your_data` делится на N фолдов. Для каждого фолда: train = остальные N-1 частей, test = текущий фолд + все MT_Free + все MT_Crack. Синтетика всегда только в train.

```bash
python run_experiment.py \
  --input-root your_data \
  --datasets-root data_out \
  --reports-dir reports \
  --n-folds 5 \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-4 \
  --num-workers 0 \
  --device auto \
  --seed 42 \
  --extra-test-class0 ../MT_Free/Imgs \
  --extra-test-class1 ../MT_Crack/Imgs
```

Результаты: `reports/metrics_cv.json` (mean ± std) и `reports/metrics_cv_per_fold.json`.

### Очистка PNG-дубликатов

```bash
# превью
python delete_png.py --dirs ../MT_Free/Imgs ../MT_Crack/Imgs --dry-run

# удалить
python delete_png.py --dirs ../MT_Free/Imgs ../MT_Crack/Imgs
```

## Метрики

Сохраняются:
- `accuracy`, `precision`, `recall`, `f1_class1`, `specificity`
- `roc_auc`, `pr_auc`
- `tn`, `fp`, `fn`, `tp`

При CV каждая метрика дополняется полем `{metric}_std` (стандартное отклонение по фолдам).

## Выходные артефакты

```text
data_out/                      # собранные train/test выборки (no_synth, synth_0.5x, synth_1x, synth_2x)
reports/
  metrics.csv / metrics.json           # single split
  metrics_cv.csv / metrics_cv.json     # CV: mean ± std
  metrics_cv_per_fold.csv / ...json    # CV: каждый фолд отдельно
```
