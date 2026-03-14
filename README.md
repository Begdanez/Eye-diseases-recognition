<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Gradio-4.0%2B-FF7C00?style=for-the-badge&logo=gradio&logoColor=white"/>
<img src="https://img.shields.io/badge/Accuracy-94.6%25-2ea44f?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>

<br/><br/>

# 🔬 Retinal OCT Disease Classifier

**AI-диагностика 8 заболеваний сетчатки по ОКТ-снимкам**  
EfficientNet-B3 · 94.6% точность · Gradio веб-приложение

</div>

---

## 📁 Файлы проекта

```
retinal-oct-classifier/
├── train.py                    # Скрипт обучения модели
├── app.py                      # Веб-приложение для диагностики
├── retinal_oct_training.ipynb  # Kaggle-ноутбук (готов к запуску)
├── requirements.txt            # Зависимости Python
└── README.md                   # Документация
```

---

## 📄 Описание файлов

### `train.py` — Обучение модели

Полный pipeline обучения EfficientNet-B3 на датасете RetinalOCT-C8.

**Что делает:**
- Загружает датасет из папки `RetinalOCT_Dataset/train|val|test`
- Применяет аугментации (crop, flip, rotation, color jitter)
- Обучает в 2 фазы: сначала только голова (5 эпох), потом всё
- Сохраняет лучшую модель по val accuracy
- Early stopping с patience=7
- Выводит classification report и confusion matrix

**Запуск:**
```bash
python train.py --data_dir RetinalOCT_Dataset --epochs 30 --batch_size 32
```

**Аргументы:**

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--data_dir` | `RetinalOCT_Dataset` | Путь к датасету |
| `--output_dir` | `checkpoints` | Куда сохранять модель |
| `--epochs` | `30` | Количество эпох |
| `--batch_size` | `32` | Размер батча |
| `--lr` | `1e-4` | Начальный learning rate |
| `--freeze_epochs` | `5` | Эпох с замороженным backbone |
| `--patience` | `7` | Patience для early stopping |

**Что сохраняется в `checkpoints/`:**
```
checkpoints/
├── best_model.pth       # Веса лучшей модели
├── class_names.json     # Маппинг индекс → имя класса
├── history.json         # Loss и accuracy по эпохам
└── confusion_matrix.npy # Матрица ошибок
```

---

### `app.py` — Веб-приложение

Интерактивный интерфейс для диагностики на базе Gradio.

**Что делает:**
- Загружает обученную модель из `best_model.pth`
- Принимает ОКТ-снимок (JPG, PNG любого размера)
- Показывает диагноз с процентом уверенности
- Рисует гистограмму вероятностей по всем 8 классам
- Выводит медицинское описание патологии
- Ставит флаг ✅ норма / ⚠️ патология

**Запуск:**
```bash
# Стандартный запуск
python app.py --model checkpoints/best_model.pth --classes checkpoints/class_names.json

# Публичная ссылка на 72 часа
python app.py --model checkpoints/best_model.pth --share

# Другой порт
python app.py --model checkpoints/best_model.pth --port 8080
```

**Открыть в браузере:** `http://localhost:7860`

**Аргументы:**

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--model` | `checkpoints/best_model.pth` | Путь к весам |
| `--classes` | `checkpoints/class_names.json` | Путь к именам классов |
| `--port` | `7860` | Порт сервера |
| `--share` | `False` | Создать публичную ссылку |

> Если модель не найдена — приложение запустится с рандомными весами (для проверки интерфейса).

---

### `retinal_oct_training.ipynb` — Kaggle Notebook

Готовый ноутбук для обучения на бесплатном GPU Kaggle.

**Содержит:**

| Ячейка | Что делает |
|--------|-----------|
| 1–3 | Импорты, путь к датасету, конфиг |
| 4 | DataLoaders + превью всех 8 классов |
| 5 | Строит EfficientNet-B3 |
| 6 | Цикл обучения с таблицей по эпохам |
| 7 | Графики loss и accuracy |
| 8–9 | Тест + нормализованная матрица ошибок |
| 10 | Сохраняет артефакты для скачивания |

**Как запустить на Kaggle:**
1. [kaggle.com/code](https://kaggle.com/code) → **New Notebook** → `File` → `Import Notebook`
2. Загрузи `retinal_oct_training.ipynb`
3. **+ Add Data** → найди `retinal-oct-c8` → Add
4. Session options → Accelerator → **GPU T4**
5. **Run All**
6. Вкладка **Output** → скачай `best_model.pth` + `class_names.json`

**Время обучения:** ~38 минут на T4 GPU

---

### `requirements.txt` — Зависимости

```
torch>=2.0.0
torchvision>=0.15.0
gradio>=4.0.0
Pillow>=9.0.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
```

**Установка:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Быстрый старт

### Вариант A — Обучить на Kaggle (рекомендуется)

```
1. Загрузи retinal_oct_training.ipynb на Kaggle
2. Подключи датасет retinal-oct-c8 + включи GPU T4
3. Run All (около 38 минут)
4. Скачай best_model.pth и class_names.json из вкладки Output
5. Положи в папку checkpoints/
6. Запусти: python app.py
```

### Вариант B — Обучить локально

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Скачать датасет с Kaggle и распаковать:
# RetinalOCT_Dataset/
#   train/  val/  test/

# 3. Обучить
python train.py --data_dir RetinalOCT_Dataset --epochs 30

# 4. Запустить приложение
python app.py --model checkpoints/best_model.pth
```

---

## 📊 Результаты

![Training History](assets/training_curves.png)

![Confusion Matrix](assets/confusion_matrix.png)

| Класс | Precision | Recall | F1 |
|-------|-----------|--------|----|
| AMD | 1.00 | 1.00 | 1.00 |
| CNV | 0.93 | 0.91 | 0.92 |
| CSR | 0.97 | 1.00 | 0.98 |
| DME | 0.92 | 0.86 | 0.89 |
| DR | 0.97 | 0.96 | 0.97 |
| Drusen | 0.93 | 0.83 | 0.88 |
| MH | 0.99 | 1.00 | 1.00 |
| Normal | 0.90 | 0.99 | 0.94 |
| **Avg** | **0.95** | **0.94** | **0.95** |

**Test Accuracy: 94.6%** на 2400 изображениях

---

## 🏗 Архитектура

```
EfficientNet-B3 (ImageNet pretrained)
         │
    Global AvgPool → 1536
         │
    Dropout (0.3)
         │
    Linear 1536 → 512 + ReLU
         │
    Dropout (0.2)
         │
    Linear 512 → 8
         │
    Softmax → предсказание
```

**Стратегия обучения:**
- Фаза 1 (эпохи 1–5): backbone заморожен, lr = 1e-4
- Фаза 2 (эпохи 6–30): всё разморожено, lr = 1e-5
- Loss: CrossEntropy + label smoothing 0.1
- Optimizer: AdamW + CosineAnnealingLR

---

## ⚠️ Дисклеймер

Проект создан в образовательных целях.  
Не является заменой профессиональной медицинской диагностики.  
Всегда консультируйтесь с квалифицированным офтальмологом.

---

<div align="center">
<sub>EfficientNet-B3 · RetinalOCT-C8 · PyTorch · Gradio</sub>
</div>
