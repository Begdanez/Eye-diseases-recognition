[20.03.2026 22:43] Slavyan: <div align="center">

# 🔬 Retinal OCT Disease Recognition System

### AI-Powered Early Detection of Retinal Diseases from OCT Scans

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-94.6%25-2ea44f?style=for-the-badge)](https://github.com/Begdanez/Eye-diseases-recognition)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[🇬🇧 English](#english-version) · [🇷🇺 Русский](#russian-version)

</div>

-----

<a name="english-version"></a>

# 🇬🇧 English Version

## 📌 Overview

Retinal OCT Disease Recognition System is an AI-powered diagnostic tool that analyzes Optical Coherence Tomography (OCT) retinal scans and automatically classifies them into 8 disease categories with 94.6% accuracy.

Retinal diseases such as AMD, DME, and CNV are among the leading causes of blindness worldwide. Early detection is critical — yet most regions lack enough ophthalmologists to screen all at-risk patients. This system aims to bridge that gap by providing an accessible, fast, and accurate AI assistant for pre-screening retinal OCT images.

> ⚠️ Medical Disclaimer: This system is intended for educational and research purposes only. It does not replace professional medical diagnosis. Always consult a qualified ophthalmologist.

-----

## 🎯 Problem Statement

- Over 2.2 billion people worldwide suffer from vision impairment (WHO, 2023).
- Many serious retinal diseases are asymptomatic in early stages — patients often notice symptoms only when significant damage has already occurred.
- There is a global shortage of ophthalmologists, especially in low- and middle-income countries.
- Manual analysis of OCT scans is time-consuming and subject to human error and fatigue.

Our solution: An AI model that can pre-screen OCT images in under 2 seconds, flagging pathologies for urgent specialist review.

-----

## ✅ Detected Disease Classes

|#|Class     |Description                                                     |
|-|----------|----------------------------------------------------------------|
|1|**Normal**|Healthy retina with no pathological signs                       |
|2|**AMD**   |Age-Related Macular Degeneration — damage to the central retina |
|3|**CNV**   |Choroidal Neovascularization — abnormal blood vessel growth     |
|4|**CSR**   |Central Serous Retinopathy — fluid accumulation under the retina|
|5|**DME**   |Diabetic Macular Edema — swelling due to diabetes               |
|6|**DR**    |Diabetic Retinopathy — vascular damage from diabetes            |
|7|**Drusen**|Drusen deposits — early AMD marker                              |
|8|**MH**    |Macular Hole — full-thickness hole in the macula                |

-----

## 🧠 Model Architecture

The classifier is built on EfficientNet-B3 — a state-of-the-art convolutional neural network pre-trained on ImageNet, fine-tuned on the RetinalOCT-C8 dataset.
EfficientNet-B3 (pretrained on ImageNet)
         │
    Global Average Pooling → 1536 features
         │
    Dropout (p = 0.3)
         │
    Linear: 1536 → 512  +  ReLU
         │
    Dropout (p = 0.2)
         │
    Linear: 512 → 8 classes
         │
    Softmax → Prediction + Confidence

### Training Strategy

|Phase                |Epochs|Learning Rate|What’s Trained                        |
|---------------------|------|-------------|--------------------------------------|
|Phase 1 (Warm-up)    |1 – 5 |1e-4         |Classifier head only (backbone frozen)|
|Phase 2 (Fine-tuning)|6 – 30|1e-5         |Entire network                        |

Key training details:

- Loss function: CrossEntropyLoss + Label Smoothing (0.1)
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
[20.03.2026 22:43] Slavyan: - Early stopping: patience = 7 epochs
- Data augmentation: random crop, horizontal flip, rotation ±15°, color jitter

-----

## 📊 Performance Results

Test Accuracy: 94.6% evaluated on 2,400 held-out images.

|Class           |Precision|Recall  |F1-Score|
|----------------|---------|--------|--------|
|AMD             |1.00     |1.00    |1.00    |
|CNV             |0.93     |0.91    |0.92    |
|CSR             |0.97     |1.00    |0.98    |
|DME             |0.92     |0.86    |0.89    |
|DR              |0.97     |0.96    |0.97    |
|Drusen          |0.93     |0.83    |0.88    |
|MH              |0.99     |1.00    |1.00    |
|Normal          |0.90     |0.99    |0.94    |
|**Weighted Avg**|**0.95** |**0.94**|**0.95**|

-----

## 🖥️ Web Application

The project includes a Gradio-based web interface that requires no coding skills to use:

- 📤 Upload any OCT image (JPG or PNG)
- 🏷️ Instant diagnosis with confidence percentage
- 📊 Probability bar chart across all 8 classes
- 📋 Medical description of the detected condition
- 🚦 Status flag: ✅ Normal / ⚠️ Pathology detected

-----

## 🗂️ Repository Structure
Eye-diseases-recognition/
├── train.py                    # Model training script
├── app.py                      # Gradio web application
├── retinal_oct_training.ipynb  # Kaggle-ready training notebook
├── requirements.txt            # Python dependencies
├── checkpoints/                # Saved model weights & artifacts
│   ├── best_model.pth          # Best model weights
│   ├── class_names.json        # Index → class name mapping
│   ├── history.json            # Loss/accuracy per epoch
│   ├── confusion_matrix.png    # Confusion matrix visualization
│   └── training_curves.png     # Training/validation curves
└── README.md                   # Project documentation

-----

## 🚀 Quick Start

### Option A — Train on Kaggle (Recommended, Free GPU)
# 1. Go to kaggle.com/code → New Notebook → Import Notebook
# 2. Upload retinal_oct_training.ipynb
# 3. Add dataset: retinal-oct-c8
# 4. Enable GPU T4 in Session Options
# 5. Run All  (~38 minutes)
# 6. Download best_model.pth and class_names.json from Output tab
# 7. Place them in the checkpoints/ folder

### Option B — Run Locally
# 1. Clone the repository
git clone https://github.com/Begdanez/Eye-diseases-recognition.git
cd Eye-diseases-recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (requires RetinalOCT-C8 dataset)
python train.py --data_dir RetinalOCT_Dataset --epochs 30 --batch_size 32

# 4. Launch the web application
python app.py --model checkpoints/best_model.pth
# Open in browser: http://localhost:7860

### Option C — Launch App with Pre-trained Weights
# If you already have best_model.pth in checkpoints/:
python app.py --model checkpoints/best_model.pth --share
# --share creates a public link valid for 72 hours

-----

## ⚙️ Training Script Arguments

|Argument         |Default             |Description                |
|-----------------|--------------------|---------------------------|
|`--data_dir`     |`RetinalOCT_Dataset`|Path to dataset            |
|`--output_dir`   |`checkpoints`       |Directory to save model    |
|`--epochs`       |`30`                |Number of training epochs  |
|`--batch_size`   |`32`                |Batch size                 |
|`--lr`           |`1e-4`              |Initial learning rate      |
|`--freeze_epochs`|`5`                 |Epochs with frozen backbone|
|`--patience`     |`7`                 |Early stopping patience    |

-----

## 📦 Requirements
torch>=2.0.0
torchvision>=0.15.0
gradio>=4.0.0
Pillow>=9.0.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0

Install with: pip install -r requirements.txt

-----

## 📚 Dataset

The model is trained on the RetinalOCT-C8 dataset, available on Kaggle:
🔗 [kaggle.com/datasets/obulisainaren/retinal-oct-c8](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8)

- Total images: ~24,000 OCT scans
- Classes: 8 (balanced)
- Split: Train / Validation / Test

-----

## 🌍 Social Impact & Innovation
[20.03.2026 22:43] Slavyan: |Aspect                |Detail                                                         |
|----------------------|---------------------------------------------------------------|
|🎯 Target Benefit  |Faster, earlier detection of retinal disease                   |
|🌐 Global Reach    |Deployable in low-resource settings without specialist hardware|
|⚡ Speed           |Inference in under 2 seconds per scan                          |
|💡 Accessibility   |Simple browser-based UI, no technical expertise required       |
|🔬 **Scientific Rigor**|94.6% accuracy on a 2,400-image test set                       |

-----

<a name="russian-version"></a>

# 🇷🇺 Русская версия

## 📌 Обзор проекта

Система распознавания заболеваний сетчатки по ОКТ-снимкам — это инструмент на основе искусственного интеллекта, который анализирует снимки оптической когерентной томографии (ОКТ) и автоматически классифицирует их по 8 категориям заболеваний с точностью 94,6%.

Заболевания сетчатки — такие как ВМД, ДМО и ХНВ — входят в число ведущих причин слепоты в мире. Ранняя диагностика критически важна, однако во многих регионах не хватает офтальмологов для обследования всех пациентов из группы риска. Данная система призвана восполнить этот пробел, предоставляя доступный, быстрый и точный ИИ-помощник для предварительного скрининга ОКТ-снимков.

> ⚠️ Медицинское предупреждение: Система предназначена исключительно для образовательных и исследовательских целей. Она не заменяет профессиональную медицинскую диагностику. Всегда консультируйтесь с квалифицированным офтальмологом.

-----

## 🎯 Постановка проблемы

- Более 2,2 миллиарда человек в мире страдают от нарушений зрения (ВОЗ, 2023 г.).
- Многие серьёзные заболевания сетчатки протекают бессимптомно на ранних стадиях — пациенты замечают ухудшение зрения уже после значительного повреждения.
- В мире наблюдается острая нехватка офтальмологов, особенно в развивающихся странах.
- Ручной анализ ОКТ-снимков трудоёмок и подвержен человеческим ошибкам.

Наше решение: ИИ-модель, способная предварительно проанализировать ОКТ-снимок менее чем за 2 секунды и выявить признаки патологии для срочной консультации специалиста.

-----

## ✅ Определяемые классы заболеваний

|№|Класс     |Описание                                                               |
|-|----------|-----------------------------------------------------------------------|
|1|**Normal**|Здоровая сетчатка без патологических признаков                         |
|2|**AMD**   |Возрастная макулярная дегенерация — поражение центральной зоны сетчатки|
|3|**CNV**   |Хориоидальная неоваскуляризация — патологический рост новых сосудов    |
|4|**CSR**   |Центральная серозная ретинопатия — скопление жидкости под сетчаткой    |
|5|**DME**   |Диабетический макулярный отёк — отёк вследствие диабета                |
|6|**DR**    |Диабетическая ретинопатия — сосудистые повреждения при диабете         |
|7|**Drusen**|Друзы — отложения, ранний маркёр ВМД                                   |
|8|**MH**    |Макулярное отверстие — сквозной дефект в центре макулы                 |

-----

## 🧠 Архитектура модели

Классификатор построен на базе EfficientNet-B3 — современной свёрточной нейронной сети с предобучением на ImageNet, дообученной на датасете RetinalOCT-C8.
EfficientNet-B3 (предобучена на ImageNet)
         │
    Global Average Pooling → 1536 признаков
         │
    Dropout (p = 0.3)
         │
    Linear: 1536 → 512  +  ReLU
         │
    Dropout (p = 0.2)
         │
    Linear: 512 → 8 классов
         │
    Softmax → Предсказание + Уверенность

### Стратегия обучения

|Фаза               |Эпохи |Learning Rate|Что обучается                                       |
|-------------------|------|-------------|----------------------------------------------------|
|Фаза 1 (разогрев)  |1 – 5 |1e-4         |Только классификационная голова (backbone заморожен)|
|Фаза 2 (дообучение)|6 – 30|1e-5         |Вся сеть целиком                                    |

Ключевые параметры:

- Функция потерь: CrossEntropyLoss + Label Smoothing (0.1)
- Оптимизатор: AdamW
[20.03.2026 22:43] Slavyan: - Планировщик: CosineAnnealingLR
- Ранняя остановка: patience = 7 эпох
- Аугментация данных: случайная обрезка, горизонтальный флип, поворот ±15°, изменение яркости/контраста

-----

## 📊 Результаты

Точность на тестовой выборке: 94,6% (2 400 изображений)

|Класс               |Precision|Recall  |F1-Score|
|--------------------|---------|--------|--------|
|AMD                 |1.00     |1.00    |1.00    |
|CNV                 |0.93     |0.91    |0.92    |
|CSR                 |0.97     |1.00    |0.98    |
|DME                 |0.92     |0.86    |0.89    |
|DR                  |0.97     |0.96    |0.97    |
|Drusen              |0.93     |0.83    |0.88    |
|MH                  |0.99     |1.00    |1.00    |
|Normal              |0.90     |0.99    |0.94    |
|**Средневзвешенное**|**0.95** |**0.94**|**0.95**|

-----

## 🖥️ Веб-приложение

Проект включает веб-интерфейс на базе Gradio, не требующий навыков программирования:

- 📤 Загрузите любой ОКТ-снимок (JPG или PNG)
- 🏷️ Мгновенный диагноз с процентом уверенности
- 📊 Столбчатая диаграмма вероятностей по всем 8 классам
- 📋 Медицинское описание выявленной патологии
- 🚦 Статус: ✅ Норма / ⚠️ Обнаружена патология

-----

## 🗂️ Структура репозитория
Eye-diseases-recognition/
├── train.py                    # Скрипт обучения модели
├── app.py                      # Веб-приложение Gradio
├── retinal_oct_training.ipynb  # Ноутбук для Kaggle (готов к запуску)
├── requirements.txt            # Python-зависимости
├── checkpoints/                # Сохранённые веса модели
│   ├── best_model.pth          # Веса лучшей модели
│   ├── class_names.json        # Маппинг индекс → имя класса
│   ├── history.json            # Loss и accuracy по эпохам
│   ├── confusion_matrix.png    # Матрица ошибок
│   └── training_curves.png     # Кривые обучения
└── README.md                   # Документация проекта

-----

## 🚀 Быстрый старт

### Вариант A — Обучение на Kaggle (рекомендуется, бесплатный GPU)
# 1. Перейдите на kaggle.com/code → New Notebook → Import Notebook
# 2. Загрузите retinal_oct_training.ipynb
# 3. Добавьте датасет: retinal-oct-c8
# 4. В Session Options включите GPU T4
# 5. Нажмите Run All (~38 минут)
# 6. Скачайте best_model.pth и class_names.json из вкладки Output
# 7. Поместите их в папку checkpoints/

### Вариант Б — Локальный запуск
# 1. Клонируйте репозиторий
git clone https://github.com/Begdanez/Eye-diseases-recognition.git
cd Eye-diseases-recognition

# 2. Установите зависимости
pip install -r requirements.txt

# 3. Обучите модель (требуется датасет RetinalOCT-C8)
python train.py --data_dir RetinalOCT_Dataset --epochs 30 --batch_size 32

# 4. Запустите веб-приложение
python app.py --model checkpoints/best_model.pth
# Откройте в браузере: http://localhost:7860

### Вариант В — Запуск с готовыми весами
# Если best_model.pth уже в папке checkpoints/:
python app.py --model checkpoints/best_model.pth --share
# --share создаёт публичную ссылку на 72 часа

-----

## ⚙️ Аргументы скрипта обучения

|Аргумент         |По умолчанию        |Описание                     |
|-----------------|--------------------|-----------------------------|
|`--data_dir`     |`RetinalOCT_Dataset`|Путь к датасету              |
|`--output_dir`   |`checkpoints`       |Папка для сохранения модели  |
|`--epochs`       |`30`                |Количество эпох обучения     |
|`--batch_size`   |`32`                |Размер батча                 |
|`--lr`           |`1e-4`              |Начальный learning rate      |
|`--freeze_epochs`|`5`                 |Эпох с замороженным backbone |
|`--patience`     |`7`                 |Patience для ранней остановки|

-----

## 📦 Зависимости
torch>=2.0.0
torchvision>=0.15.0
gradio>=4.0.0
Pillow>=9.0.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0

Установка: pip install -r requirements.txt

-----

## 📚 Датасет

Модель обучена на датасете RetinalOCT-C8, доступном на Kaggle:
🔗 [kaggle.com/datasets/obulisainaren/retinal-oct-c8](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8)

- Всего изображений: ~24 000 ОКТ-снимков
- Классов: 8 (сбалансированные)
[20.03.2026 22:43] Slavyan: - Разбивка: Обучение / Валидация / Тест

-----

## 🌍 Социальная значимость и инновационность

|Аспект                  |Описание                                                    |
|------------------------|------------------------------------------------------------|
|🎯 Польза            |Более ранняя и быстрая диагностика заболеваний сетчатки     |
|🌐 **Масштаб применения**|Подходит для регионов с ограниченным доступом к специалистам|
|⚡ Скорость          |Анализ одного снимка — менее 2 секунд                       |
|💡 Доступность       |Простой браузерный интерфейс без технических навыков        |
|🔬 Научная строгость |Точность 94,6% на тестовой выборке из 2 400 снимков         |

-----

<div align="center">

EfficientNet-B3 · RetinalOCT-C8 · PyTorch · Gradio

*Разработано в образовательных целях · Developed for educational purposes*

</div>
