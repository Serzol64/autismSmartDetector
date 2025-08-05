## AutismSmartDetector

Проект "AutismSmartDetector" представляет собой инновационную систему на основе искусственного интеллекта, предназначенную для автоматического определения черт аутистического спектра по фотографиям лиц. Система использует свёрточную нейронную сеть (CNN), обученную на большом наборе данных, чтобы классифицировать изображения на две категории: "Autistic" и "Non-Autistic".

### Преимущества для различных отраслей

#### Для врачей и медицинских учреждений:
- **Ранняя диагностика**: Система позволяет врачам быстро и эффективно проводить предварительную оценку пациентов на наличие аутистических черт, что способствует ранней диагностике и своевременному началу лечения.
- **Улучшение качества обслуживания**: Автоматизация процесса диагностики позволяет врачам сосредоточиться на более сложных случаях и уделить больше времени пациентам.
- **Повышение точности**: Использование искусственного интеллекта снижает вероятность ошибок в диагностике, что повышает точность и надежность результатов.

#### Для финансовых организаций:
- **Оценка рисков**: Система может использоваться для оценки рисков при выдаче кредитов или страховых полисов, учитывая особенности поведения и состояния здоровья клиентов.
- **Персонализация услуг**: Финансовые организации могут предлагать персонализированные услуги и продукты, учитывая индивидуальные особенности клиентов.

#### Для цифровых экосистем:
- **Улучшение пользовательского опыта**: Система может быть интегрирована в платформы для улучшения пользовательского опыта, предлагая персонализированные рекомендации и услуги.
- **Анализ поведения пользователей**: Анализ черт аутистического спектра может помочь в понимании поведения пользователей и адаптации интерфейсов и сервисов под их потребности.

#### Для соцсетей:
- **Безопасность и модерация**: Система может использоваться для модерации контента и обеспечения безопасности пользователей, особенно тех, кто может быть уязвим из-за особенностей поведения.
- **Персонализация контента**: Социальные сети могут предлагать персонализированный контент и рекомендации, учитывая индивидуальные особенности пользователей.

#### Для сервисов знакомств:
- **Персонализация рекомендаций**: Система может помочь в подборе партнеров, учитывая индивидуальные особенности и потребности пользователей.
- **Безопасность и защита**: Сервисы знакомств могут использовать систему для защиты пользователей от мошенников и недобросовестных участников.

### Основные возможности
- **Обучение модели**: Обучение модели на основе данных, собранных из различных источников.
- **Предсказание по одному изображению**: Возможность загрузки и анализа одного изображения для определения наличия аутистических черт.
- **Предсказание по множеству изображений**: Анализ нескольких изображений одновременно.
- **Предсказание по URL**: Возможность анализа изображений, загруженных по URL.

### Требования
- Python 3.8+
- PyTorch
- Torchvision
- Pillow
- Requests
- Glob
- Matplotlib

### Установка
Для установки всех необходимых библиотек выполните следующую команду:

```bash
pip install torch torchvision pillow requests matplotlib
```

### Структура проекта
```
.
|-- README.md
|-- best_model.pth
|-- data
|   |-- consolidated
|   |   |-- Autistic
|   |   |-- Non-Autistic
|   |-- train
|   |   |-- Autistic
|   |   |-- Non-Autistic
|   |-- valid
|   |   |-- Autistic
|   |   |-- Non-Autistic
|   |-- test
|   |   |-- Autistic
|   |   |-- Non-Autistic
|-- main.py
```

### Использование
1. **Обучение модели**:
   - Запустите скрипт `main.py` для обучения модели на данных из папки `data`.
   - После завершения обучения модель будет сохранена в файле `best_model.pth`.

2. **Предсказание по одному изображению**:
   - Используйте следующий код для анализа одного изображения:

```python
import torch
from torchvision import transforms
from PIL import Image

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutismClassifier().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Загрузка и предобработка изображения
image_path = "/kaggle/input/autism-image-data/AutismDataset/consolidated/Autistic/0015.jpg"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open(image_path)
image_tensor = transform(image).unsqueeze_(0).to(device)

# Прогон через модель
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probabilities, dim=1)

# Интерпретация результата
classes = ["Non-Autistic", "Autistic"]
print(f"Прогноз для изображения '{image_path}' — {classes[pred_class.item()]}. Вероятности: {probabilities.tolist()[0]} ")
```

3. **Предсказание по множеству изображений**:
   - Используйте следующий код для анализа нескольких изображений:

```python
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Список путей к изображениям
images_paths = glob.glob("/kaggle/input/autism-image-data/AutismDataset/valid/Autistic/*.jpg")

# Преобразование данных
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Пользовательская функция для открытия и преобразования изображений
def load_and_transform(path):
    image = Image.open(path)
    return transform(image)

# Создаем генератор изображений
image_tensors = [load_and_transform(path) for path in images_paths]

# Пакуем в батч
image_batch = torch.stack(image_tensors).to(device)

# Прогон через модель
with torch.no_grad():
    outputs = model(image_batch)
    probabilities = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probabilities, dim=1)

# Выводим прогнозы
for i, path in enumerate(images_paths):
    print(f"Изображение {path}: {classes[preds[i].item()]}, Вероятности: {probabilities[i].tolist()} ")
```

4. **Предсказание по URL**:
   - Используйте следующий код для анализа изображения по URL:

```python
import requests
from io import BytesIO
from torchvision import transforms
from PIL import Image

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutismClassifier().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Загрузка изображения по URL
url = "https://i.pinimg.com/originals/55/b7/93/55b793764f01746f78f54820caa9caeb.jpg"
response = requests.get(url)
image_bytes = response.content

# Преобразуем байты в изображение и далее по схеме
image = Image.open(BytesIO(image_bytes))
image_tensor = transform(image).unsqueeze_(0).to(device)

# Прогон через модель
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probabilities, dim=1)

# Результат
print(f"Прогноз для изображения по URL '{url}' — {classes[pred_class.item()]}. Вероятности: {probabilities.tolist()[0]} ")
```
### Заключение

Проект "AutismSmartDetector" предлагает уникальные возможности для различных отраслей, улучшая качество обслуживания, повышая точность диагностики и обеспечивая персонализированный подход к каждому пользователю.

### Вклад
Вклад в проект приветствуется! Пожалуйста, создайте pull request с вашими изменениями.

### Благодарности
Спасибо всем, кто внес вклад в развитие проекта и предоставил данные для обучения модели.
