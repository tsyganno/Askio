import logging
from os import path, makedirs

logging.getLogger("aiogram.event").propagate = False
logging.getLogger("aiogram").propagate = False

# Путь к директории, где находится этот файл
work_dir = path.dirname(path.abspath(__file__))

# Подняться на уровень выше
project_root = path.abspath(path.join(work_dir, "../.."))

# Путь к лог-файлу в папке logs в корне проекта
log_dir = path.join(project_root, "logs")
log_file_path = path.join(log_dir, "prod.log")

# Создать папку logs, если не существует
makedirs(log_dir, exist_ok=True)

# Открываем файл для записи (w+ или a+ — по необходимости)
out = open(log_file_path, 'w+')

# Настройка базового логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание обработчика для записи в файл
file_handler = logging.StreamHandler(out)
file_handler.setLevel(logging.INFO)

# Форматирование логов
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавляем обработчик к логгеру
logger.addHandler(file_handler)

# Тест
logger.info("Логирование успешно настроено!")
logger.info("Сервис работает!")
