import logging
import os
from datetime import datetime

date_format = "%d-%m-%Y-%H-%M-%S"
LOG_FILE = f"{datetime.now().strftime(date_format)}.log"
logs_path = "artifacts/logs"
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s\n %(message)s",
    level=logging.INFO,
)
