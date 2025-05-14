import logging
import sys
# from python_json_logger import jsonlogger # If using JSON logs
from pythonjsonlogger.json import JsonFormatter
from app.core.config import settings

# For plain text logging (simpler to start)
# logging.basicConfig(
#     level=logging.INFO if not settings.DEBUG_MODE else logging.DEBUG,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)]
# )

# For JSON logging
logger = logging.getLogger() # Get root logger
logHandler = logging.StreamHandler(sys.stdout)

class CustomJsonFormatter(JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = log_record.pop('asctime', record.created)
        log_record['level'] = log_record.pop('levelname', record.levelname)
        log_record['module'] = record.module
        log_record['funcName'] = record.funcName
        log_record['lineno'] = record.lineno
        if 'message' not in log_record and message_dict.get('message'):
             log_record['message'] = message_dict['message']


formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(module)s %(funcName)s %(lineno)d %(message)s')
# formatter = jsonlogger.JsonFormatter(
#     '%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)d %(message)s'
# ) # Simpler JSON formatter

logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO if not settings.DEBUG_MODE else logging.DEBUG)

# Example: Disable uvicorn access logs if too noisy, or reformat them
# logging.getLogger("uvicorn.access").disabled = True
# logging.getLogger("uvicorn.error").propagate = False # If you want to handle uvicorn errors yourself

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)