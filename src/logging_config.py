import logging
import sys
from datetime import datetime

from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        # Mapeia os campos para os nomes que queremos no JSON
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        log_record['level'] = record.levelname
        log_record['service'] = 'fiap-mobile-churn'
        log_record['logger'] = record.name
        
        if not log_record.get('message'):
            log_record['message'] = record.getMessage()

def setup_logging():
    logger = logging.getLogger("api")
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)    
    
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s',
        json_ensure_ascii=False
    )    
    handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
    
    logger.propagate = False
    return logger

logger = setup_logging()