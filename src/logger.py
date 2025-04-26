import logging
import os
from .config import LOGS_DIR

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'pipeline.log'),
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

logger = logging.getLogger(__name__)
