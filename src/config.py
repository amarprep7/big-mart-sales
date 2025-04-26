import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_v9rqX0R.csv')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_AbJTz2l.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.joblib')

RANDOM_STATE = 42
