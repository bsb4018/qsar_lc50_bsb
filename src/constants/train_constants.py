import os

SAVED_MODEL_DIR = os.path.join("saved_models")
TARGET_COLUMN = "responseLC50"
PIPELINE_NAME: str = "toxicity"
ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "toxicitypred.parquet"


'''
Defining basic and common file names
'''
TRAIN_FILE_NAME: str = "train.parquet"
TEST_FILE_NAME: str = "test.parquet"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("src","configurations", "schema.yaml")
VALID_SCHEMA_FILE_PATH = os.path.join("src","configurations", "validate.json")
SCHEMA_DROP_COLS = "drop_columns"

'''
Data Ingestion related constant start with DATA_INGESTION VAR NAME
'''
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_DATA_STORE_DIR: str = "data_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2
DATA_INGESTION_RANDOM_STATE: int = 42
