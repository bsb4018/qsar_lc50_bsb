from datetime import datetime
import os
from src.constants import train_constants

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = train_constants.PIPELINE_NAME
        self.artifact_dir: str = os.path.join(train_constants.ARTIFACT_DIR, timestamp)
        self.timestamp: str = timestamp

class DataIngestionConfig:
    def __init__(self,train_constants_config:TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
                train_constants_config.artifact_dir, train_constants.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, train_constants.DATA_INGESTION_FEATURE_STORE_DIR, train_constants.FILE_NAME
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, train_constants.DATA_INGESTION_INGESTED_DIR, train_constants.TRAIN_FILE_NAME
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, train_constants.DATA_INGESTION_INGESTED_DIR, train_constants.TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = train_constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.random_state: int = train_constants.DATA_INGESTION_RANDOM_STATE