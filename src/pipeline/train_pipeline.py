import sys
from src.components.data_ingestion import DataIngestion
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants.database import TRAINING_BUCKET_NAME
from src.constants.train_constants import SAVED_MODEL_DIR
from src.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from src.exception import ToxicityException
from src.logger import logging,LOG_FILE_PATH
from src.configurations.s3_sync_config import S3Sync


class TrainPipeline:
    is_pipeline_running=False
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            
            logging.info(
              "Entered the start_data_ingestion method of TrainPipeline class"
            )
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            
            return data_ingestion_artifact
    
        except Exception as e:
            
            raise ToxicityException(e, sys) from e