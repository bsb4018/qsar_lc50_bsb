from datetime import datetime
import os
from src.constants.train_constants import *
class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = PIPELINE_NAME
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, timestamp)
        self.timestamp: str = timestamp

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
                training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
        )
        self.data_store_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_DATA_STORE_DIR, DATA_INGESTION_FILE_NAME
        )
        #self.training_file_path: str = os.path.join(
        #    self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME
        #)
        #self.testing_file_path: str = os.path.join(
        #    self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME
        #)
        #self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        #self.random_state: int = DATA_INGESTION_RANDOM_STATE

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, DATA_VALIDATION_VALID_DIR)
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, TEST_FILE_NAME)
        self.train_test_split_ratio: float = DATA_VALIDATION_TRAIN_TEST_SPLIT_RATION
        self.random_state: int = DATA_VALIDATION_RANDOM_STATE


class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join( training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME )
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TRAIN_FILE_NAME.replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TEST_FILE_NAME.replace("csv", "npy"), )
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            PREPROCSSING_OBJECT_FILE_NAME,)

class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, 
            MODEL_FILE_NAME
        )
        self.expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
        self.mlflow_artifact_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_TRAINER_ML_FLOW_ARTIFACTS_DIR
        )
        self.model_trainer_exp_name: str = MODEL_TRAINER_ML_FLOW_EXP_NAME
        self.model_trainer_run_name: str = MODEL_TRAINER_ML_FLOW_RUN_NAME
        self.model_trainer_reg_model_name: str = MODEL_TRAINER_ML_FLOW_REG_MODEL_NAME
        self.model_trainer_server_uri: str = MODEL_TRAINER_ML_FLOW_REMOTE_SERVER_URI


class ModelEvaluationConfig: 
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR_NAME
        )
        self.report_file_path = os.path.join(self.model_evaluation_dir,MODEL_EVALUATION_REPORT_NAME)
        self.change_threshold = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE