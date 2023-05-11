import os

SAVED_MODEL_DIR = os.path.join("saved_models")
TARGET_COLUMN = "responseLC50"
PIPELINE_NAME: str = "toxicity"
ARTIFACT_DIR: str = "artifact"


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
DATA_INGESTION_FILE_NAME: str = "toxicitypred.parquet"

'''
Data Validation related constant start with DATA_VALIDATION VAR NAME
'''
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_TRAIN_TEST_SPLIT_RATION: float = 0.20
DATA_VALIDATION_RANDOM_STATE: int = 100

'''
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
'''
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

'''
MODEL TRAINER related constant start with MODEL_TRAINER var name
'''
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.5
MODEL_TRAINER_ML_FLOW_ARTIFACTS_DIR: str = "mlflow_train_artifacts"
MODEL_TRAINER_ML_FLOW_EXP_NAME: str = "LC50_Toxicity_Pred_Regression"
MODEL_TRAINER_ML_FLOW_RUN_NAME: str = "mlflow_tracker"
MODEL_TRAINER_ML_FLOW_REG_MODEL_NAME: str = "LC50ToxicModel"
MODEL_TRAINER_ML_FLOW_REMOTE_SERVER_URI: str = "http://127.0.0.1:8001"


'''
MODEL EVALUATION ralated constant start with MODEL_EVALUATION VAR NAME
'''
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.05
MODEL_EVALUATION_REPORT_NAME= "report.yaml"


'''
Model Pusher ralated constant start with MODEL_PUSHER VAR NAME
'''
MODEL_PUSHER_DIR_NAME = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = SAVED_MODEL_DIR