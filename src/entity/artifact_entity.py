from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    data_store_file_path: str
    #trained_file_path: str
    #test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class RegressionMetricArtifact:
    rmse_value: float
    r2_score: float
    mae_value: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: RegressionMetricArtifact
    test_metric_artifact: RegressionMetricArtifact

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    train_model_metric_artifact: RegressionMetricArtifact
    best_model_metric_artifact: RegressionMetricArtifact