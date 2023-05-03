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