import os
import sys

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.data.toxicity_data import ToxicityData
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception import ToxicityException
from src.logger import logging
from src.utils.all_utils import read_yaml_file
from src.constants.train_constants import SCHEMA_FILE_PATH
import warnings
warnings.filterwarnings("ignore")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise ToxicityException(e,sys)

    def load_data_from_data_store(self) -> DataFrame:
        try:
            logging.info("Exporting data from astra cassandra database")
            toxic_data = ToxicityData()
            #dataframe = toxic_data.export_from_astra_database_to_dataframe_using_driver()
            dataframe = toxic_data.export_from_astra_database_to_dataframe_using_restapi()
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            data_store_file_path = self.data_ingestion_config.data_store_file_path
            dir_path = os.path.dirname(data_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(
                f"Saving exported data into feature store file path: {data_store_file_path}"
            )
            dataframe.to_parquet(data_store_file_path, engine='pyarrow', index=False)
            return dataframe

        except Exception as e:
            raise ToxicityException(e, sys)
    
    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        
        try:
            logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

            train_set, test_set = train_test_split(
                dataframe, 
                test_size=self.data_ingestion_config.train_test_split_ratio, 
                random_state=self.data_ingestion_config.random_state
            )
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")
            train_set.to_parquet(
                self.data_ingestion_config.training_file_path, engine='pyarrow', index=False
            )
            test_set.to_parquet(
                self.data_ingestion_config.testing_file_path, engine='pyarrow', index=False
            )

            logging.info(f"Exported train and test file path.")
        
        except Exception as e:
            raise ToxicityException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        try:
            logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
            
            dataframe = self.load_data_from_data_store()
            dataframe = dataframe.drop(self._schema_config["drop_columns"], axis=1)
            dataframe.replace({"na": np.nan}, inplace=True)

            logging.info("Got the data from AstraDB")
            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")
            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise ToxicityException(e, sys) from e
