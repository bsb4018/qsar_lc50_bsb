import os, sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants.train_constants import SCHEMA_FILE_PATH, VALID_SCHEMA_FILE_PATH
from src.exception import ToxicityException
from src.logger import logging
from src.utils.all_utils import read_yaml_file, write_yaml_file, read_json_file
import warnings
warnings.filterwarnings("ignore")

class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            self._valid_schema = read_json_file(file_path=VALID_SCHEMA_FILE_PATH)
        except Exception as e:
            raise ToxicityException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            raise ToxicityException(e, sys) from e

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        '''
        Takes input a dataframe and returns 'True' 
        if all required columns are present
        '''
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"]) - 1
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise ToxicityException(e, sys) from e

    def is_numerical_column_exist(self, df: DataFrame) -> bool:
        '''
        Takes input a dataframe and returns 'True' if 
        all the designated numerical columns are present
        '''
        try:
            dataframe_columns = df.columns
            status = True
            missing_numerical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    status = False
                    missing_numerical_columns.append(column)
            
            logging.info(f"Missing numerical column: {missing_numerical_columns}")
            return status

        except Exception as e:
            raise ToxicityException(e, sys) from e

    def is_categorical_column_exist(self, df: DataFrame) -> bool:
        '''
        Takes input a dataframe and returns 'True' if 
        all the designated categorical columns are present
        '''
        try:
            dataframe_columns = df.columns
            status = True
            missing_categorical_columns = []
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    status = False
                    missing_categorical_columns.append(column)
            
            logging.info(f"Missing categorical column: {missing_categorical_columns}")
            return status

        except Exception as e:
            raise ToxicityException(e, sys) from e
    
    def drop_unrequired_columns_rows_and_replace_nulls(self, dataframe: DataFrame) -> DataFrame:
        try:
            dataframe = dataframe.drop(self._schema_config["drop_columns"], axis=1)
            drop_row_categories = self._schema_config["drop_categorical_data_rows"]
            for category in drop_row_categories:
                for key,value in category.items():
                    for val in value:
                        dataframe.drop(dataframe.loc[dataframe[key] == val].index, inplace=True)
            dataframe.replace({"na": np.nan}, inplace=True)
            return dataframe

        except Exception as e:
            raise ToxicityException(e, sys)

    def drop_invalid_columns_data(self,df: DataFrame) -> DataFrame:
        try:
            #numerical columns
            cico_min = self._valid_schema["CIC0"]["min"]
            cico_max = self._valid_schema["CIC0"]["max"]
            #cico_val_list = list(df["CIC0"].values)

            gats_min = self._valid_schema["GATS1i"]["min"]
            gats_max = self._valid_schema["GATS1i"]["max"]
            #gats_val_list = list(df["GATS1i"].values)

            mlogp_min = self._valid_schema["MLOGP"]["min"]
            mlogp_max = self._valid_schema["MLOGP"]["max"]
            #mlogp_val_list = list(df["MLOGP"].values)

            smdz_min = self._valid_schema["SM1_DzZ"]["min"]
            smdz_max = self._valid_schema["SM1_DzZ"]["max"]
            #smdz_val_list = list(df["SM1_DzZ"].values)


            #categorical columns
            valid_ndsch_list = self._valid_schema["NdsCH"]
            #ndsch_values_list = list(df['NdsCH'].values)

            valid_ndssc_list = self._valid_schema["NdssC"]
            #ndssc_values_list = list(df['NdsCH'].values)


            dfnew = df[(df['CIC0'] >= cico_min) & (df['CIC0'] <= cico_max) & (df['GATS1i'] >= gats_min) & (df["GATS1i"] <= gats_max) & \
                       (df["MLOGP"] >= mlogp_min) & (df["MLOGP"] <= mlogp_max) & (df["SM1_DzZ"] >= smdz_min) & (df["SM1_DzZ"] <= smdz_max)\
                        (df['NdsCH'].isin(valid_ndsch_list)) & (df['NdssC'].isin(valid_ndssc_list))]
                        
            return dfnew
   
        except Exception as e:
            raise ToxicityException(e, sys) from e
        
    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        
        try:
            logging.info("Entered split_data_as_train_test method of Data_Ingestion class")
            
            #stratify_list
            stratify_categories = self._schema_config["categorical_columns"]        
            
            train_set, test_set = train_test_split(
                dataframe,
                stratify=dataframe[stratify_categories],
                test_size=self.data_validation_config.train_test_split_ratio, 
                random_state=self.data_validation_config.random_state
            )
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            return train_set,test_set

             
            #dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            #os.makedirs(dir_path, exist_ok=True)

            #logging.info(f"Exporting train and test file path.")
            #train_set.to_parquet(
            #    self.data_ingestion_config.training_file_path, engine='pyarrow', index=False
            #)
            #test_set.to_parquet(
            #    self.data_ingestion_config.testing_file_path, engine='pyarrow', index=False
            #)

            #logging.info(f"Exported train and test file path.")
        
        except Exception as e:
            raise ToxicityException(e, sys) from e


    def initiate_data_validation(self) -> DataValidationArtifact:
        '''
        initiates the complete data validation component
        '''
        try:
            logging.info("Starting data validation")

            ingested_dataframe = DataValidation.read_data(self.data_ingestion_artifact.data_store_file_path)
            cleaned_dataframe = self.drop_unrequired_columns_rows_and_replace_nulls(dataframe = ingested_dataframe)

            train_df, test_df = self.split_data_as_train_test(cleaned_dataframe)


            #Validating number of columns
            validation_error_msg = ""
            validation_status = True

            status = self.validate_number_of_columns(dataframe=train_df)
            
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe "
                validation_status = False
            else:
                validation_error_msg += f"No columns are missing in training dataframe "
            
            status = self.validate_number_of_columns(dataframe=test_df)
            
            if not status:
                validation_error_msg += f"Columns are missing in testing dataframe "
                validation_status = False
            else:
                validation_error_msg += f"No columns are missing in testing dataframe "

            logging.info(f"All Columns Validation Message: {validation_error_msg}")
            if not validation_status:
                raise Exception(validation_error_msg)
            
            #Validating numerical columns
            validation_error_msg = ""
            validation_status = True

            status = self.is_numerical_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Numerical columns are missing in training dataframe "
                validation_status = False
            else:
                validation_error_msg += f"No numerical columns are missing in training dataframe "
                

            status = self.is_numerical_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Numerical columns are missing in testing dataframe "
                validation_status = False
            else:
                validation_error_msg += f"No numerical columns are missing in testing dataframe "
                
            
            logging.info(f"Numerical Columns Validation Message: {validation_error_msg}")
            if not validation_status:
                raise Exception(validation_error_msg)

            
            #Validating categorical columns
            validation_error_msg = ""
            validation_status = True

            status = self.is_categorical_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Categorical columns are missing in training dataframe "
                validation_status = False
            else:
                validation_error_msg += f"No categorical columns are missing in training dataframe "

            status = self.is_categorical_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Categorical columns are missing in testing dataframe "
                validation_status = False
            else:
                validation_error_msg += f"No categorical columns are missing in testing dataframe "

            logging.info(f"Categorical Columns Validation Message: {validation_error_msg}")
            if not validation_status:
                raise Exception(validation_error_msg)

            
            dir_path_valid_train = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path_valid_train, exist_ok=True)

            dir_path_valid_test = os.path.dirname(self.data_validation_config.valid_test_file_path)
            os.makedirs(dir_path_valid_test, exist_ok=True)

            #valid_train_df = self.drop_invalid_columns_data(df=train_df)
            #valid_test_df = self.drop_invalid_columns_data(df=test_df)

            valid_train_df = train_df
            valid_test_df = test_df

            valid_train_df.to_parquet(
                self.data_validation_config.valid_train_file_path, engine='pyarrow', index=False
            )
            valid_test_df.to_parquet(
               self.data_validation_config.valid_test_file_path, engine='pyarrow', index=False
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
  
            return data_validation_artifact
            
        except Exception as e:
            raise ToxicityException(e,sys) from e






  