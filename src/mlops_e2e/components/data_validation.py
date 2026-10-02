from mlops_e2e.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from mlops_e2e.entity.config_entity import DataValidationConfig
from mlops_e2e.exception.execption import NetworkSecurityException
from mlops_e2e.logging.logger import logging
from mlops_e2e.constants.training_pipeline import SCHEMA_FILE_PATH
from mlops_e2e.utils.main_utils.utils import read_yaml_file

from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import os, sys

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_data(self, df: pd.DataFrame, valid_file_path: str, invalid_file_path: str):
        try:
            number_of_columns = len(self._schema["columns"])
            logging.info(f"Expected number of columns: {number_of_columns}")
            logging.info(f"Actual number of columns: {len(df.columns)}")
            if len(df.columns) != number_of_columns:
                logging.error(f"Expected {number_of_columns} columns, but got {len(df.columns)}")
                return False
            return True

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05):
        try:
            drift_report = {}
            for column in base_df.columns:
                base_data = base_df[column].dropna()
                current_data = current_df[column].dropna()

                if len(base_data) == 0 or len(current_data) == 0:
                    logging.warning(f"Column {column} has no data to compare.")
                    continue

                ks_statistic= ks_2samp(base_data, current_data)
                drift_report[column] = {
                    "p_value": float(ks_statistic.pvalue),
                    "drift_detected": bool(ks_statistic.pvalue <= threshold)
                }

            # Drift report written to a YAML file

            drift_report_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
            with open(drift_report_path, "w") as report_file:
                import yaml
                yaml.dump(drift_report, report_file)
            logging.info(f"Data drift report saved at {drift_report_path}")

            return bool(ks_statistic.pvalue <= threshold)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            # Create directories for valid and invalid data
            os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
            os.makedirs(self.data_validation_config.invalid_data_dir, exist_ok=True)

            # Read the ingested train and test data
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)   
 
            ## Validate the train data
            status_train = self.validate_data(train_df, self.data_validation_config.valid_train_file_path, self.data_validation_config.invalid_train_file_path)
            if not status_train:
                error_message = f"Train dataframe does not match the schema. Invalid train data saved at {self.data_validation_config.invalid_train_file_path}"
                logging.error(error_message)
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
            else:
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
             
            ## Validate the test data
            status_test = self.validate_data(test_df, self.data_validation_config.valid_test_file_path, self.data_validation_config.invalid_test_file_path)
            if not status_test:
                error_message = f"Test dataframe does not match the schema. Invalid test data saved at {self.data_validation_config.invalid_test_file_path}"
                logging.error(error_message)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False) 
            else:
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            # Detect data drift between train and test data
            drift_status = self.detect_data_drift(train_df, test_df)

            data_validation_artifact = DataValidationArtifact(
                validation_status= status_train and status_test and not drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
 